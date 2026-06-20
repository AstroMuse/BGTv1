# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
OrchestratorAgent — BFS Tree Pipeline Coordinator.

Responsibilities:
  - Own the SearchNode tree and BFS search queue.
  - Drive the three-phase loop (per tree depth):
      Phase 1 — query-level retrieval + scoring  (RetrievalAgent + ScoringAgent)
      Phase 2 — reference expansion + scoring    (ReferenceAgent + ScoringAgent)
      Phase 3 — context-driven query generation  (QueryAgent)
  - Decide when to stop (depth or doc-count threshold).
  - Collect, diversify, and rank the final result set.
  - Visualize the search tree with Graphviz.

Public interface:
  search(initial_query, end_date) -> Dict[paper_id, doc]
  visualize_tree(filename, save_format, view)
"""

import json
import math
import random
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import tqdm
from graphviz import Digraph

from global_config import (
    SAVE_ID2DOCS,
    DO_REFERENCE_SEARCH,
    DO_PHASE3,
    ENABLE_EARLY_STOP,
    DOCS_TO_EXPAND,
    REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY,
    REFERENCE_PRUNE_STRATEGY,
    REFERENCE_PRUNE_TOPN,
    QUERY_NUM_PRUNED,
    SEARCH_ROUTES,
    ENABLE_RERANK,
    RERANK_CITATION_WEIGHT,
    PHASE1_NODE_PARALLEL,
    AGENT_B_ENABLED,
    AGENT_C_ENABLED,
)
from local_db_v2 import db_path, ArxivDatabase
from log import logger
from search_node import SearchNode
from agents.query_agent import QueryAgent
from agents.retrieval_agent import RetrievalAgent
from agents.scoring_agent import ScoringAgent
from agents.reference_agent import ReferenceAgent
from agents.recall_agent import RecallAgent
from agents.constraint_agent import ConstraintAgent
from rerank import Reranker


class OrchestratorAgent:
    """
    Coordinates the 4 specialist agents across BFS tree depths.
    Replaces the former AcademicSearchTree with clear agent delegation.
    """

    def __init__(
        self,
        max_depth: int = 1,
        max_docs: int = 50,
        similarity_threshold: float = 0.6,
    ):
        self.max_depth = max_depth
        self.max_docs = max_docs
        self.sim_threshold = similarity_threshold
        self.high_score_thresh = 0.75

        # Specialist agents
        self.query_agent = QueryAgent()
        self.retrieval_agent = RetrievalAgent()
        self.scoring_agent = ScoringAgent()
        self.reference_agent = ReferenceAgent()
        # Parallel candidate agents (gated; run beside main retrieval at depth 1)
        self.recall_agent = RecallAgent()          # Agent B
        self.constraint_agent = ConstraintAgent()  # Agent C
        self.reranker = Reranker()
        self._bge_reranker = None  # lazy: only built when ENABLE_RERANK

        # Search state (reset each call to search())
        self.root: SearchNode = SearchNode()
        self.user_query: str = ""
        self.search_date: str = ""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, initial_query: str, end_date: str = "") -> Dict:
        """
        Execute the full BFS tree search and return ranked relevant documents.
        """
        search_start = time.time()
        self.search_date = end_date
        self.root = SearchNode(query_str=initial_query, status="INIT")
        self.user_query = initial_query

        logger.info("Initializing SPAR search tree")
        # Per-phase wall-clock timings (seconds), accumulated across tree depths.
        # Persisted into the result JSON via SearchNode.convert_to_dict().
        timing = {
            "phase0_query_expansion": 0.0,
            "phase1_retrieval_scoring": 0.0,
            "phase2_reference": 0.0,
            "phase3_context_query": 0.0,
            "collect_results": 0.0,
            "search_total": 0.0,
        }
        self.root.extra["timing"] = timing  # set early so it survives an abort
        try:
            # --- Phase 0: initial query expansion ---
            _t = time.time()
            expanded_queries, query_node_relations = self._phase_query_fusion()
            timing["phase0_query_expansion"] += time.time() - _t
            search_queue = deque([expanded_queries])
            current_depth = 0

            while search_queue and not self.meet_stop_condition(current_depth):
                current_depth += 1
                next_level: List[SearchNode] = []
                level_queries = search_queue.popleft()
                logger.info(f"=== Depth {current_depth} | {len(level_queries)} queries ===")

                # --- Phase 1: retrieval + scoring ---
                _t = time.time()
                level_nodes, next_level = self._phase_retrieval_and_scoring(
                    level_queries, query_node_relations, next_level, current_depth
                )
                timing["phase1_retrieval_scoring"] += time.time() - _t

                if self.meet_stop_condition(current_depth):
                    for n in level_nodes:
                        n.status = "STOP"
                    break

                # --- Phase 2: reference expansion ---
                if DO_REFERENCE_SEARCH:
                    _t = time.time()
                    level_nodes = self._phase_reference_expansion(level_nodes)
                    timing["phase2_reference"] += time.time() - _t
                    if self.meet_stop_condition(current_depth):
                        break

                # --- Phase 3: context-driven query generation (gated, default off) ---
                if DO_PHASE3:
                    _t = time.time()
                    level_nodes, search_queue, query_node_relations = (
                        self._phase_context_query_expansion(level_nodes, next_level, search_queue)
                    )
                    timing["phase3_context_query"] += time.time() - _t
                # else: search_queue stays empty -> BFS ends after this depth

            elapsed = time.time() - search_start
            logger.info(f"Search completed in {elapsed:.1f}s | {len(self.root.searched_docs)} docs found")

        except Exception:
            logger.error(f"Search failed: {traceback.format_exc()}")
        finally:
            self._cleanup_resources()

        # --- Phase 4: result collection ---
        _t = time.time()
        results = self._collect_results()
        timing["collect_results"] = time.time() - _t

        # --- Optional Phase 5: BGE cross-encoder rerank of B_raw ---
        if ENABLE_RERANK:
            _t = time.time()
            reranked = self._phase_rerank()
            if reranked:
                results = reranked
            timing["rerank"] = time.time() - _t

        timing["search_total"] = time.time() - search_start
        for k in timing:
            timing[k] = round(timing[k], 3)
        logger.info(f"Phase timings (s): {timing}")
        return results

    # ------------------------------------------------------------------
    # Phase 0: initial query fusion
    # ------------------------------------------------------------------

    def _phase_query_fusion(self) -> Tuple[List[str], Dict]:
        logger.info("Phase 0: query expansion")
        try:
            expanded_queries_info = self.query_agent.expand_query(self.root.query_str)
            expanded_queries_info["QUERY_NUM_PRUNED"] = QUERY_NUM_PRUNED
            self.root.extra["expanded_queries_info"] = expanded_queries_info
            expanded_queries = expanded_queries_info.get("expanded_queries", [])

            if self.root.query_str not in expanded_queries:
                expanded_queries = [self.root.query_str] + expanded_queries

            query_node_relations = {}
            for q in expanded_queries:
                node = SearchNode(query_str=q, status="START")
                query_node_relations[q] = {"own_node": node, "parent_node": self.root}

            return expanded_queries, query_node_relations
        except Exception:
            logger.error(f"_phase_query_fusion failed: {traceback.format_exc()}")
            return [self.root.query_str], {}

    # ------------------------------------------------------------------
    # Phase 1: retrieval + scoring
    # ------------------------------------------------------------------

    def _phase_retrieval_and_scoring(
        self,
        expanded_queries: List[str],
        query_node_relations: Dict,
        next_level: List[SearchNode],
        current_depth: int,
    ) -> Tuple[List[SearchNode], List[SearchNode]]:
        logger.info(f"Phase 1: retrieval + scoring ({len(expanded_queries)} queries)")
        if not expanded_queries:
            return [], next_level

        # Determine sources
        sources = SEARCH_ROUTES
        if hasattr(self.root, "extra") and "expanded_queries_info" in self.root.extra:
            sources = self.root.extra["expanded_queries_info"].get("suitable_sources", SEARCH_ROUTES)
        if "arxiv" not in sources:
            sources = ["arxiv"] + list(sources)
        if current_depth > 1:
            sources = ["arxiv"]

        # Agent B (LLM-recall) runs ASYNC: launch it now so its ~2-min thinking
        # recall + verify + self-scoring overlaps the main retrieval/scoring below
        # instead of blocking it. Its scored docs are merged in after the main pass.
        agent_b_executor = None
        agent_b_future = None
        if current_depth == 1 and AGENT_B_ENABLED:
            agent_b_executor = ThreadPoolExecutor(max_workers=1)
            agent_b_future = agent_b_executor.submit(self._agent_b_scored_docs)
            logger.info("AgentB: launched async (recall+verify+score), not blocking Phase 1")

        try:
            query2docs, id2docs, query_source_map, query_keywords2raw = (
                self.retrieval_agent.search(
                    expanded_queries,
                    sources=sources,
                    end_date=self.search_date,
                    searched_docs=self.root.searched_docs,
                )
            )

            self.root.searched_queries.update(expanded_queries)
            self.root.add_signature_for_doc(list(id2docs.values()))

            if SAVE_ID2DOCS:
                self._save_id2info(id2docs)

            # Build SearchNodes from query results
            current_level_nodes = []
            for query, docs in query2docs.items():
                source = query_source_map.get(query, "unknown")
                status = "Finished" if docs else "Failed"
                if query in query_node_relations:
                    node = query_node_relations[query]["own_node"]
                    parent = query_node_relations[query]["parent_node"]
                    node.status = status
                    node.source = source
                    node.parent = parent
                    node.raw_query = query
                    parent.add_child(node)
                else:
                    raw_query = query_keywords2raw.get(query, query)
                    parent = query_node_relations.get(raw_query, {}).get("parent_node", self.root)
                    node = SearchNode(
                        query_str=query, status=status, parent=parent,
                        source=source, raw_query=raw_query
                    )
                    parent.add_child(node)
                current_level_nodes.append(node)

            # --- Agent C (intent-constraint) injected pre-scoring ---
            # Agent C is fast (~seconds), so it is injected as extra query-nodes and
            # scored together with the retrieved docs below (Pass A dedups against
            # B_raw). Agent B is NOT here — it runs async (launched above) and self-
            # scores, so the main scoring never waits on its slow thinking recall.
            if current_depth == 1 and AGENT_C_ENABLED:
                current_level_nodes = self._inject_agent_c(
                    query2docs, current_level_nodes
                )

            # --- Score each node's documents ---
            # Pass A (sequential, no LLM): pick valid docs per node, deduping
            # across nodes up front so parallel scoring doesn't double-score the
            # same paper. Mirrors the old per-node cal_sim_docs check.
            valid_total = 0
            seen_ids = set(self.root.cal_sim_docs.keys())
            work: List[Tuple[SearchNode, List[Dict]]] = []
            for node in current_level_nodes:
                raw_docs = query2docs.get(node.query_str, [])
                if not raw_docs:
                    node.status = "Failed"
                    next_level.append(node)
                    continue
                valid_docs = []
                for d in raw_docs:
                    if not (d.get("title") and d.get("abstract")):
                        continue
                    pid = d.get("paper_id", d.get("arxivId"))
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    valid_docs.append(d)
                valid_total += len(valid_docs)
                if not valid_docs:
                    node.status = "NO Valid Docs"
                    continue
                work.append((node, valid_docs))

            # Pass B: score nodes concurrently (each node ≈ one batched LLM call).
            def _score_node(node: SearchNode, valid_docs: List[Dict]):
                return node, self.scoring_agent.score(
                    query=self.root.query_str,
                    docs=valid_docs,
                    search_time=self.search_date,
                    score_thresh=self.sim_threshold,
                    source=f"retrieval [{node.source}] {node.query_str}",
                )

            # Pass C: merge results in THIS thread only (as_completed is consumed
            # serially), so root.searched_docs / cal_sim_docs are mutated by a
            # single thread — no lock needed.
            rel_total = 0
            workers = max(1, min(PHASE1_NODE_PARALLEL, len(work)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(_score_node, n, vd) for n, vd in work]
                for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Scoring docs"):
                    node, (relevant, irrelevant) = fut.result()
                    self.root.add_signature_for_doc(relevant + irrelevant)
                    self.root.cal_sim_docs.update(
                        {d.get("paper_id", d.get("arxivId", "")): d for d in relevant + irrelevant}
                    )
                    node.docs.extend(relevant)
                    node.irrelevant_docs.extend(irrelevant)
                    rel_total += len(relevant)
                    if relevant:
                        node.status = "Expand"
                    elif irrelevant:
                        node.status = "NO Relevance"
                    else:
                        node.status = "Failed"
                        if node not in next_level:
                            next_level.append(node)

            # --- Join async Agent B and merge its self-scored docs into B_raw ---
            if agent_b_future is not None:
                self._merge_agent_b(agent_b_future, current_level_nodes, next_level)

            logger.info(f"Phase 1 done: {valid_total} valid docs, {rel_total} relevant")
            return current_level_nodes, next_level

        except Exception:
            logger.error(f"_phase_retrieval_and_scoring failed: {traceback.format_exc()}")
            return [], next_level
        finally:
            if agent_b_executor is not None:
                agent_b_executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Agent C — injected pre-scoring (fast, scored with the retrieved docs)
    # ------------------------------------------------------------------

    def _inject_agent_c(
        self, query2docs: Dict[str, List[Dict]], current_level_nodes: List[SearchNode]
    ) -> List[SearchNode]:
        """Run Agent C and inject its docs as a synthetic query-node.

        The full-metadata docs go into searched_docs (so collect/Phase-2 see all
        fields) and are registered in query2docs, so the Pass A/B scoring below
        scores them exactly like retrieved docs — Pass A's seen_ids set drops any
        id already claimed by the normal retrieval ("already in B_raw -> don't
        re-add"). Failures are isolated and never abort Phase 1.
        """
        try:
            docs = self.constraint_agent.search(self.root.query_str, self.search_date) or []
        except Exception:
            logger.error(f"agentC failed: {traceback.format_exc()}")
            return current_level_nodes

        docs = [
            d for d in docs
            if d.get("title") and d.get("abstract") and (d.get("paper_id") or d.get("arxivId"))
        ]
        if not docs:
            logger.info("agentC: no valid docs to inject")
            return current_level_nodes
        for d in docs:
            d.setdefault("paper_id", d.get("arxivId"))
        self.root.add_signature_for_doc(docs)
        syn_query = ("[AgentC-constraint] " + self.root.query_str)[:300]
        node = SearchNode(
            query_str=syn_query, status="START", parent=self.root, source="agentC"
        )
        self.root.add_child(node)
        query2docs[syn_query] = docs
        current_level_nodes.append(node)
        logger.info(f"agentC: injected {len(docs)} candidate docs into B_raw")
        return current_level_nodes

    # ------------------------------------------------------------------
    # Agent B — async (recall thinking + verify flash + self-scoring)
    # ------------------------------------------------------------------

    def _agent_b_scored_docs(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Agent B end-to-end, IN ITS OWN THREAD.

        Round 1 recall (V4-Pro thinking) + round 2 verify (flash) live in
        RecallAgent; here we additionally SELF-SCORE the verified docs as one
        batch through the SAME ScoringAgent the main pass uses (identical prompts
        / batch mechanism), so the scored candidates can be merged straight into
        B_raw without the main pass ever waiting on the slow recall.
        Returns (full_docs, relevant, irrelevant).
        """
        try:
            docs = self.recall_agent.recall(self.root.query_str, self.search_date) or []
            docs = [
                d for d in docs
                if d.get("title") and d.get("abstract") and (d.get("paper_id") or d.get("arxivId"))
            ]
            for d in docs:
                d.setdefault("paper_id", d.get("arxivId"))
            if not docs:
                return [], [], []
            relevant, irrelevant = self.scoring_agent.score(
                query=self.root.query_str,
                docs=docs,
                search_time=self.search_date,
                score_thresh=self.sim_threshold,
                source="agentB recall",
            )
            return docs, relevant, irrelevant
        except Exception:
            logger.error(f"_agent_b_scored_docs failed: {traceback.format_exc()}")
            return [], [], []

    def _merge_agent_b(
        self, future, level_nodes: List[SearchNode], next_level: List[SearchNode]
    ) -> None:
        """Join the async Agent B job and merge its self-scored docs into B_raw.

        Dedup: only ids not already scored by the main pass (absent from
        cal_sim_docs) are merged, so we never overwrite a main score or
        double-count. The merge runs in the main thread after the main scoring
        pass, so searched_docs / cal_sim_docs stay single-writer.
        """
        try:
            docs, relevant, irrelevant = future.result()
        except Exception:
            logger.error(f"AgentB async result failed: {traceback.format_exc()}")
            return
        if not docs:
            logger.info("AgentB: no verified docs to merge")
            return

        new_ids = {d["paper_id"] for d in docs if d["paper_id"] not in self.root.cal_sim_docs}
        full_new = [d for d in docs if d["paper_id"] in new_ids]
        rel_new = [s for s in relevant if s.get("paper_id") in new_ids]
        irr_new = [s for s in irrelevant if s.get("paper_id") in new_ids]
        if not full_new:
            logger.info(f"AgentB merged: all {len(docs)} verified docs already in B_raw")
            return

        # Full metadata first, then merge the sim_scores onto the same entries
        # (mirrors the main Pass C merge order).
        self.root.add_signature_for_doc(full_new)
        self.root.add_signature_for_doc(rel_new + irr_new)
        self.root.cal_sim_docs.update(
            {s["paper_id"]: s for s in rel_new + irr_new if s.get("paper_id")}
        )
        node = SearchNode(
            query_str=("[AgentB-recall] " + self.root.query_str)[:300],
            status="Expand" if rel_new else "NO Relevance",
            parent=self.root,
            source="agentB",
        )
        node.docs.extend(rel_new)
        node.irrelevant_docs.extend(irr_new)
        self.root.add_child(node)
        level_nodes.append(node)
        if not rel_new and node not in next_level:
            next_level.append(node)
        logger.info(
            f"AgentB merged: {len(docs)} verified -> {len(full_new)} new into B_raw "
            f"({len(rel_new)} relevant); {len(docs) - len(full_new)} already present"
        )

    # ------------------------------------------------------------------
    # Phase 2: reference expansion
    # ------------------------------------------------------------------

    def _phase_reference_expansion(self, level_nodes: List[SearchNode]) -> List[SearchNode]:
        """Two-pass reference expansion.

        Pass 1 (fetch): pull each top-doc's reference list, aggregate into a
        global pool (deduped, with co-occurrence counts) — no LLM yet.
        Prune: keep a global top-N by the selected strategy (the dominant Phase 2
        cost is LLM scoring, so this is the key lever).
        Pass 2 (score): LLM-score the pruned set once each (cross-phase deduped),
        then attribute results back to the doc's node.
        """
        logger.info(f"Phase 2: reference expansion ({len(level_nodes)} nodes)")
        if not level_nodes:
            return level_nodes

        # Top docs to expand, by sim_score
        all_doc_node_pairs = []
        for node in level_nodes:
            for doc in node.docs + node.irrelevant_docs:
                all_doc_node_pairs.append((node, doc))
        all_doc_node_pairs.sort(key=lambda x: x[1].get("sim_score", 0), reverse=True)

        # Expansion cap. Normally the top DOCS_TO_EXPAND by sim. But when early-stop
        # is OFF and depth-1 already has more than max_docs high-rel (sim>0.75) docs,
        # expand ONLY the top-max_docs high-rel docs (those would have tripped the
        # early stop) — references of the strongest docs, not a wide net. With
        # high-rel <= max_docs, keep the original DOCS_TO_EXPAND behaviour.
        high_rel_n = sum(
            1 for d in self.root.searched_docs.values()
            if d.get("sim_score", -1) > self.high_score_thresh
        )
        if not ENABLE_EARLY_STOP and high_rel_n > self.max_docs:
            expand_cap = self.max_docs
            logger.info(
                f"Phase 2 (early-stop off): high-rel {high_rel_n} > {self.max_docs} "
                f"-> expanding top {expand_cap} high-rel docs"
            )
        else:
            expand_cap = DOCS_TO_EXPAND
        all_doc_node_pairs = all_doc_node_pairs[:min(len(all_doc_node_pairs), expand_cap)]

        # Sub-step timing/counters (accumulated across tree depths into extra).
        detail = self.root.extra.setdefault("timing_phase2_detail", {
            "ref_fetch_s": 0.0, "ref_score_s": 0.0,
            "n_docs_expanded": 0, "n_refs_fetched_unique": 0,
            "n_refs_pruned": 0, "n_refs_scored": 0, "n_relevant": 0,
        })
        fetch_s = 0.0
        docs_expanded = 0

        # ---- Pass 1: fetch references, aggregate globally ----
        ref_pool: Dict[str, Dict] = {}   # paper_id -> ref doc (deduped)
        ref_freq: Dict[str, int] = {}    # paper_id -> co-occurrence count across docs
        ref_owner: Dict[str, SearchNode] = {}  # paper_id -> node that first cited it
        processed_ids = set()

        for node, doc in tqdm.tqdm(all_doc_node_pairs, desc="Ref fetch"):
            paper_id = doc.get("paper_id", "")
            if not paper_id or paper_id in processed_ids:
                continue
            processed_ids.add(paper_id)

            doc_info = self.root.searched_docs.get(paper_id)
            if not doc_info:
                continue

            refs = doc_info.get("references", [])
            has_full_refs = any(r.get("title") and r.get("abstract") for r in refs)
            if not refs or not has_full_refs:
                _t = time.time()
                doc_info = self.reference_agent.get_references(doc_info)
                fetch_s += time.time() - _t
                refs = doc_info.get("references", [])

            valid_refs = [
                r for r in refs
                if r.get("title") and r.get("abstract") and r.get("paper_id")
            ]
            if not valid_refs:
                continue
            docs_expanded += 1

            self.root.add_signature_for_doc([doc_info])
            self.root.add_signature_for_doc(valid_refs)
            if SAVE_ID2DOCS:
                self._save_id2info({r["paper_id"]: r for r in valid_refs})
                self._save_id2info({paper_id: doc_info})

            node.references.extend(r["paper_id"] for r in valid_refs)
            for r in valid_refs:
                rid = r["paper_id"]
                ref_freq[rid] = ref_freq.get(rid, 0) + 1
                if rid not in ref_pool:
                    ref_pool[rid] = r
                    ref_owner[rid] = node

        if not ref_pool:
            detail["ref_fetch_s"] = round(detail["ref_fetch_s"] + fetch_s, 3)
            detail["n_docs_expanded"] += docs_expanded
            logger.info("Phase 2 done: no references fetched")
            return level_nodes

        # ---- Cap the pool before the expensive LLM scoring ----
        # With a working BGE prefilter, hand the FULL deduped pool to score():
        # the relevance-aware coarse filter keeps top-K for the LLM (cheaper than
        # freq-pruned LLM scoring, and rarely-co-cited gold refs — freq=1/cite=0,
        # which count-based pruning ranks dead last — stay reachable).
        # Otherwise fall back to count-based freq/citation pruning to cap LLM cost.
        if self.scoring_agent.prefilter_ready():
            pruned = list(ref_pool.values())
            prune_tag = "bge-pool"
        else:
            pruned = self._prune_references(ref_pool, ref_freq)
            prune_tag = REFERENCE_PRUNE_STRATEGY
        # Cross-phase dedup: skip refs already scored in Phase 1 / earlier.
        to_score = [r for r in pruned if r["paper_id"] not in self.root.cal_sim_docs]
        logger.info(
            f"Phase 2 prune[{prune_tag}]: {len(ref_pool)} unique refs "
            f"-> {len(pruned)} kept -> {len(to_score)} to score"
        )

        # ---- Pass 2: score the pruned set once, attribute back to nodes ----
        _t = time.time()
        relevant_refs, irrelevant_refs = self.scoring_agent.score(
            query=self.root.query_str,
            docs=to_score,
            search_time=self.search_date,
            score_thresh=self.sim_threshold,
            source="reference (pruned)",
        )
        score_s = time.time() - _t
        self.root.add_signature_for_doc(relevant_refs + irrelevant_refs)
        self.root.cal_sim_docs.update(
            {d["paper_id"]: d for d in relevant_refs + irrelevant_refs if d.get("paper_id")}
        )
        for d in relevant_refs:
            n = ref_owner.get(d.get("paper_id"), level_nodes[0])
            n.relevance_refs.append(d)
            n.docs.append(d)
        for d in irrelevant_refs:
            n = ref_owner.get(d.get("paper_id"), level_nodes[0])
            n.irrelevant_refs.append(d)
            n.irrelevant_docs.append(d)

        # accumulate sub-step timing/counts into extra (survives across depths)
        detail["ref_fetch_s"] = round(detail["ref_fetch_s"] + fetch_s, 3)
        detail["ref_score_s"] = round(detail["ref_score_s"] + score_s, 3)
        detail["n_docs_expanded"] += docs_expanded
        detail["n_refs_fetched_unique"] += len(ref_pool)
        detail["n_refs_pruned"] += len(pruned)
        detail["n_refs_scored"] += len(to_score)
        detail["n_relevant"] += len(relevant_refs)
        logger.info(
            f"Phase 2 done: fetch {fetch_s:.1f}s, score {score_s:.1f}s | "
            f"scored {len(to_score)} refs, {len(relevant_refs)} relevant"
        )
        return level_nodes

    @staticmethod
    def _ref_citation(ref: Dict) -> int:
        """Unified citation count across sources (S2: citationCount, OpenAlex: cited_by_count)."""
        return int(ref.get("citationCount") or ref.get("cited_by_count") or 0)

    def _prune_references(
        self, ref_pool: Dict[str, Dict], ref_freq: Dict[str, int]
    ) -> List[Dict]:
        """Select which references survive to LLM scoring (see REFERENCE_PRUNE_STRATEGY)."""
        refs = list(ref_pool.values())
        if REFERENCE_PRUNE_STRATEGY == "citation":
            top_n = REFERENCE_PRUNE_TOPN or 200
            refs.sort(key=lambda r: self._ref_citation(r), reverse=True)
        else:  # "freq" (default): co-occurrence count, then citation count
            top_n = REFERENCE_PRUNE_TOPN or 150
            refs.sort(
                key=lambda r: (ref_freq.get(r["paper_id"], 0), self._ref_citation(r)),
                reverse=True,
            )
        return refs[:top_n]

    # ------------------------------------------------------------------
    # Phase 3: context-driven query generation
    # ------------------------------------------------------------------

    def _phase_context_query_expansion(
        self,
        level_nodes: List[SearchNode],
        next_level: List[SearchNode],
        search_queue: deque,
    ) -> Tuple[List[SearchNode], deque, Dict]:
        logger.info("Phase 3: context-driven query generation")

        # Collect relevant docs not yet used for query generation
        valid_doc_node_pairs = []
        for node in tqdm.tqdm(level_nodes, desc="Context query prep"):
            for doc in node.docs:
                if doc["paper_id"] not in self.root.doc_used_to_gen_query:
                    valid_doc_node_pairs.append([doc, node])

        logger.info(f"Candidate docs for context expansion: {len(valid_doc_node_pairs)}")

        if not valid_doc_node_pairs:
            docs_for_gen = [[{}, self.root]]
        else:
            sorted_pairs = sorted(valid_doc_node_pairs, key=lambda x: x[0]["sim_score"], reverse=True)
            top_pairs = sorted_pairs[:REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY]
            docs_for_gen = [
                [self.root.searched_docs[doc["paper_id"]], node]
                for doc, node in top_pairs
                if doc["paper_id"] in self.root.searched_docs
            ]
            used_ids = [doc["paper_id"] for doc, _ in top_pairs]
            self.root.doc_used_to_gen_query.update(used_ids)

        new_query_node_pairs = self.query_agent.generate_queries_from_docs(
            self.root.query_str,
            docs_for_gen,
            list(self.root.searched_queries),
        )
        logger.info(f"Generated {len(new_query_node_pairs)} new queries")

        # Deduplicate new queries
        seen_queries = []
        next_level_prepare = []
        for new_q, parent_node in new_query_node_pairs:
            if new_q not in seen_queries:
                seen_queries.append(new_q)
                child = SearchNode(query_str=new_q)
                next_level_prepare.append((parent_node, child))

        # Build query_node_relations for the next BFS level
        query_node_relations: Dict = {}
        queries_to_next: List[str] = []

        for node in next_level:
            query_node_relations[node.query_str] = {
                "own_node": node, "parent_node": node.parent
            }
            queries_to_next.append(node.query_str)

        if next_level_prepare:
            if QUERY_NUM_PRUNED < len(next_level_prepare):
                next_level_prepare = random.sample(next_level_prepare, QUERY_NUM_PRUNED)
            for parent_node, child_node in next_level_prepare:
                child_node.status = "Expand"
                queries_to_next.append(child_node.query_str)
                query_node_relations[child_node.query_str] = {
                    "own_node": child_node, "parent_node": parent_node
                }
            search_queue.append(queries_to_next)

        return level_nodes, search_queue, query_node_relations

    # ------------------------------------------------------------------
    # Stop condition
    # ------------------------------------------------------------------

    def meet_stop_condition(self, current_depth: int = None) -> bool:
        relevance_docs = {
            doc_id
            for doc_id, doc in self.root.searched_docs.items()
            if doc.get("sim_score", -1) > self.high_score_thresh
        }
        self.root.hight_relevance_docs = relevance_docs
        enough_docs = len(relevance_docs) > self.max_docs

        if current_depth is None:
            # Mid-phase call (inside refchain): always run to completion
            logger.info(
                f"Stop check (mid-phase) | high-rel {len(relevance_docs)}/{self.max_docs}"
            )
            return False

        depth_exceeded = current_depth >= self.max_depth
        # With early-stop disabled, the high-rel count never stops the tree — only
        # depth does — so Phase 2 reference expansion still runs at depth 1.
        enough_stop = enough_docs and ENABLE_EARLY_STOP
        logger.info(
            f"Stop check | depth {current_depth}/{self.max_depth} | "
            f"high-rel {len(relevance_docs)}/{self.max_docs} | "
            f"total {len(self.root.searched_docs)} | "
            f"early_stop={'on' if ENABLE_EARLY_STOP else 'off'}"
        )
        return depth_exceeded or enough_stop

    # ------------------------------------------------------------------
    # Result collection
    # ------------------------------------------------------------------

    def _collect_results(self) -> Dict:
        # B-raw: all papers that received a valid sim_score (excludes scoring failures)
        scored = {
            doc_id: doc
            for doc_id, doc in self.root.searched_docs.items()
            if doc.get("sim_score", -1) >= 0
        }
        if not scored:
            return {}

        # cite_norm/ahp_score are still annotated on every scored doc: evaluate.py's
        # ranked-B_raw metrics use ahp_score as a rank-key fallback, and --rerank can
        # blend cite_norm. They no longer decide Set C below.
        cite_values = [doc.get("citationCount") or 0 for doc in scored.values()]
        min_cite = min(cite_values)
        max_cite = max(cite_values)

        for doc_id, doc in scored.items():
            cite = doc.get("citationCount") or 0
            if max_cite > min_cite:
                cite_norm = 0.01 + (cite - min_cite) / (max_cite - min_cite) * (1.0 - 0.01)
            else:
                # all papers share the same citation count → authority uninformative
                cite_norm = 0.5
            doc["cite_norm"] = round(cite_norm, 4)
            doc["ahp_score"] = round(0.5 * doc["sim_score"] + 0.5 * cite_norm, 4)

        # --- Set C: A_raw pool -> citation gate -> sim ranking ---
        # 1) candidate pool = A_raw (sim_score > high_score_thresh, post Phase 2);
        #    falls back to the full scored pool when A_raw is empty
        # 2) keep only the top-70% by citationCount ranking
        # 3) sort by sim_score (citationCount as tie-break), return top max_docs
        a_raw = {
            doc_id: doc
            for doc_id, doc in scored.items()
            if doc["sim_score"] > self.high_score_thresh
        }
        pool = a_raw if a_raw else scored
        if not a_raw:
            logger.info("Collect: A_raw empty, falling back to all scored docs")

        by_cite = sorted(
            pool.items(), key=lambda x: x[1].get("citationCount") or 0, reverse=True
        )
        keep_n = max(1, math.ceil(len(by_cite) * 0.7))
        kept = by_cite[:keep_n]
        ranked = sorted(
            kept,
            key=lambda x: (x[1]["sim_score"], x[1].get("citationCount") or 0),
            reverse=True,
        )
        logger.info(
            f"Collect: pool {len(pool)} (A_raw {len(a_raw)}) -> cite-top70% {keep_n} "
            f"-> top {min(self.max_docs, keep_n)} by sim"
        )
        return dict(ranked[: self.max_docs])

    # ------------------------------------------------------------------
    # Optional BGE rerank
    # ------------------------------------------------------------------

    @property
    def bge_reranker(self):
        if self._bge_reranker is None:
            from bge_rerank import BgeReranker
            self._bge_reranker = BgeReranker()
        return self._bge_reranker

    def _phase_rerank(self) -> Optional[Dict]:
        """Rerank the full B_raw pool with the BGE cross-encoder; return new top-k.

        Writes `bge_score` into each candidate in searched_docs (persisted to JSON).
        Pure bge by default; set RERANK_CITATION_WEIGHT>0 to blend in cite_norm.
        Returns None (keep AHP ranking) if the reranker is unavailable.
        """
        candidates = [
            d for d in self.root.searched_docs.values()
            if d.get("title") and d.get("abstract") and d.get("paper_id")
        ]
        if not candidates:
            return None
        scores = self.bge_reranker.score(self.root.query_str, candidates)
        if not scores:
            logger.warning("BGE rerank unavailable; keeping AHP ranking")
            return None
        for d, s in zip(candidates, scores):
            d["bge_score"] = round(float(s), 4)

        w = RERANK_CITATION_WEIGHT
        if w > 0:
            bs = [d["bge_score"] for d in candidates]
            lo, hi = min(bs), max(bs)
            for d in candidates:
                bge_norm = (d["bge_score"] - lo) / (hi - lo) if hi > lo else 0.5
                d["rerank_final"] = round(
                    (1 - w) * bge_norm + w * (d.get("cite_norm") or 0.0), 4
                )
            key = lambda d: d.get("rerank_final", 0.0)
        else:
            key = lambda d: d.get("bge_score", -1e9)

        ranked = sorted(candidates, key=key, reverse=True)[: self.max_docs]
        logger.info(f"BGE rerank: scored {len(candidates)} B_raw docs -> top {len(ranked)}")
        return {d["paper_id"]: d for d in ranked}

    def _rank_query_doc_list(self, docs: List[Dict]) -> List[Dict]:
        """Bucket by sim_score (0.05 bins), then sort by citationCount + year within each bucket."""
        from collections import defaultdict

        def bucket(score):
            return round(score / 0.05) * 0.05

        bucketed: Dict[float, list] = defaultdict(list)
        for doc in docs:
            doc.setdefault("year", -1)
            if doc["year"] is None:
                doc["year"] = -1
            bucketed[bucket(doc.get("sim_score", 0))].append(doc)

        result = []
        for score_key in sorted(bucketed, reverse=True):
            bucket_docs = sorted(
                bucketed[score_key],
                key=lambda d: (-d.get("citationCount", 0), -d.get("year", 0)),
            )
            result.extend(bucket_docs)
        return result

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_id2info(self, id2docs: Dict[str, Any]):
        if not id2docs:
            return
        try:
            with ArxivDatabase(db_path) as db:
                for arxiv_id, info in id2docs.items():
                    if info.get("source") == "Search From Local":
                        continue
                    cleaned = {k: v for k, v in info.items() if "sim_score" not in k}
                    db.update_or_insert(arxiv_id, cleaned)
        except Exception:
            logger.error(f"_save_id2info failed: {traceback.format_exc()}")

    def _cleanup_resources(self):
        try:
            if hasattr(self.scoring_agent, "_emd_model") and self.scoring_agent._emd_model:
                if hasattr(self.scoring_agent._emd_model, "cleanup"):
                    self.scoring_agent._emd_model.cleanup()
            if hasattr(self.root, "cal_sim_docs"):
                self.root.cal_sim_docs.clear()
        except Exception:
            logger.error(f"_cleanup_resources failed: {traceback.format_exc()}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_tree(
        self,
        filename: str = "search_tree",
        save_format: str = "pdf",
        view: bool = False,
    ):
        dot = Digraph(comment="SPAR Search Tree")
        if not self.root:
            return

        def wrap_text(text: str, words_per_line: int = 10) -> str:
            words = text.split()
            lines = [" ".join(words[i: i + words_per_line]) for i in range(0, len(words), words_per_line)]
            return "\n".join(lines)

        queue = deque([(self.root, "0")])
        counter = 0
        while queue:
            node, node_id = queue.popleft()
            if node_id == "0":
                dot.node(node_id, "SPAR\nStart", shape="hexagon")
            else:
                n_rel = len(node.docs) if node.docs else 0
                n_irrel = len(node.irrelevant_docs) if node.irrelevant_docs else 0
                n_refs = len(node.references)
                n_rel_refs = len(node.relevance_refs) if node.relevance_refs else 0
                q_text = wrap_text(node.query_str, 4)
                label = f"[{node.source}]: {q_text}"
                if node.raw_query and node.raw_query != node.query_str:
                    label += f"\n[RAW]: {wrap_text(node.raw_query, 4)}"
                label += f"\nRel:{n_rel} Irrel:{n_irrel} Refs:{n_refs} RelRef:{n_rel_refs}"
                label += f"\n{node.status}"
                dot.node(node_id, label, shape="box")

            for child in node.children:
                counter += 1
                cid = str(counter)
                dot.edge(node_id, cid)
                queue.append((child, cid))

        filepath = dot.render(filename=filename, format=save_format, view=view)
        logger.info(f"Search tree saved: {filepath}")
