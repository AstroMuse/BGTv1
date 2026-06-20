# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
ScoringAgent — Document Relevance Scoring.

Responsibilities:
  - Score each retrieved document against the user query.
  - Default path: LLM-based multi-dimensional scoring (v4 JSON rubric).
  - Alternative paths: single-score LLM (v5), BGE-M3 embeddings, PASA selector.
  - Parallel batch execution with retry and fallback.

Public interface:
  score(query, docs, search_time, score_thresh, source)
      -> (relevant_docs, irrelevant_docs)
         Each doc gets "sim_score" and "sim_info_details" fields injected.
"""

import traceback
import time
import json
import re
import concurrent.futures
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from global_config import (
    LLM_MODEL_NAME,
    LLM_TRY_COUNT,
    LLM_PARREL_NUM,
    BEGIN_SIM_THRESHOLD,
    PASS_SIM_THRESHOLD,
    BGE_PREFILTER,
    BGE_PREFILTER_TOPK,
    SCORING_PROMPT,
    SCORE_BATCH_SIZE,
)
from instruction import (
    template_sim_between_query_doc_v2_inst,
    template_sim_between_query_doc_v2_example,
    template_sim_intent_align_inst,
    template_sim_intent_align_example,
    template_sim_intent_align_batch_inst,
    template_sim_intent_align_batch_example,
    evaluation_prompt,
)
from local_request_v2 import get_from_llm
from utils import fetch_string
from log import logger


class ScoringAgent:
    """
    Relevance scoring for retrieved documents. Supports three interchangeable
    backends: LLM v4 (default), LLM v5, BGE-M3, and PASA selector.
    BGE and PASA are lazy-loaded only when their respective methods are called.
    """

    def __init__(self):
        self._emd_model = None   # lazy-loaded BGE-M3
        self._selector = None    # lazy-loaded PASA
        self._bge_prefilter = None  # lazy-loaded BGE cross-encoder (prefilter mode)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def prefilter_ready(self) -> bool:
        """True iff BGE prefilter is enabled AND the cross-encoder actually loads.

        Callers that want to hand an oversized pool to score() (relying on the
        BGE coarse filter to cap LLM cost) MUST check this first — otherwise a
        graceful reranker-load failure would dump the full pool on the LLM.
        """
        return BGE_PREFILTER and self.bge_prefilter._ensure_loaded()

    def score(
        self,
        query: str,
        docs: List[Dict],
        search_time: str = "",
        score_thresh: float = 0.5,
        source: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Score docs and split into (relevant, irrelevant) by score_thresh.

        Routing (Phase 1 retrieval scoring + Phase 2 reference scoring both go
        through here):
          1. Opt-in legacy BGE coarse filter (default OFF) — kept for ablation.
          2. Default: batched LLM scoring (SCORE_BATCH_SIZE docs per call) for the
             "intent" prompt — index-anchored with per-doc fallback re-score.
          3. Per-doc LLM scoring (SCORE_BATCH_SIZE<=1, single doc, or "rubric").
        """
        if BGE_PREFILTER and len(docs) > BGE_PREFILTER_TOPK:
            return self._score_bge_then_llm(query, docs, search_time, score_thresh, source)
        if SCORE_BATCH_SIZE > 1 and len(docs) > 1 and SCORING_PROMPT == "intent":
            return self._score_llm_batch(query, docs, search_time, score_thresh, source)
        return self._score_llm_all(query, docs, search_time, score_thresh, source)

    # ------------------------------------------------------------------
    # Batched LLM scoring (default for the "intent" prompt)
    # ------------------------------------------------------------------

    def _score_llm_batch(
        self,
        query: str,
        docs: List[Dict],
        search_time: str = "",
        score_thresh: float = 0.5,
        source: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """Score docs in groups of SCORE_BATCH_SIZE with ONE LLM call per group.

        Groups are scored concurrently (LLM_PARREL_NUM workers). Each group is
        index-anchored ([DOC i] -> "idx": i); any doc the batch fails to score is
        re-scored individually inside the group, so a malformed/truncated batch
        never silently drops a whole group to sim_score=-1 (the failure mode the
        per-doc path guards by construction).
        """
        relevant, irrelevant = [], []
        chunks = [
            docs[i: i + SCORE_BATCH_SIZE]
            for i in range(0, len(docs), SCORE_BATCH_SIZE)
        ]
        logger.info(
            f"Batch scoring {len(docs)} docs in {len(chunks)} group(s) of "
            f"<= {SCORE_BATCH_SIZE} (source: {source})"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_PARREL_NUM) as executor:
            future_to_chunk = {
                executor.submit(self._score_chunk, query, search_time, chunk): chunk
                for chunk in chunks
            }
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    results = future.result(timeout=300)
                except Exception:
                    logger.error(f"Batch group failed: {traceback.format_exc()}")
                    results = {}
                for i, doc in enumerate(chunk):
                    r = results.get(i)
                    if r:
                        doc.update(r)
                    else:
                        doc["sim_score"] = -1
                        doc["sim_info_details"] = {}
                    simple_info = {
                        "arxivId": doc.get("arxivId", ""),
                        "paper_id": doc.get("paper_id", doc.get("arxivId", "")),
                        "sim_score": doc.get("sim_score", -1),
                        "sim_info_details": doc.get("sim_info_details", {}),
                        "source": source,
                    }
                    if simple_info["sim_score"] >= score_thresh:
                        relevant.append(simple_info)
                    else:
                        irrelevant.append(simple_info)
        return relevant, irrelevant

    def _score_chunk(self, query: str, search_time: str, chunk: List[Dict]) -> Dict[int, Dict]:
        """Score one group in a single LLM call; return {chunk_index: result}.

        Missing indices (batch parse failure / model dropped them) fall back to
        per-doc scoring so nothing is lost.
        """
        out: Dict[int, Dict] = {}
        try:
            prompt = self._build_batch_prompt(query, search_time, chunk)
            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            out.update(self._parse_batch(response, len(chunk)))
        except Exception:
            logger.error(f"Batch chunk scoring failed: {traceback.format_exc()}")

        missing = [i for i in range(len(chunk)) if i not in out]
        if missing:
            logger.warning(
                f"Batch returned {len(chunk) - len(missing)}/{len(chunk)} docs; "
                f"re-scoring {len(missing)} per-doc"
            )
            for i in missing:
                try:
                    r = self._score_with_retry(query, search_time, chunk[i])
                    if r:
                        out[i] = r
                except Exception:
                    logger.error(f"Per-doc fallback failed: {traceback.format_exc()}")
        return out

    def _build_batch_prompt(self, query: str, search_time: str, chunk: List[Dict]) -> str:
        blocks = []
        for i, d in enumerate(chunk):
            authors = (
                "; ".join(a["name"] if isinstance(a, dict) else a for a in d["authors"])
                if d.get("authors") else ""
            )
            fos = (
                (d["fieldsOfStudy"] if isinstance(d["fieldsOfStudy"], str)
                 else ";".join(d["fieldsOfStudy"]))
                if d.get("fieldsOfStudy") else ""
            )
            blocks.append(
                f"[DOC {i}]\n"
                f"  Title: {d.get('title', '')}\n"
                f"  Abstract: {d.get('abstract', '')}\n"
                f"  Author: {authors}\n"
                f"  fieldsOfStudy: {fos}\n"
                f"  Publication Year: {d.get('publicationYear', '') or ''}"
            )
        return (
            template_sim_intent_align_batch_inst.format(
                searchTime=search_time,
                userQuery=query,
                documents="\n\n".join(blocks),
            )
            + template_sim_intent_align_batch_example
        )

    def _parse_batch(self, response: Optional[str], n: int) -> Dict[int, Dict]:
        """Parse the batch JSON into {chunk_index: {sim_score, sim_info_details}}.

        Tolerant: accepts a top-level list or {"papers": [...]} / {"results": [...]};
        uses each item's "idx" (falls back to array position); skips items missing
        a score key or with an out-of-range index. Same weighting as _score_llm_v6.
        """
        out: Dict[int, Dict] = {}
        if not response:
            return out
        try:
            data = json.loads(fetch_string(response).strip())
        except Exception:
            logger.error(f"Batch JSON parse failed: {traceback.format_exc()}")
            return out
        if isinstance(data, dict):
            items = data.get("papers") or data.get("results") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []
        keys = list(self.INTENT_WEIGHTS.keys())
        for pos, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("idx", pos))
            except (TypeError, ValueError):
                idx = pos
            if not (0 <= idx < n) or idx in out:
                continue
            if not all(k in item for k in keys):
                continue
            try:
                sim = sum(float(item[k]) * w for k, w in self.INTENT_WEIGHTS.items()) / 5.0
            except (TypeError, ValueError):
                continue
            out[idx] = {"sim_score": sim, "sim_info_details": item}
        return out

    @property
    def bge_prefilter(self):
        """Lazy-loaded fine-tuned bge-reranker (cross-encoder) used as the coarse filter."""
        if self._bge_prefilter is None:
            from bge_rerank import BgeReranker
            self._bge_prefilter = BgeReranker()
        return self._bge_prefilter

    def _score_bge_then_llm(
        self,
        query: str,
        docs: List[Dict],
        search_time: str,
        score_thresh: float,
        source: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Coarse BGE cross-encoder ranking -> keep top-K -> LLM fine scoring.

        The query passed in is the original user query (the orchestrator scores
        against self.root.query_str), which matches how the reranker was trained
        (question -> paper). Dropped docs are returned as irrelevant with their
        bge score recorded but sim_score=-1, so they stay out of B_raw.
        """
        bge_scores = self.bge_prefilter.score(query, docs)
        if not bge_scores:
            # reranker unavailable -> no quality loss, just no speedup
            logger.warning("BGE prefilter unavailable; falling back to full LLM scoring")
            return self._score_llm_all(query, docs, search_time, score_thresh, source)

        paired = sorted(zip(docs, bge_scores), key=lambda x: x[1], reverse=True)
        kept = [d for d, _ in paired[:BGE_PREFILTER_TOPK]]
        dropped = paired[BGE_PREFILTER_TOPK:]
        logger.info(
            f"BGE prefilter: {len(docs)} docs -> LLM-score top {len(kept)}, "
            f"dropped {len(dropped)} (source: {source})"
        )

        relevant, irrelevant = self._score_llm_all(
            query, kept, search_time, score_thresh, source
        )
        for d, s in dropped:
            irrelevant.append({
                "arxivId": d.get("arxivId", ""),
                "paper_id": d.get("paper_id", d.get("arxivId", "")),
                "sim_score": -1,  # not LLM-scored -> excluded from B_raw
                "bge_prefilter_score": float(s),
                "sim_info_details": {"reason": "bge-prefilter-dropped", "bge_score": float(s)},
                "source": source,
            })
        return relevant, irrelevant

    def _score_llm_all(
        self,
        query: str,
        docs: List[Dict],
        search_time: str = "",
        score_thresh: float = 0.5,
        source: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Score docs in parallel using the LLM backend.
        Returns (relevant_docs, irrelevant_docs) with sim_score injected.
        """
        relevant = []
        irrelevant = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_PARREL_NUM) as executor:
            future_to_doc = {
                executor.submit(self._score_with_retry, query, search_time, doc): doc
                for doc in docs
            }
            for future in concurrent.futures.as_completed(future_to_doc):
                doc = future_to_doc[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        doc.update(result)
                    else:
                        doc["sim_score"] = -1
                        doc["sim_info_details"] = {}
                except Exception:
                    logger.error(f"Scoring future failed: {traceback.format_exc()}")
                    doc["sim_score"] = -1
                    doc["sim_info_details"] = {}
                finally:
                    simple_info = {
                        "arxivId": doc.get("arxivId", ""),
                        "paper_id": doc.get("paper_id", doc.get("arxivId", "")),
                        "sim_score": doc.get("sim_score", -1),
                        "sim_info_details": doc.get("sim_info_details", {}),
                        "source": source,
                    }
                    if simple_info["sim_score"] >= score_thresh:
                        relevant.append(simple_info)
                    else:
                        irrelevant.append(simple_info)

        return relevant, irrelevant

    # ------------------------------------------------------------------
    # LLM scoring backends
    # ------------------------------------------------------------------

    def _score_with_retry(
        self,
        query: str,
        search_time: str,
        doc: Dict,
        max_retries: int = LLM_TRY_COUNT,
    ) -> Dict:
        score_fn = (
            self._score_llm_v6 if SCORING_PROMPT == "intent" else self._score_llm_v4
        )
        for attempt in range(max_retries):
            try:
                result = score_fn(query, doc, search_time)
                if result:
                    return result
            except Exception:
                logger.error(
                    f"Scoring attempt {attempt+1}/{max_retries} failed for "
                    f"'{doc.get('title', '?')}': {traceback.format_exc()}"
                )
                time.sleep(20)
        return {}

    def _score_llm_v4(self, query: str, doc: Dict, search_time: str) -> Dict:
        """Multi-dimensional JSON rubric scoring (topic_match, contextual_relevance, depth_completeness)."""
        model_inp = (
            template_sim_between_query_doc_v2_inst.format(
                searchTime=search_time,
                userQuery=query,
                Title=doc.get("title", ""),
                Abstract=doc.get("abstract", ""),
                Author=(
                    "; ".join(
                        a["name"] if isinstance(a, dict) else a
                        for a in doc["authors"]
                    )
                    if doc.get("authors") else ""
                ),
                fieldsOfStudy=(
                    (doc["fieldsOfStudy"] if isinstance(doc["fieldsOfStudy"], str)
                     else ";".join(doc["fieldsOfStudy"]))
                    if doc.get("fieldsOfStudy") else ""
                ),
                publicationYear=doc.get("publicationYear", "") or "",
            )
            + template_sim_between_query_doc_v2_example
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
        response = fetch_string(response)
        parsed = json.loads(response.strip())
        scores = [parsed[k] for k in ["topic_match", "contextual_relevance", "depth_completeness"]]
        return {
            "sim_score": float(np.mean(scores)) / 5.0,
            "sim_info_details": parsed,
        }

    # Weights for intent-alignment scoring: the alignment of the paper's core
    # contribution with the query's search intent dominates; scope and
    # directness refine it. sim_score stays in [0, 1].
    INTENT_WEIGHTS = {
        "intent_alignment": 0.5,
        "task_scope_match": 0.3,
        "evidence_directness": 0.2,
    }

    def _score_llm_v6(self, query: str, doc: Dict, search_time: str) -> Dict:
        """Intent-alignment scoring: paper core intent vs query search intent.

        The prompt first makes the LLM articulate query_intent and
        paper_core_intent, then score alignment — so papers that merely *use*
        the queried technique (different core goal) score low even with full
        keyword overlap.
        """
        model_inp = (
            template_sim_intent_align_inst.format(
                searchTime=search_time,
                userQuery=query,
                Title=doc.get("title", ""),
                Abstract=doc.get("abstract", ""),
                Author=(
                    "; ".join(
                        a["name"] if isinstance(a, dict) else a
                        for a in doc["authors"]
                    )
                    if doc.get("authors") else ""
                ),
                fieldsOfStudy=(
                    (doc["fieldsOfStudy"] if isinstance(doc["fieldsOfStudy"], str)
                     else ";".join(doc["fieldsOfStudy"]))
                    if doc.get("fieldsOfStudy") else ""
                ),
                publicationYear=doc.get("publicationYear", "") or "",
            )
            + template_sim_intent_align_example
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
        response = fetch_string(response)
        parsed = json.loads(response.strip())
        sim = sum(
            float(parsed[k]) * w for k, w in self.INTENT_WEIGHTS.items()
        ) / 5.0
        return {
            "sim_score": sim,
            "sim_info_details": parsed,
        }

    def _score_llm_v5(self, query: str, doc: Dict) -> Dict:
        """Single-score LLM scoring (Score: X.X format)."""
        model_inp = evaluation_prompt.format(
            title=doc.get("title", ""),
            abstract=doc.get("abstract", ""),
            user_query=query,
        )
        response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
        match = re.search(r"Score:\s*([0-1]\.\d+|\d+)", response)
        if match:
            return {
                "sim_score": float(match.group(1)),
                "sim_info_details": response,
            }
        return {}

    # ------------------------------------------------------------------
    # Alternative scoring backends (lazy-loaded)
    # ------------------------------------------------------------------

    @property
    def emd_model(self):
        if self._emd_model is None:
            from embedding_agent import BGEM3EmbeddingAgent
            self._emd_model = BGEM3EmbeddingAgent()
        return self._emd_model

    @property
    def selector(self):
        if self._selector is None:
            from pasa_agent import Agent as PasaAgent
            self._selector = PasaAgent(
                "/share/project/shixiaofeng/data/model_hub/pasa/pasa-7b-selector"
            )
        return self._selector

    def score_bge(
        self,
        query: str,
        docs: List[Dict],
        search_time: str = "",
        score_thresh: float = BEGIN_SIM_THRESHOLD,
        source: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """BGE-M3 embedding-based scoring."""
        logger.info("score_bge ...")
        relevant, irrelevant = [], []
        paper_texts = [
            f"Title:{d.get('title','')}\nAbstract:{d.get('abstract','')}"
            f"Authors:{';'.join(a['name'] for a in (d.get('authors') or []))}"
            for d in docs
        ]
        scores = self.emd_model.get_score(query, paper_texts, batch_size=6)
        for doc, sim_score in zip(docs, scores):
            info = {
                "arxivId": doc.get("arxivId", ""),
                "paper_id": doc.get("paper_id", doc.get("arxivId", "")),
                "sim_score": sim_score,
                "source": source,
                "sim_info_details": {"reason": "bge-m3", "sim_score": sim_score},
            }
            (relevant if sim_score >= score_thresh else irrelevant).append(info)
        return relevant, irrelevant

    def score_pasa(
        self,
        query: str,
        docs: List[Dict],
        search_time: str = "",
        score_thresh: float = PASS_SIM_THRESHOLD,
        source: str = "",
    ) -> Tuple[List[Dict], List[Dict]]:
        """PASA selector-based scoring."""
        logger.info("score_pasa ...")
        relevant, irrelevant = [], []
        docs = [d for d in docs if d.get("title") and d.get("abstract")]
        prompt_template = (
            "You are an elite researcher in the field of AI, conducting research on {user_query}. "
            "Evaluate whether the following paper fully satisfies the detailed requirements of the user query "
            "and provide your reasoning.\n\n"
            "Searched Paper:\nTitle: {title}\nAbstract: {abstract}\n\n"
            "User Query: {user_query}\n\nOutput format: Decision: True/False\nReason:...\nDecision:"
        )
        prompts = [
            prompt_template.format(
                title=d.get("title", ""), abstract=d.get("abstract", ""), user_query=query
            )
            for d in docs
        ]
        scores = self.selector.batch_infer_score(prompts, 4)
        for doc, sim_score in zip(docs, scores):
            info = {
                "arxivId": doc.get("arxivId", ""),
                "paper_id": doc.get("paper_id", doc.get("arxivId", "")),
                "sim_score": sim_score,
                "source": source,
                "sim_info_details": {"reason": "pasa-scorer", "sim_score": sim_score},
            }
            (relevant if sim_score >= score_thresh else irrelevant).append(info)
        return relevant, irrelevant

    def rerank_bge(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Re-rank a doc list by BGE-M3 score (returns same list with rerank_score_bge added)."""
        logger.info("rerank_bge ...")
        texts = [
            f"Title:{d.get('title','')}\nAbstract:{d.get('abstract','')}"
            f"Authors:{';'.join(a['name'] for a in (d.get('authors') or []))}"
            for d in docs
        ]
        scores = self.emd_model.get_score(query, texts, batch_size=12)
        paired = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        result = []
        for doc, score in paired:
            doc["rerank_score_bge"] = score
            result.append(doc)
        return result
