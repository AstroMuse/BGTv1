# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
QueryAgent — Query Understanding and Generation.

Responsibilities:
  1. Analyze user query intent, research domain, and suitable search sources.
  2. Build a compact 2-level domain tree (root → L1×3 → L2×9, 13 nodes total).
     Nodes are query-focused and siblings are mutually independent.
     No pruning step — the tree is built compact from the start.
  3. Extract L1+L2 vocabulary from the tree and inject it into query expansion,
     so expanded queries use precise domain terminology (Scheme 2).
  4. Generate new search queries from retrieved document context (context-to-query
     loop used by OrchestratorAgent between tree depths).

Public interface:
  expand_query(query)
      -> dict(query_intent, domain, suitable_sources, needs_expansion,
              expansion_reason, expanded_queries, domain_tree)
  generate_queries_from_docs(query, docs_with_nodes, searched_queries)
      -> List[(new_query_str, parent_node)]
"""

import json
import re
import time
import traceback
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from global_config import (
    LLM_MODEL_NAME,
    LLM_TRY_COUNT,
    LLM_PARREL_NUM,
    SLEEP_TIME_LLM,
    BUILD_DOMAIN_TREE,
    DOMAIN_TREE_DEPTH,
    FUSION_TEMPLATE,
)
from instruction import (
    template_query_intent,
    template_query_expand_judge_opt,
    template_query_fusion_survery_forcus,
    template_domain_aware_query_expansion,
    template_domain_aware_query_expansion_with_vocab,
    template_query_fusion_pasa,
    template_query_fusion_with_score_inst,
    template_query_fusion_with_score_user,
    template_domain_tree,
    template_context_query_generation,
    template_query_domain_complex,
)
from local_request_v2 import get_from_llm
from utils import fetch_string
from log import logger


# ---------------------------------------------------------------------------
# Module-level tree helpers (also imported by orchestrator for logging)
# ---------------------------------------------------------------------------

def render_domain_tree(node, prefix=""):
    """Render a {'name','children'} tree as indented text lines."""
    if not isinstance(node, dict):
        return [f"{prefix}<invalid node>"]
    lines = [f"{prefix}{node.get('name', '?')}"]
    for child in node.get("children", []) or []:
        lines.extend(render_domain_tree(child, prefix + "  "))
    return lines


def count_domain_tree(node):
    """Count nodes in a {'name','children'} tree."""
    if not isinstance(node, dict):
        return 0
    return 1 + sum(count_domain_tree(c) for c in (node.get("children") or []))


# ---------------------------------------------------------------------------
# QueryAgent
# ---------------------------------------------------------------------------

class QueryAgent:
    """
    Handles all query-related LLM calls: intent analysis, domain tree construction,
    vocabulary-augmented query expansion, and context-driven new query generation.
    """

    def __init__(self):
        self._survey_intent_cache: Dict[str, bool] = {}
        self._domain_complexity_cache: Dict[str, bool] = {}

    # ------------------------------------------------------------------
    # Primary entry points
    # ------------------------------------------------------------------

    def expand_query(self, query: str) -> dict:
        """
        Full query understanding pipeline.

        Returns a dict with keys:
          query_intent, domain, suitable_sources, needs_expansion,
          expansion_reason, expanded_queries, domain_tree
        """
        result = {
            "query_intent": "",
            "domain": "",
            "suitable_sources": [],
            "needs_expansion": False,
            "expansion_reason": "",
            "expanded_queries": [],
            "domain_tree": None,
        }

        # Step 1: intent + domain + source analysis
        try:
            intent_analysis = self._analyze_query_intent(query)
            if intent_analysis:
                result.update(intent_analysis)
            else:
                result["query_intent"] = "general research"
                result["domain"] = "undefined"
                result["suitable_sources"] = ["arxiv"]
        except Exception:
            logger.error(f"Error in query intent analysis: {traceback.format_exc()}")
            result["query_intent"] = "general research"
            result["domain"] = "undefined"
            result["suitable_sources"] = ["arxiv", "openalex"]

        # Step 1.5 (Idea 1 / Scheme 2): build compact domain tree, extract vocabulary
        domain_vocabulary: Dict[str, List[str]] = {}
        try:
            if not BUILD_DOMAIN_TREE:
                logger.info("BUILD_DOMAIN_TREE is off; skipping domain tree")
            else:
                domain_tree = self.build_domain_tree(
                    query, result["domain"], depth=DOMAIN_TREE_DEPTH
                )
                if domain_tree:
                    result["domain_tree"] = domain_tree
                    logger.info(
                        "Domain tree (%d nodes):\n%s"
                        % (
                            count_domain_tree(domain_tree),
                            "\n".join(render_domain_tree(domain_tree)),
                        )
                    )
                    domain_vocabulary = self._extract_domain_vocabulary(domain_tree)
                    logger.info(f"Domain vocabulary (L1→L2): {domain_vocabulary}")
                else:
                    logger.warning("Domain tree build returned nothing; vocabulary empty")
        except Exception:
            logger.error(f"Domain tree step failed: {traceback.format_exc()}")

        # Step 2: decide whether expansion is needed
        try:
            expansion_analysis = self._evaluate_expansion_need(query, result["domain"])
            if expansion_analysis:
                result["needs_expansion"] = expansion_analysis["needs_expansion"]
                result["expansion_reason"] = expansion_analysis["reason"]
            else:
                result["needs_expansion"] = False
                result["expansion_reason"] = "Analysis failed, keeping original query"
        except Exception:
            logger.error(f"Error in expansion analysis: {traceback.format_exc()}")
            result["needs_expansion"] = False

        # Step 3: generate expanded queries (domain vocabulary injected here)
        if result["needs_expansion"]:
            try:
                expanded = self._generate_expanded_queries(
                    query,
                    result["domain"],
                    result["query_intent"],
                    domain_vocabulary=domain_vocabulary,
                )
                result["expanded_queries"] = expanded
                logger.info(f"Generated {len(expanded)} expanded queries")
            except Exception:
                logger.error(f"Error generating expanded queries: {traceback.format_exc()}")
                result["expanded_queries"] = []

        return result

    def generate_queries_from_docs(
        self,
        query: str,
        docs_with_nodes: List,
        searched_queries: List[str],
    ) -> List[Tuple[str, Any]]:
        """
        Generate new search queries from retrieved document context.
        Used by OrchestratorAgent between tree depths.
        Returns List of (new_query_str, parent_node).
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=LLM_PARREL_NUM) as executor:
            future_to_pair = {
                executor.submit(
                    self._generate_query_from_reference, query, ref_doc, searched_queries
                ): [ref_doc, node]
                for ref_doc, node in docs_with_nodes
            }
            for future in concurrent.futures.as_completed(future_to_pair):
                ref_doc, node = future_to_pair[future]
                try:
                    new_queries = future.result(timeout=60)
                    if new_queries:
                        for new_q in new_queries:
                            results.append((new_q, node))
                except Exception:
                    logger.error(
                        f"generate_queries_from_docs future failed: {traceback.format_exc()}"
                    )
        return results

    # ------------------------------------------------------------------
    # Domain tree (build only — no prune step)
    # ------------------------------------------------------------------

    def build_domain_tree(self, query: str, domain: str, depth: int = 2) -> Optional[dict]:
        """
        Build a compact query-focused domain tree.
        With depth=2, 3 children per node: root + 3(L1) + 9(L2) = 13 nodes, ~390 tokens.
        Siblings at the same level are constrained to be mutually independent.
        """
        try:
            prompt = template_domain_tree.format(query=query, domain=domain, depth=depth)
            for _ in range(LLM_TRY_COUNT):
                try:
                    response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                    tree = json.loads(fetch_string(response))
                    if isinstance(tree, dict) and "name" in tree:
                        return tree
                    logger.warning(f"domain tree bad shape: {str(tree)[:200]}")
                except Exception:
                    logger.warning(
                        f"build_domain_tree attempt failed: {traceback.format_exc()}"
                    )
                    time.sleep(SLEEP_TIME_LLM)
        except Exception:
            logger.error(f"build_domain_tree failed: {traceback.format_exc()}")
        return None

    # ------------------------------------------------------------------
    # Domain vocabulary extraction (Scheme 2)
    # ------------------------------------------------------------------

    def _extract_domain_vocabulary(self, tree: dict) -> Dict[str, List[str]]:
        """
        Extract L1→L2 structure as a vocabulary dict, excluding the root.
        Returns {L1_name: [L2_name, ...], ...}
        """
        vocab: Dict[str, List[str]] = {}
        for l1_node in tree.get("children", []) or []:
            l1_name = l1_node.get("name", "").strip()
            if not l1_name:
                continue
            l2_names = [
                child.get("name", "").strip()
                for child in (l1_node.get("children", []) or [])
                if child.get("name", "").strip()
            ]
            vocab[l1_name] = l2_names
        return vocab

    def _format_domain_vocabulary(self, vocab: Dict[str, List[str]]) -> str:
        """
        Render the L1→L2 vocabulary dict as a human-readable bulleted string
        for injection into the LLM prompt.
        """
        if not vocab:
            return "(no domain vocabulary available)"
        lines = []
        for l1_name, l2_names in vocab.items():
            lines.append(f"• {l1_name}")
            for l2 in l2_names:
                lines.append(f"  - {l2}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _analyze_query_intent(self, query: str) -> Optional[dict]:
        current_year = datetime.now().year
        previous_year = current_year - 1
        prompt = template_query_intent.format(
            query=query, current_year=current_year, previous_year=previous_year
        )
        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                result = json.loads(fetch_string(response))
                if all(
                    k in result
                    for k in ["query_intent", "domain", "suitable_sources", "source_reason"]
                ):
                    return result
                logger.warning(f"Incomplete intent response: {result}")
            except Exception:
                logger.warning(
                    f"Intent analysis attempt {attempt+1} failed: {traceback.format_exc()}"
                )
                time.sleep(SLEEP_TIME_LLM)
        return None

    def _evaluate_expansion_need(self, query: str, domain: str) -> Optional[dict]:
        prompt = template_query_expand_judge_opt.format(query=query, domain=domain)
        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                result = json.loads(fetch_string(response))
                if "needs_expansion" in result and "reason" in result:
                    return result
                logger.warning(f"Incomplete expansion response: {result}")
            except Exception:
                logger.warning(
                    f"Expansion eval attempt {attempt+1} failed: {traceback.format_exc()}"
                )
                time.sleep(SLEEP_TIME_LLM)
        return None

    def _generate_expanded_queries(
        self,
        query: str,
        domain: str,
        intent: str,
        domain_vocabulary: Dict[str, List[str]] = None,
    ) -> List[str]:
        """
        Generate expanded queries.
        When domain_vocabulary is provided (Scheme 2), uses the vocabulary-augmented
        template so that queries use precise L2-level domain terminology.
        Falls back to the standard template when vocabulary is absent.
        """
        current_year = datetime.now().year
        previous_year = current_year - 1

        # --- choose template and prompt ---
        if domain_vocabulary and FUSION_TEMPLATE == "AUTOMATIC":
            vocab_text = self._format_domain_vocabulary(domain_vocabulary)
            prompt = template_domain_aware_query_expansion_with_vocab.format(
                user_input_N=5,
                user_query=query,
                intent=intent,
                domain=domain,
                domain_vocabulary_text=vocab_text,
                current_year=current_year,
                previous_year=previous_year,
            )
            prompt_type = "domain_vocab"
            logger.info("Using vocabulary-augmented expansion (Scheme 2)")

        elif FUSION_TEMPLATE == "AUTOMATIC" and self._is_survey_focused(intent):
            prompt = template_query_fusion_survery_forcus.format(
                user_query=query,
                user_input_N=5,
                current_year=current_year,
                previous_year=previous_year,
            )
            prompt_type = "survey"

        elif FUSION_TEMPLATE == "AUTOMATIC" and self._is_complex_domain(domain):
            prompt = template_domain_aware_query_expansion.format(
                user_input_N=5,
                user_query=query,
                intent=intent,
                domain=domain,
                current_year=current_year,
                previous_year=previous_year,
            )
            prompt_type = "domain"

        elif FUSION_TEMPLATE == "PASA":
            prompt = template_query_fusion_pasa.format(user_query=query)
            prompt_type = "pasa"

        elif FUSION_TEMPLATE == "WITHEXPLAIN":
            prompt = (
                template_query_fusion_with_score_inst
                + template_query_fusion_with_score_user.format(
                    user_query=query, user_input_N=5
                )
            )
            prompt_type = "withexplain"

        else:
            prompt = template_domain_aware_query_expansion.format(
                user_input_N=5,
                user_query=query,
                intent=intent,
                domain=domain,
                current_year=current_year,
                previous_year=previous_year,
            )
            prompt_type = "domain"

        # --- call LLM with retry, keep best result ---
        best_response = None
        best_count = 0

        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                response = fetch_string(response)
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", response, re.DOTALL)
                    parsed = json.loads(match.group(0)) if match else None
                    if parsed is None:
                        match = re.search(r"\[.*\]", response, re.DOTALL)
                        parsed = json.loads(match.group(0)) if match else None
                if parsed is None:
                    continue
                expanded = self._extract_queries_from_response(parsed, prompt_type)
                if expanded and len(expanded) > best_count:
                    best_response = expanded
                    best_count = len(expanded)
                if best_count >= 3:
                    break
            except Exception:
                logger.warning(
                    f"Expand attempt {attempt+1} failed: {traceback.format_exc()}"
                )
                time.sleep(SLEEP_TIME_LLM)

        if best_response:
            return best_response
        logger.error("All expand attempts failed, using fallback")
        return self._generate_fallback_queries(query, domain)

    def _extract_queries_from_response(self, response, prompt_type: str) -> List[str]:
        queries = []
        try:
            if isinstance(response, list):
                queries = [q for q in response if isinstance(q, str)]
            elif isinstance(response, dict):
                if "expanded_queries" in response:
                    for item in response["expanded_queries"]:
                        if isinstance(item, str):
                            queries.append(item)
                        elif isinstance(item, dict) and "query" in item:
                            queries.append(item["query"])
                elif prompt_type == "withexplain" and "rewritten_queries" in response:
                    for item in response["rewritten_queries"]:
                        if isinstance(item, dict) and "rewritten_query" in item:
                            queries.append(item["rewritten_query"])
        except Exception:
            logger.error(f"Error extracting queries: {traceback.format_exc()}")
        return queries

    def _generate_query_from_reference(
        self, user_query: str, one_doc: dict, searched_queries: List[str]
    ) -> List[str]:
        """Generate new search queries from a single retrieved document."""
        model_inp = template_context_query_generation.format(
            user_query=user_query,
            searched_queries=searched_queries,
            doc_title=one_doc.get("title", ""),
            doc_abstract=one_doc.get("abstract", ""),
            doc_field=one_doc.get("fieldsOfStudy", ""),
        )
        for _ in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
                response = fetch_string(response)
                query_list = json.loads(response)
                return [q for q in query_list if q and q not in searched_queries]
            except Exception:
                logger.error(
                    f"_generate_query_from_reference failed: {traceback.format_exc()}"
                )
                time.sleep(SLEEP_TIME_LLM)
        return []

    def _generate_fallback_queries(self, query: str, domain: str) -> List[str]:
        return [
            f"survey papers on {query}",
            f"literature review {query}",
            f"state-of-the-art {query}",
            f"recent advances in {query}",
            f"{domain} {query} methodologies",
        ]

    def _generate_emergency_fallback_queries(self, query: str) -> List[str]:
        return [
            f"survey papers on {query}",
            f"literature review {query}",
            f"state-of-the-art {query}",
        ]

    # ------------------------------------------------------------------
    # Domain / intent classifiers (fast keyword path + LLM slow path)
    # ------------------------------------------------------------------

    def _is_survey_focused(self, intent: str) -> bool:
        intent_lower = intent.lower()
        survey_indicators = [
            "survey", "review", "overview", "state-of-the-art",
            "literature", "comprehensive", "summary", "taxonomy",
            "comparative", "meta-analysis",
        ]
        if any(ind in intent_lower for ind in survey_indicators):
            return True
        implicit_patterns = [
            r"what (is|are) the current",
            r"(summarize|summarizing) (recent|current)",
            r"broad (understanding|overview)",
            r"comprehensive (analysis|study)",
            r"(existing|available) (approaches|methods)",
            r"compare (different|various)",
            r"trends in",
        ]
        if any(re.search(p, intent_lower) for p in implicit_patterns):
            return True
        contextual_pairs = [
            ("literature", "field"), ("papers", "compare"), ("research", "directions"),
            ("developments", "field"), ("comprehensive", "understanding"),
            ("overview", "approaches"), ("different", "techniques"),
            ("evolution", "development"), ("progress", "area"), ("history", "development"),
        ]
        if any(all(t in intent_lower for t in pair) for pair in contextual_pairs):
            return True
        cache_key = f"survey_intent:{intent_lower}"
        if cache_key in self._survey_intent_cache:
            return self._survey_intent_cache[cache_key]
        try:
            prompt = (
                f'Determine if this academic research intent primarily seeks SURVEY or REVIEW '
                f'papers rather than primary research:\n\nIntent: "{intent}"\n\n'
                f'Respond only "Yes" or "No".'
            )
            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            is_survey = "yes" in response.lower()
            self._survey_intent_cache[cache_key] = is_survey
            return is_survey
        except Exception:
            return "overview" in intent_lower or "review" in intent_lower

    def _is_complex_domain(self, domain: str) -> bool:
        domain_lower = domain.lower()
        known_complex = {
            "quantum computing", "genomics", "bioinformatics", "neuroscience",
            "computational linguistics", "cryptography", "nanomaterials",
            "immunology", "pharmacology", "astrophysics", "high energy physics",
            "theoretical computer science", "robotics", "material science",
        }
        if any(c in domain_lower for c in known_complex):
            return True
        technical_indicators = [
            "quantum", "computational", "theoretical", "stochastic", "bayesian"
        ]
        if any(ind in domain_lower for ind in technical_indicators):
            return True
        if len(domain_lower.split()) >= 3:
            return True
        cache_key = f"domain_complexity:{domain_lower}"
        if cache_key in self._domain_complexity_cache:
            return self._domain_complexity_cache[cache_key]
        try:
            prompt = template_query_domain_complex.format(domain=domain)
            response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
            is_complex = "yes" in response.lower()
            self._domain_complexity_cache[cache_key] = is_complex
            return is_complex
        except Exception:
            return len(domain_lower.split()) >= 2
