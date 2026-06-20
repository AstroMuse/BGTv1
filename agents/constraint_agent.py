# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
ConstraintAgent (Agent C) — multi-dimensional intent-constraint decomposition.

A parallel candidate route that runs beside the main retrieval at tree depth 1:

  1. LLM decomposes the query into a structured constraint spec — time, domain,
     literature type, keywords, and (only if needed) up to two free extension
     dimensions; each dimension is left empty when the query does not constrain
     it (default = minimum constraint).
  2. A second LLM pass ranks each constraint HARD vs SOFT, emits the HARD
     constraints that are natively parametrizable (currently the time range) as
     a "params" dict, and folds every non-parametrizable HARD constraint plus
     the SOFT constraints into ONE rewritten natural-language query ("new_query").
  3. Search Semantic Scholar + Google(arXiv via Serper) + OpenAlex concurrently
     with that query / params, drop low-impact papers (citation count known and
     below AGENT_C_MIN_CITATION), and return the survivors (E_raw).

The orchestrator merges E_raw into the Phase-1 scoring pool (B_raw): ids already
present are deduped away, the rest are scored and expanded like any retrieved doc.

Public interface:
  search(query, end_date) -> List[Dict]   # E_raw candidate docs
"""

import json
import traceback
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any

from global_config import (
    LLM_MODEL_NAME,
    LLM_TRY_COUNT,
    API_TRY_COUNT,
    AGENT_C_MIN_CITATION,
)
from instruction import (
    template_agentc_decompose,
    template_agentc_priority,
)
from local_request_v2 import get_from_llm
from utils import fetch_string
from api_web import (
    search_paper_via_query_from_openalex,
    search_paper_via_query_from_semantic,
    google_search_arxiv_id,
    parallel_search_search_paper_from_arxiv,
)
from log import logger


class ConstraintAgent:
    """Agent C: intent-constraint-driven structured search across 3 sources."""

    def __init__(self, min_citation: int = AGENT_C_MIN_CITATION):
        self.min_citation = min_citation

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, query: str, end_date: str = "") -> List[Dict]:
        try:
            constraints = self._decompose(query, end_date)
        except Exception:
            logger.error(f"AgentC decompose failed: {traceback.format_exc()}")
            constraints = {}
        if not constraints:
            logger.info("AgentC: decomposition empty, skipping")
            return []

        plan = self._prioritize(query, constraints)
        new_query = plan.get("new_query") or query
        keywords = constraints.get("keywords") or []
        eff_end_date = self._effective_end_date(plan.get("params", {}), end_date)
        logger.info(
            f"AgentC plan: new_query={new_query!r} | end_date={eff_end_date or '(none)'} "
            f"| keywords={keywords}"
        )

        raw = self._multi_source_search(new_query, keywords, eff_end_date)
        e_raw = self._citation_filter(raw)
        logger.info(
            f"AgentC: {len(raw)} raw candidates -> {len(e_raw)} after citation>={self.min_citation} filter"
        )
        return e_raw

    # ------------------------------------------------------------------
    # Step 1: decompose into constraint JSON
    # ------------------------------------------------------------------

    def _decompose(self, query: str, end_date: str) -> Dict[str, Any]:
        search_time = self._fmt_date(end_date) or datetime.now().strftime("%Y-%m-%d")
        prompt = template_agentc_decompose.format(userQuery=query, searchTime=search_time)
        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                data = json.loads(fetch_string(response).strip())
                if isinstance(data, dict):
                    return data
            except Exception:
                logger.warning(f"AgentC decompose attempt {attempt+1} failed: {traceback.format_exc()}")
        return {}

    # ------------------------------------------------------------------
    # Step 2: hard/soft priority + parametrize + rewrite
    # ------------------------------------------------------------------

    def _prioritize(self, query: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
        prompt = template_agentc_priority.format(
            userQuery=query,
            constraints=json.dumps(constraints, ensure_ascii=False),
        )
        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=LLM_MODEL_NAME)
                data = json.loads(fetch_string(response).strip())
                if isinstance(data, dict) and data.get("new_query"):
                    data.setdefault("params", {})
                    return data
            except Exception:
                logger.warning(f"AgentC prioritize attempt {attempt+1} failed: {traceback.format_exc()}")
        # Fallback: no rewrite, reuse the time constraint straight from decomposition.
        tc = (constraints.get("time_constraint") or {}) if isinstance(constraints, dict) else {}
        return {
            "params": {"start_date": tc.get("start", ""), "end_date": tc.get("end", "")},
            "new_query": query,
            "source_query": query,
        }

    # ------------------------------------------------------------------
    # Step 3: 3-source search + citation filter
    # ------------------------------------------------------------------

    def _multi_source_search(
        self, new_query: str, keywords: List[str], end_date: str
    ) -> Dict[str, Dict]:
        """Search Semantic Scholar + arXiv(Serper) + OpenAlex concurrently.

        OpenAlex is keyword-based (poor on natural language), so it gets the
        distilled keyword string; Semantic/arXiv get the rewritten NL query.
        """
        kw_str = ", ".join(k for k in keywords if k) or new_query

        def _semantic():
            try:
                return search_paper_via_query_from_semantic(
                    new_query, max_paper_num=15, end_date=end_date or None
                )
            except Exception:
                logger.error(f"AgentC semantic failed: {traceback.format_exc()}")
                return {}

        def _openalex():
            try:
                return search_paper_via_query_from_openalex(
                    kw_str, per_page=10, end_date=end_date or ""
                )
            except Exception:
                logger.error(f"AgentC openalex failed: {traceback.format_exc()}")
                return {}

        def _arxiv():
            try:
                ids = google_search_arxiv_id(
                    new_query, try_num=API_TRY_COUNT, num=10, end_date=end_date or ""
                )
                if not ids:
                    return {}
                return parallel_search_search_paper_from_arxiv(list(set(ids)), max_workers=1)
            except Exception:
                logger.error(f"AgentC arxiv failed: {traceback.format_exc()}")
                return {}

        merged: Dict[str, Dict] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
            futs = {
                ex.submit(_semantic): "semantic",
                ex.submit(_openalex): "openalex",
                ex.submit(_arxiv): "arxiv",
            }
            for fut in concurrent.futures.as_completed(futs):
                src = futs[fut]
                try:
                    docs = fut.result(timeout=60) or {}
                except Exception:
                    logger.error(f"AgentC {src} future failed: {traceback.format_exc()}")
                    docs = {}
                for pid, doc in docs.items():
                    if not isinstance(doc, dict):
                        continue
                    pid = doc.get("paper_id") or doc.get("arxivId") or pid
                    if not pid or pid in merged:
                        continue
                    doc["paper_id"] = pid
                    merged[pid] = doc
        return merged

    def _citation_filter(self, docs: Dict[str, Dict]) -> List[Dict]:
        """Drop docs whose citation count is KNOWN and below the threshold.

        Docs with no citation info (typically arXiv-resolved) are kept — they
        cannot be judged low-impact, and dropping them would void the arXiv source.
        """
        out = []
        for doc in docs.values():
            cc = doc.get("citationCount")
            if cc is None:
                cc = doc.get("cited_by_count")
            if cc is not None:
                try:
                    if int(cc) < self.min_citation:
                        continue
                except (TypeError, ValueError):
                    pass
            doc["source"] = "Search From AgentC"
            out.append(doc)
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_date(yyyymmdd: str) -> str:
        if not yyyymmdd:
            return ""
        try:
            return datetime.strptime(yyyymmdd, "%Y%m%d").strftime("%Y-%m-%d")
        except Exception:
            return ""

    def _effective_end_date(self, params: Dict[str, Any], run_end_date: str) -> str:
        """Prefer the LLM-extracted end_date (YYYY-MM-DD -> YYYYMMDD); else the
        run's benchmark end_date (already YYYYMMDD)."""
        p_end = (params or {}).get("end_date") or ""
        if p_end:
            try:
                return datetime.strptime(p_end, "%Y-%m-%d").strftime("%Y%m%d")
            except Exception:
                logger.warning(f"AgentC: bad params end_date {p_end!r}, using run end_date")
        return run_end_date or ""
