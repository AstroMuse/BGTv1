# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
RetrievalAgent — Multi-source Parallel Academic Paper Search.

Responsibilities:
  - Execute parallel searches across arXiv (via Google Serper), OpenAlex,
    PubMed, and Semantic Scholar.
  - Extract source-specific keywords (LLM-assisted) for structured databases.
  - Deduplicate and merge results from multiple sources and queries.

Public interface:
  search(queries, sources, end_date, searched_docs)
      -> (query2docs, id2docs, query_source_map, query_keywords2raw)
"""

import concurrent.futures
import traceback
import time
from collections import defaultdict
from typing import List, Dict, Any, Optional, Set

from global_config import (
    LLM_MODEL_NAME,
    API_TRY_COUNT,
    API_PARALLEL_REQUEST,
    KEY_WORDS_NUM,
)
from instruction import template_extract_keywords_source_aware
from local_request_v2 import get_from_llm
from base_class import SearchResult
from api_web import (
    google_search_arxiv_id,
    search_paper_via_query_from_openalex,
    search_paper_via_query_from_semantic,
    search_from_pubmed,
    parallel_search_search_paper_from_arxiv,
)
from local_db_v2 import db_path, ArxivDatabase
from log import logger
import re
import os


def get_info_from_local(id_list: List[str]):
    """Look up paper info from the local SQLite cache."""
    already = []
    to_process = []
    if os.path.exists(db_path):
        with ArxivDatabase(db_path) as db:
            for _id in id_list:
                db_info = db.get(_id)
                if db_info is None:
                    to_process.append(_id)
                else:
                    already.append(_id)
        logger.info(f"Local cache: {len(already)} hit, {len(to_process)} miss")
        return already, to_process
    return [], id_list


class RetrievalAgent:
    """
    Parallel multi-source academic paper retrieval.
    Replaces the former MultiSearchAgent with a cleaner interface.
    """

    def __init__(self, max_workers: int = 3, batch_size: int = 10):
        self.max_workers = max_workers
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def search(
        self,
        queries: List[str],
        sources: List[str] = None,
        end_date: str = "",
        searched_docs: Dict[str, Any] = None,
    ):
        """
        Execute parallel search across multiple sources for all queries.

        Returns:
          query2docs       – {query_str: [doc, ...]}
          id2docs          – {paper_id: doc}
          query_source_map – {query_str: source_name}
          query_keywords2raw – {keyword_str: original_query}
        """
        if sources is None:
            sources = ["arxiv"]
        if searched_docs is None:
            searched_docs = {}
        if not queries:
            logger.error("Empty query list")
            return {}, {}, {}, {}

        keyword_extraction_sources = {"openalex", "pubmed"}
        valid_sources = [s for s in sources if s in {"arxiv", "openalex", "pubmed", "semantic"}]
        if not valid_sources:
            logger.error(f"No valid sources in: {sources}")
            return {}, {}, {}, {}

        # Step 1: parallel keyword extraction for structured databases
        query_keywords_by_source: Dict[str, Dict[str, List[str]]] = {}
        if set(valid_sources) & keyword_extraction_sources:
            # Collect all (source, query) tasks that need LLM keyword extraction
            extraction_tasks = [
                (source, query)
                for source in valid_sources
                if source in keyword_extraction_sources
                for query in queries
            ]

            # Fire all LLM keyword-extraction calls in parallel
            raw_kw_results: Dict[tuple, List[str]] = {}
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(len(extraction_tasks), 8)
            ) as kw_executor:
                future_to_task = {
                    kw_executor.submit(self._extract_keywords, query, source): (source, query)
                    for source, query in extraction_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    source, query = future_to_task[future]
                    try:
                        raw_kw_results[(source, query)] = future.result(timeout=30)
                    except Exception:
                        logger.error(
                            f"Keyword extraction timed out for ({source}, {query})"
                        )
                        raw_kw_results[(source, query)] = []

            # Apply cross-query deduplication per source in original query order
            for source in valid_sources:
                if source not in keyword_extraction_sources:
                    continue
                query_keywords_by_source[source] = {}
                seen: List[str] = []
                for query in queries:
                    kws = raw_kw_results.get((source, query), [])
                    unique_kws = [k for k in kws if k not in seen]
                    seen.extend(unique_kws)
                    query_keywords_by_source[source][query] = unique_kws if unique_kws else [query]

        # Step 2: submit all search tasks in parallel
        search_funcs = {
            "arxiv": self._arxiv_search,
            "semantic": self._semantic_search,
            "openalex": self._openalex_search,
            "pubmed": self._pubmed_search,
        }
        tasks = []
        future_to_meta = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for source in valid_sources:
                if source == "arxiv":
                    future = executor.submit(self._arxiv_search, queries, end_date, searched_docs)
                    future_to_meta[future] = (source, "arxiv_batch")
                    tasks.append(future)
                elif source in keyword_extraction_sources:
                    for query in queries:
                        for kw in query_keywords_by_source.get(source, {}).get(query, [query]):
                            future = executor.submit(search_funcs[source], kw, query, end_date)
                            future_to_meta[future] = (source, f"{query[:20]}:{kw}")
                            tasks.append(future)
                elif source == "semantic":
                    for query in queries:
                        future = executor.submit(search_funcs[source], query, query, end_date)
                        future_to_meta[future] = (source, query[:30])
                        tasks.append(future)

            results_by_source: Dict[str, List[SearchResult]] = {s: [] for s in valid_sources}
            for future in concurrent.futures.as_completed(tasks):
                source, label = future_to_meta[future]
                try:
                    result = future.result(timeout=30)
                    results_by_source[source].append(result)
                    logger.info(f"Completed {source} search for '{label}'")
                except concurrent.futures.TimeoutError:
                    logger.error(f"{source} search timed out for '{label}'")
                except Exception:
                    logger.error(f"{source} search failed for '{label}': {traceback.format_exc()}")

        # Step 3: merge within each source
        merged: Dict[str, SearchResult] = {}
        for source in valid_sources:
            if not results_by_source[source]:
                continue
            if source == "arxiv":
                merged[source] = results_by_source[source][0]
            else:
                merged[source] = self._merge_results_grouped(results_by_source[source], source)

        # Step 4: unify across sources
        final_query2docs: Dict[str, List] = {}
        final_id2docs: Dict[str, Any] = {}
        query_source_map: Dict[str, str] = {}
        query_keywords2raw: Dict[str, str] = {}

        for source, result in merged.items():
            if not result.query2paper:
                continue
            if source == "arxiv":
                for query, papers in result.query2paper.items():
                    final_query2docs.setdefault(query, []).extend(papers)
                    query_keywords2raw[query] = query
                    query_source_map[query] = source
                    final_id2docs.update(
                        {p.get("paper_id", p.get("arxivId", "")): p for p in papers}
                    )
            else:
                for query, papers in result.query2paper.items():
                    kw_strs = result.extra.get("merged_query_to_keywords", {}).get(query, [])
                    kw_key = "|".join(kw_strs)
                    final_query2docs.setdefault(kw_key, []).extend(papers)
                    query_keywords2raw[kw_key] = query
                    query_source_map[kw_key] = source
                    final_id2docs.update({p["paper_id"]: p for p in papers if "paper_id" in p})

        logger.info(f"Total retrieved papers: {len(final_id2docs)}")
        return final_query2docs, final_id2docs, query_source_map, query_keywords2raw

    # ------------------------------------------------------------------
    # Per-source search methods
    # ------------------------------------------------------------------

    def _arxiv_search(
        self,
        queries: List[str],
        end_date: str = "",
        searched_docs: Dict[str, Any] = None,
    ) -> SearchResult:
        if searched_docs is None:
            searched_docs = {}
        merged_papers = {}
        query2docs = {q: [] for q in queries}
        results = {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_q = {
                    executor.submit(google_search_arxiv_id, q, API_TRY_COUNT, 10, end_date): q
                    for q in queries
                }
                for future in concurrent.futures.as_completed(future_to_q):
                    q = future_to_q[future]
                    try:
                        results[q] = future.result(timeout=30)
                    except Exception:
                        logger.error(f"arXiv search failed for '{q}': {traceback.format_exc()}")
                        results[q] = []

            unique_arxiv: Set[str] = set()
            for arxiv_ids in results.values():
                for aid in arxiv_ids:
                    if aid not in searched_docs:
                        unique_arxiv.add(aid)

            id2docs = {}
            if unique_arxiv:
                # max_workers=1: export.arxiv.org must be hit on a single
                # connection (one request / 3s) or it returns HTTP 429.
                id2docs = parallel_search_search_paper_from_arxiv(
                    list(unique_arxiv), max_workers=1, batch_size=self.batch_size
                )

            requeued = 0
            for q, arxiv_ids in results.items():
                for aid in arxiv_ids:
                    if aid in id2docs:
                        merged_papers[aid] = id2docs[aid]
                        query2docs[q].append(id2docs[aid])
                    elif aid in searched_docs and "sim_score" not in searched_docs[aid]:
                        # Known but never scored (e.g. a Phase-2 ref that was
                        # pruned out before LLM scoring): reuse the stored
                        # metadata so this query's scoring batch picks it up.
                        # Presence in searched_docs is NOT proof it was scored —
                        # only skip ids whose entry already carries a sim_score.
                        merged_papers[aid] = searched_docs[aid]
                        query2docs[q].append(searched_docs[aid])
                        requeued += 1
            if requeued:
                logger.info(
                    f"arXiv dedup: requeued {requeued} known-but-unscored docs for scoring"
                )
        except Exception:
            logger.error(f"_arxiv_search error: {traceback.format_exc()}")
        return SearchResult(source="arxiv", papers=merged_papers, query2paper=query2docs)

    def _openalex_search(
        self, keyword: str, raw_query: str, end_date: str = ""
    ) -> SearchResult:
        logger.info(f"OpenAlex search for '{keyword}'")
        try:
            papers = search_paper_via_query_from_openalex(keyword, per_page=10, end_date=end_date)
            logger.info(f"OpenAlex: {len(papers)} papers for '{keyword}'")
            return SearchResult(source="openalex", papers=papers, keyword=keyword, raw_query=raw_query)
        except Exception:
            logger.error(f"OpenAlex search failed: {traceback.format_exc()}")
            return SearchResult(source="openalex", papers={}, raw_query=raw_query)

    def _pubmed_search(
        self, keyword: str, raw_query: str, end_date: str = "", max_results: int = 10
    ) -> SearchResult:
        logger.info(f"PubMed search for '{keyword}'")
        try:
            papers = search_from_pubmed(keyword, max_results=max_results)
            logger.info(f"PubMed: {len(papers)} papers for '{keyword}'")
            return SearchResult(
                source="pubmed",
                papers={p["paper_id"]: p for p in papers},
                keyword=keyword,
                raw_query=raw_query,
            )
        except Exception:
            logger.error(f"PubMed search failed: {traceback.format_exc()}")
            return SearchResult(source="pubmed", papers={}, raw_query=raw_query)

    def _semantic_search(
        self, keyword: str, raw_query: str, end_date: str = "", max_papers: int = 15
    ) -> SearchResult:
        logger.info(f"Semantic Scholar search for '{keyword}'")
        try:
            papers = search_paper_via_query_from_semantic(
                query=keyword, max_paper_num=max_papers, end_date=end_date or None
            )
            logger.info(f"Semantic: {len(papers)} papers for '{keyword}'")
            return SearchResult(source="semantic", papers=papers, keyword=keyword, raw_query=raw_query)
        except Exception:
            logger.error(f"Semantic search failed: {traceback.format_exc()}")
            return SearchResult(source="semantic", papers={}, raw_query=raw_query)

    # ------------------------------------------------------------------
    # Merging helpers
    # ------------------------------------------------------------------

    def _merge_results_grouped(
        self, results: List[SearchResult], source: str
    ) -> SearchResult:
        """Merge results from the same source, grouped by raw_query."""
        grouped: Dict[str, List[SearchResult]] = defaultdict(list)
        for r in results:
            if r.raw_query:
                grouped[r.raw_query].append(r)

        merged_query2paper: Dict[str, List] = {}
        merged_kw_map: Dict[str, List[str]] = {}

        for raw_query, group in grouped.items():
            merged_query2paper[raw_query] = []
            merged_kw_map[raw_query] = []
            for r in group:
                if not r.papers:
                    continue
                merged_query2paper[raw_query].extend(list(r.papers.values()))
                if r.keyword:
                    merged_kw_map[raw_query].append(r.keyword)

        return SearchResult(
            source=source,
            papers=[],
            query2paper=merged_query2paper,
            extra={"merged_query_to_keywords": merged_kw_map},
        )

    def _merge_paper_info(
        self, existing: Dict[str, Any], new: Dict[str, Any], source: str
    ) -> Dict[str, Any]:
        merged = existing.copy()
        for field in ["abstract", "title", "publicationYear", "authors", "fieldsOfStudy"]:
            if not merged.get(field):
                merged[field] = new.get(field)
        merged.setdefault("sources", [existing.get("source", "unknown")])
        merged["sources"] = list(set(merged["sources"] + [source]))
        for score_field in ["citationCount", "referenceCount"]:
            if score_field in new:
                merged[score_field] = max(merged.get(score_field, 0), new[score_field])
        return merged

    # ------------------------------------------------------------------
    # Keyword extraction
    # ------------------------------------------------------------------

    def _extract_keywords(self, query: str, source: str) -> List[str]:
        model_inp = template_extract_keywords_source_aware.format(
            user_query=query.lower(), source=source
        )
        for _ in range(4):
            try:
                response = get_from_llm(model_inp, model_name=LLM_MODEL_NAME)
                match = re.search(r"\[Start\](.*?)\[End\]", response)
                if match:
                    return [kw.strip() for kw in match.group(1).split(",") if kw.strip()][:KEY_WORDS_NUM]
            except Exception:
                logger.error(f"Keyword extraction failed: {traceback.format_exc()}")
        return []


# Backward-compatible alias for demo_app_with_front.py
MultiSearchAgent = RetrievalAgent
