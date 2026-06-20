# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
ReferenceAgent — Citation Graph Expansion.

Responsibilities:
  - Fetch the reference list of a retrieved document from external APIs
    (Semantic Scholar for arXiv papers, PubMed, or OpenAlex depending on source).
  - Provides the raw references back to OrchestratorAgent, which then passes
    them to ScoringAgent for relevance scoring.

Public interface:
  get_references(doc_info) -> doc_info  (with "references" key populated)
"""

import traceback
from typing import Dict, Any

from api_web import (
    get_doc_info_from_semantic_scholar_by_arxivid,
    get_openalex_referenced_works,
    search_doc_via_url_parallel_openalex,
    search_doc_via_url_from_openalex,
    fetch_pubmed_json,
)
from global_config import REFERENCE_API, CITATION_STORE_PATH
from local_db_v2 import db_path, ArxivDatabase
from log import logger


class ReferenceAgent:
    """
    Fetches the reference list for a retrieved paper.
    Supports arXiv (via OpenAlex or Semantic Scholar), PubMed, and OpenAlex sources.

    For arXiv papers the citation API is selectable (global_config.REFERENCE_API /
    run_spr_agent.py --ref-api): "openalex" (default), "s2", or "local" (offline
    arXiv citation LMDB). Whichever is chosen is tried first; the others are used
    as fallbacks if it returns no references (e.g. S2 throttled/arrears, OpenAlex
    missing the work, or the local store lacking that paper).
    """

    def __init__(self, ref_api: str = None):
        self.ref_api = (ref_api or REFERENCE_API or "openalex").lower()
        self._citation_store = None  # lazy: only built when ref_api == "local"

    def get_references(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Populate doc_info["references"] by querying the appropriate external API.
        Returns the updated doc_info dict.
        """
        try:
            if doc_info.get("arxivId"):
                if self.ref_api == "local":
                    # offline graph first; S2 then OpenAlex as online fallbacks
                    chain = [
                        self._fetch_via_local_citation_store,
                        self._fetch_via_semantic_scholar,
                        self._fetch_via_openalex_arxiv,
                    ]
                elif self.ref_api == "s2":
                    chain = [self._fetch_via_semantic_scholar, self._fetch_via_openalex_arxiv]
                else:
                    chain = [self._fetch_via_openalex_arxiv, self._fetch_via_semantic_scholar]
                for i, fetch in enumerate(chain):
                    doc_info = fetch(doc_info)
                    if doc_info.get("references"):
                        break
                    if i + 1 < len(chain):
                        logger.info(f"{self.ref_api}: source {i} returned no refs; trying fallback")
                return doc_info
            elif "PMID" in doc_info:
                return self._fetch_via_pubmed(doc_info)
            elif doc_info.get("referenceWorksOpenAlex"):
                doc_info = self._fetch_via_openalex(doc_info)
                # Fallback (mirrors the arXiv path): if the OpenAlex reference
                # batch yielded nothing, try S2 — but only when the work carries
                # an arXiv handle. Pure OpenAlex works have no arxiv/doi/pmid to
                # query S2 by, so there is nothing to fall back to for those.
                if not doc_info.get("references"):
                    if doc_info.get("arxivId"):
                        logger.info("openalex refs empty; trying S2 fallback")
                        doc_info = self._fetch_via_semantic_scholar(doc_info)
                    else:
                        logger.info(
                            "openalex refs empty and no arxiv/doi handle; "
                            "no fallback available"
                        )
                return doc_info
        except Exception:
            logger.error(f"ReferenceAgent.get_references failed: {traceback.format_exc()}")
        return doc_info

    # ------------------------------------------------------------------
    # Per-source reference fetchers
    # ------------------------------------------------------------------

    @property
    def citation_store(self):
        if self._citation_store is None:
            from citation_store import CitationStore
            self._citation_store = CitationStore(CITATION_STORE_PATH)
        return self._citation_store

    def _fetch_via_local_citation_store(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Reference list from the offline arXiv citation LMDB.

        Forward edges (cited arXiv ids) come from the local graph; each ref's
        title/abstract is resolved from recovered.db only (no network) — refs
        missing locally are dropped, leaving references empty so get_references
        falls back to an online source.
        """
        try:
            from citation_store import CitationStore
            if not CitationStore.exists(CITATION_STORE_PATH):
                return doc_info
            ref_ids = self.citation_store.references(doc_info["arxivId"])
            if not ref_ids:
                return doc_info
            refs = []
            with ArxivDatabase(db_path) as db:
                for rid in ref_ids:
                    meta = db.get(rid)
                    if meta and meta.get("title") and meta.get("abstract"):
                        # Drop the ref's OWN references/citations: a reference
                        # entry must stay flat (title/abstract only), never carry
                        # its own reference list, or the parent doc re-nests in
                        # memory (the recovered.db bloat / OOM class of bug).
                        meta = {k: v for k, v in meta.items()
                                if k not in ("references", "citations")}
                        meta["paper_id"] = rid
                        meta.setdefault("arxivId", rid)
                        meta["source"] = "Search From Local"
                        refs.append(meta)
            doc_info["references"] = refs
            logger.info(
                f"Local citation store for {doc_info.get('arxivId')}: "
                f"{len(ref_ids)} cited -> {len(refs)} resolved from recovered.db"
            )
        except Exception:
            logger.error(f"_fetch_via_local_citation_store failed: {traceback.format_exc()}")
        return doc_info

    def _fetch_via_semantic_scholar(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        # S2 free tier has intermittent 429s; use more retries
        doc_info_new = get_doc_info_from_semantic_scholar_by_arxivid(
            doc_info["arxivId"], try_num=6
        )
        if doc_info_new is not None:
            doc_info.update(doc_info_new)
        return doc_info

    def _fetch_via_openalex_arxiv(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Reference list for an arXiv paper via OpenAlex (no per-second throttle).

        OpenAlex returns each referenced work already carrying title + abstract,
        so — unlike the S2 path — no extra arXiv metadata round-trip is needed.
        """
        ref_ids = get_openalex_referenced_works(doc_info["arxivId"])
        refs = search_doc_via_url_parallel_openalex(ref_ids) if ref_ids else []
        normalized = []
        for r in refs:
            # OpenAlex work id (e.g. "W123...") becomes the cache/dedup key.
            oid = (r.get("openalex_url") or "").rstrip("/").split("/")[-1]
            if not oid:
                continue
            r["paper_id"] = oid
            r.setdefault("arxivId", "")
            r["source"] = "Search From OpenAlex"
            normalized.append(r)
        doc_info["references"] = normalized
        logger.info(
            f"OpenAlex refs for {doc_info.get('arxivId')}: "
            f"{len(ref_ids)} cited -> {len(normalized)} with metadata"
        )
        return doc_info

    def _fetch_via_pubmed(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Fetching PubMed references for PMID {doc_info.get('PMID')}")
        refs = doc_info.get("references", [])
        valid_pmids = [r["pmid"] for r in refs if "pmid" in r]
        if valid_pmids:
            doc_info["references"] = fetch_pubmed_json(valid_pmids)
        return doc_info

    def _fetch_via_openalex(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Fetching OpenAlex references for {doc_info.get('paper_id', '?')}")
        references = search_doc_via_url_from_openalex(doc_info["referenceWorksOpenAlex"])
        doc_info["references"] = references
        return doc_info
