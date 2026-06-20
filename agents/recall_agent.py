# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
RecallAgent (Agent B) — LLM "memory" recall + existence verification.

A parallel candidate route that runs beside the main multi-source retrieval at
tree depth 1:

  1. Ask the LLM, from its own knowledge, for the N papers that best match the
     user query (title + arXiv id), honouring the time constraint.
  2. Verify each recommendation really EXISTS: resolve its arXiv id against the
     local recovered.db first, then export.arxiv.org; for recommendations with
     no/wrong id, fall back to a Google Serper title search to find the real id
     and resolve that. Resolution yields real title + abstract = the evidence.
  3. A second LLM pass checks each recommendation against its retrieved evidence
     and keeps only the ones confirmed to be real and on-topic (hallucinated ids
     / titles are dropped).
  4. Return the verified, fully-resolved docs. The orchestrator injects them into
     the Phase-1 scoring pool (B_raw), where they are LLM-scored like any other
     retrieved doc (ids already in B_raw are deduped away by the orchestrator).

Public interface:
  recall(query, end_date) -> List[Dict]   # verified full-metadata docs
"""

import json
import traceback
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any

from global_config import (
    LLM_TRY_COUNT,
    API_TRY_COUNT,
    AGENT_B_RECALL_NUM,
    AGENT_B_MODEL,
    AGENT_B_VERIFY_MODEL,
)
from instruction import (
    template_agentb_recall,
    template_agentb_verify,
)
from local_request_v2 import get_from_llm
from utils import fetch_string
from api_web import (
    normalize_arxiv_id,
    google_search_arxiv_id,
    parallel_search_search_paper_from_arxiv,
)
from log import logger


class RecallAgent:
    """Agent B: LLM-recalled candidates, existence-verified before they enter B_raw."""

    def __init__(
        self,
        recall_num: int = AGENT_B_RECALL_NUM,
        model: str = AGENT_B_MODEL,
        verify_model: str = AGENT_B_VERIFY_MODEL,
    ):
        self.recall_num = recall_num
        self.model = model          # round 1 recall: thinking model (deepseek-v4-pro)
        self.verify_model = verify_model  # round 2 verify: fast flash model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def recall(self, query: str, end_date: str = "") -> List[Dict]:
        try:
            candidates = self._llm_recall(query, end_date)
        except Exception:
            logger.error(f"AgentB recall LLM failed: {traceback.format_exc()}")
            return []
        if not candidates:
            logger.info("AgentB: LLM returned no candidates")
            return []
        logger.info(f"AgentB: LLM recalled {len(candidates)} candidate papers")

        resolved = self._resolve_evidence(candidates, end_date)
        keep = self._llm_verify(query, candidates)

        out: List[Dict] = []
        seen = set()
        for i, c in enumerate(candidates):
            if not keep.get(i, False):
                continue
            doc = c.get("_evidence_doc")
            if not doc:
                continue
            pid = doc.get("paper_id") or doc.get("arxivId")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            doc["paper_id"] = pid
            doc.setdefault("arxivId", pid)
            doc["source"] = "Search From AgentB"
            out.append(doc)
        logger.info(
            f"AgentB: {len(out)} verified real papers (of {len(candidates)} recalled, "
            f"{len(resolved)} resolved)"
        )
        return out

    # ------------------------------------------------------------------
    # Step 1: LLM recall
    # ------------------------------------------------------------------

    def _time_constraint_text(self, end_date: str) -> str:
        if not end_date:
            return "No time constraint."
        try:
            d = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")
            return f"Only recommend papers published on or before {d}."
        except Exception:
            return "No time constraint."

    def _llm_recall(self, query: str, end_date: str) -> List[Dict]:
        prompt = template_agentb_recall.format(
            userQuery=query,
            timeConstraint=self._time_constraint_text(end_date),
            recall_num=self.recall_num,
        )
        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=self.model)
                data = json.loads(fetch_string(response).strip())
                papers = data.get("papers", data) if isinstance(data, dict) else data
                out = []
                for p in papers or []:
                    if not isinstance(p, dict):
                        continue
                    title = (p.get("title") or "").strip()
                    if not title:
                        continue
                    out.append({
                        "title": title,
                        "arxiv_id": (p.get("arxiv_id") or "").strip(),
                        "_norm_id": normalize_arxiv_id(p.get("arxiv_id") or ""),
                        "_serper_ids": [],
                        "_evidence_doc": None,
                    })
                if out:
                    return out[: self.recall_num]
            except Exception:
                logger.warning(f"AgentB recall attempt {attempt+1} failed: {traceback.format_exc()}")
        return []

    # ------------------------------------------------------------------
    # Step 2: existence verification (resolve real metadata = evidence)
    # ------------------------------------------------------------------

    def _resolve_evidence(self, candidates: List[Dict], end_date: str) -> Dict[str, Dict]:
        """Resolve each candidate to a real doc; annotate _evidence_doc in place.

        Returns the {paper_id: doc} map of everything that resolved (for logging).
        arXiv resolution checks recovered.db first, then export.arxiv.org.
        """
        resolved: Dict[str, Dict] = {}

        # Pass 1: resolve by the claimed arXiv id (one batched, throttle-safe call).
        claimed_ids = list({c["_norm_id"] for c in candidates if c["_norm_id"]})
        if claimed_ids:
            try:
                resolved.update(
                    parallel_search_search_paper_from_arxiv(claimed_ids, max_workers=1)
                )
            except Exception:
                logger.error(f"AgentB id-resolve failed: {traceback.format_exc()}")

        # Pass 2: for candidates still unresolved, Serper title search -> real id.
        pending = [
            c for c in candidates
            if not (c["_norm_id"] and c["_norm_id"] in resolved) and c["title"]
        ]
        if pending:
            def _serper(c):
                try:
                    return c, google_search_arxiv_id(
                        c["title"], try_num=API_TRY_COUNT, num=5, end_date=end_date
                    )
                except Exception:
                    logger.error(f"AgentB serper failed: {traceback.format_exc()}")
                    return c, []

            serper_ids = set()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
                for c, sids in ex.map(_serper, pending):
                    c["_serper_ids"] = sids or []
                    serper_ids.update(c["_serper_ids"])
            new_ids = [i for i in serper_ids if i not in resolved]
            if new_ids:
                try:
                    resolved.update(
                        parallel_search_search_paper_from_arxiv(new_ids, max_workers=1)
                    )
                except Exception:
                    logger.error(f"AgentB serper-id resolve failed: {traceback.format_exc()}")

        # Attach the best evidence doc to each candidate.
        for c in candidates:
            ev = None
            if c["_norm_id"] and c["_norm_id"] in resolved:
                ev = resolved[c["_norm_id"]]
            else:
                for sid in c.get("_serper_ids", []):
                    if sid in resolved:
                        ev = resolved[sid]
                        break
            c["_evidence_doc"] = ev
        return resolved

    # ------------------------------------------------------------------
    # Step 3: LLM second existence check (claim vs evidence)
    # ------------------------------------------------------------------

    def _llm_verify(self, query: str, candidates: List[Dict]) -> Dict[int, bool]:
        blocks = []
        for i, c in enumerate(candidates):
            ev = c.get("_evidence_doc")
            if ev:
                ev_title = ev.get("title", "")
                ev_abs = (ev.get("abstract", "") or "")[:600]
            else:
                ev_title, ev_abs = "(none found)", "(none found)"
            blocks.append(
                f"[CAND {i}]\n"
                f"  Recommended title: {c.get('title','')}\n"
                f"  Recommended arXiv id: {c.get('arxiv_id','') or '(none)'}\n"
                f"  Resolved title: {ev_title}\n"
                f"  Resolved abstract: {ev_abs}"
            )
        prompt = template_agentb_verify.format(
            userQuery=query, candidates="\n\n".join(blocks)
        )
        for attempt in range(LLM_TRY_COUNT):
            try:
                response = get_from_llm(prompt, model_name=self.verify_model)
                data = json.loads(fetch_string(response).strip())
                items = data.get("results", data) if isinstance(data, dict) else data
                out: Dict[int, bool] = {}
                for pos, item in enumerate(items or []):
                    if not isinstance(item, dict):
                        continue
                    try:
                        idx = int(item.get("idx", pos))
                    except (TypeError, ValueError):
                        idx = pos
                    out[idx] = bool(item.get("exists", False))
                if out:
                    return out
            except Exception:
                logger.warning(f"AgentB verify attempt {attempt+1} failed: {traceback.format_exc()}")
        # Fallback: if the verifier failed entirely, trust resolution alone —
        # a candidate that resolved to real metadata clearly exists.
        logger.warning("AgentB verify failed; keeping all candidates that resolved")
        return {i: (c.get("_evidence_doc") is not None) for i, c in enumerate(candidates)}
