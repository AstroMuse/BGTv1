#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""One-query CLI for the AstaBench bridge.

Runs the SPAR pipeline (reference + bge-prefilter + phase3 config) on a single
query and writes PaperFindingBench-formatted results to --out as JSON:

    {"results": [{"paper_id": "<s2 corpus_id>", "markdown_evidence": "..."}]}

Invoked by astabench_bridge/spar_solver.py with the SPAR conda python and
cwd = SPAR repo root. stdout/stderr are free to carry pipeline logs; only the
--out file is parsed.
"""
import argparse
import json
import os
import sys

# SPAR run configuration must be in the environment BEFORE global_config is
# imported. setdefault so values exported by the caller win.
os.environ.setdefault("SPAR_DO_REFERENCE", "1")
os.environ.setdefault("SPAR_BGE_PREFILTER", "1")
os.environ.setdefault("SPAR_DO_PHASE3", "1")
os.environ.setdefault(
    "BGE_LORA_PATH", "./bge/output/lora-specter-hardneg/best_model"
)

_SPAR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_SPAR_ROOT)  # relative paths (./database, ./bge) must resolve
sys.path.insert(0, _SPAR_ROOT)

EVIDENCE_ABSTRACT_CHARS = 1200
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"


def to_s2_external_id(doc):
    """Best-effort external id usable by the S2 batch endpoint."""
    aid = doc.get("arxivId") or ""
    if not aid:
        from evaluate import norm_arxiv  # SPAR repo helper

        aid = norm_arxiv(doc.get("paper_id", "")) or ""
    if aid:
        return f"ARXIV:{aid}"
    doi = doc.get("doi") or (doc.get("externalIds") or {}).get("DOI")
    if doi:
        return f"DOI:{doi}"
    return None


def map_to_corpus_ids(docs):
    """Resolve docs -> S2 corpus_id via one batch call. Returns {idx: corpus_id}."""
    import requests
    from global_config import PROXIES, SEMANTIC_SCHOLAR_API_KEY

    ext_ids, idx_of = [], []
    for i, d in enumerate(docs):
        eid = to_s2_external_id(d)
        if eid:
            ext_ids.append(eid)
            idx_of.append(i)
    if not ext_ids:
        return {}
    headers = {"Content-Type": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    resp = requests.post(
        S2_BATCH_URL,
        params={"fields": "corpusId"},
        json={"ids": ext_ids},
        headers=headers,
        proxies=PROXIES,
        timeout=60,
    )
    resp.raise_for_status()
    out = {}
    for i, item in zip(idx_of, resp.json()):
        if isinstance(item, dict) and item.get("corpusId") is not None:
            out[i] = str(item["corpusId"])
    return out


def evidence(doc):
    title = (doc.get("title") or "").strip()
    year = doc.get("publicationYear") or doc.get("year") or ""
    abstract = " ".join((doc.get("abstract") or "").split())
    return f"**{title}** ({year}). {abstract[:EVIDENCE_ABSTRACT_CHARS]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--end-date", default="", help="insertion cutoff, e.g. 2025-06-01")
    ap.add_argument("--topn", type=int, default=50,
                    help="max results returned, ranked by sim_score (cap; <=250)")
    ap.add_argument("--min-sim", type=float, default=None,
                    help="if set, return only docs with sim_score > this "
                         "(A high-rel pool, e.g. 0.55), still in sim order, "
                         "capped at --topn. Default None = full scored pool.")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    from pipeline_spar import AcademicSearchTree
    from local_request_v2 import reset_token_usage, get_token_usage
    from global_config import LLM_MODEL_NAME

    reset_token_usage()
    agent = AcademicSearchTree(max_depth=2, max_docs=10, similarity_threshold=0.5)
    agent.search(args.query, end_date=args.end_date)
    usage = get_token_usage()

    searched = agent.root.searched_docs or {}
    scored = [d for d in searched.values() if (d.get("sim_score") or -1) >= 0]
    scored.sort(key=lambda d: d.get("sim_score") or -1, reverse=True)
    # Selection: full scored pool by default; with --min-sim, the A high-rel
    # pool (sim_score > min_sim) keeping the LLM relevance order. Either way the
    # top-N cap applies (N<=250, the asta-bench per-query limit).
    if args.min_sim is not None:
        selected = [d for d in scored if (d.get("sim_score") or -1) > args.min_sim]
    else:
        selected = scored
    top = selected[: args.topn]

    corpus_of = map_to_corpus_ids(top)
    results, seen = [], set()
    for i, doc in enumerate(top):
        cid = corpus_of.get(i)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        results.append({"paper_id": cid, "markdown_evidence": evidence(doc)})

    # usage: the SPAR agent's real DeepSeek token spend for this query. The
    # solver injects it into inspect's ModelUsage so the leaderboard cost axis
    # reflects the agent (these calls bypass inspect's model API otherwise).
    out_obj = {
        "results": results,
        "usage": {
            "model": LLM_MODEL_NAME,
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "calls": usage["calls"],
        },
    }
    with open(args.out, "w", encoding="utf-8") as fw:
        json.dump(out_obj, fw, ensure_ascii=False)
    print(f"[spar_query_cli] wrote {len(results)} results "
          f"(scored pool {len(scored)}, top {len(top)}, mapped {len(corpus_of)}) "
          f"| DeepSeek tokens in={usage['input_tokens']} out={usage['output_tokens']} "
          f"calls={usage['calls']}")


if __name__ == "__main__":
    main()
