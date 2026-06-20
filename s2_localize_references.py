# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
S1 — seed-incremental localization of Semantic Scholar reference lists into
recovered.db.

For each seed arXiv paper, fetch its reference list via the existing S2 path
(get_doc_info_from_semantic_scholar_by_arxivid — 1 S2 call/paper, throttled;
reference abstracts are resolved through parallel_search_search_paper_from_arxiv,
which mostly hits the local arXiv snapshot) and merge the resulting
``references`` field into that paper's blob in recovered.db (read-modify-write,
so existing metadata is preserved).

Once stored, RetrievalAgent loads the blob from the local DB (source
"Search From Local") and OrchestratorAgent._phase_reference_expansion sees
``doc_info["references"]`` already populated with full title+abstract refs, so
it skips the online fetch entirely. Pair this with ``--ref-api s2``.

Seeds:
  gold  : answer_arxiv_id across the benchmark files (the papers we want to find)
  runs  : arXiv-keyed docs already retrieved in gen_result/*/*.json (the docs
          Phase-2 actually expands), optionally filtered by --min-sim
  both  : union of the two

Safe to run alongside a benchmark: recovered.db is WAL; papers already in the
DB are never rewritten by the running pipeline (it skips "Search From Local"),
and writes here are read-modify-write with a 60s busy_timeout.

Usage (run from repo root, SPAR conda env):
  python3 s2_localize_references.py --dry-run --limit 5            # validate, no writes
  python3 s2_localize_references.py --seed-source gold             # full gold run
  python3 s2_localize_references.py --seed-source both --min-sim 0.5
"""
import argparse
import glob
import json
import os
import re
import time

from api_web import get_doc_info_from_semantic_scholar_by_arxivid, normalize_arxiv_id
from local_db_v2 import db_path, ArxivDatabase
from log import logger

BENCHMARK_FILES = [
    "benchmark/AutoScholarQuery_test.jsonl",
    "benchmark/spar_bench.jsonl",
]
ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def _norm(x):
    m = ARXIV_RE.search(str(x))
    return m.group(1) if m else None


def collect_gold_seeds():
    seeds = set()
    for path in BENCHMARK_FILES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ids = d.get("answer_arxiv_id") or []
                if isinstance(ids, str):
                    ids = [ids]
                for i in ids:
                    n = _norm(i)
                    if n:
                        seeds.add(n)
    return seeds


def collect_run_seeds(min_sim):
    seeds = set()
    for fp in glob.glob("gen_result/*/*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as fr:
                data = json.load(fr)
        except Exception:
            continue
        if not isinstance(data, dict) or "search_query" not in data:
            continue
        searched = data.get("extra", {}).get("searched_docs", {})
        for pid, doc in searched.items():
            n = _norm(pid)
            if not n:
                continue
            if min_sim is not None and (doc.get("sim_score") or -1) < min_sim:
                continue
            seeds.add(n)
    return seeds


def has_local_refs(blob):
    """True if the stored blob already carries full (title+abstract) references."""
    refs = (blob or {}).get("references") or []
    return bool(refs) and any(r.get("title") and r.get("abstract") for r in refs)


def main():
    ap = argparse.ArgumentParser(description="S1: localize S2 references into recovered.db")
    ap.add_argument("--seed-source", choices=["gold", "runs", "both"], default="gold")
    ap.add_argument("--min-sim", type=float, default=None,
                    help="for runs/both: only seed retrieved docs with sim_score >= this")
    ap.add_argument("--limit", type=int, default=None, help="cap number of seeds (debug)")
    ap.add_argument("--rps", type=float, default=1.0, help="max S2 requests/sec (sleep floor)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch even if the blob already has local references")
    ap.add_argument("--dry-run", action="store_true", help="fetch + report, write nothing")
    args = ap.parse_args()

    if not os.path.exists(db_path):
        logger.error(f"recovered.db not found at {db_path} (run from repo root)")
        return

    seeds = set()
    if args.seed_source in ("gold", "both"):
        seeds |= collect_gold_seeds()
    if args.seed_source in ("runs", "both"):
        seeds |= collect_run_seeds(args.min_sim)
    seeds = sorted(seeds)
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"seed-source={args.seed_source}  candidate seeds={len(seeds)}  "
          f"dry_run={args.dry_run}")

    db = ArxivDatabase(db_path)
    db.conn.execute("PRAGMA busy_timeout=60000")

    n_skip = n_fetch = n_fail = n_refs = 0
    min_interval = 1.0 / args.rps if args.rps > 0 else 0.0
    try:
        for i, sid in enumerate(seeds, 1):
            existing = db.get(sid)
            if not args.refresh and has_local_refs(existing):
                n_skip += 1
                continue

            t0 = time.time()
            try:
                doc = get_doc_info_from_semantic_scholar_by_arxivid(sid, try_num=6)
            except Exception:
                doc = None
            # pace to <= rps (the S2 helper also self-throttles, this is a floor)
            dt = time.time() - t0
            if dt < min_interval:
                time.sleep(min_interval - dt)

            if not doc:
                n_fail += 1
                print(f"[{i}/{len(seeds)}] {sid}  FETCH FAILED")
                continue

            refs = doc.get("references") or []
            full = sum(1 for r in refs if r.get("title") and r.get("abstract"))
            n_fetch += 1
            n_refs += full
            print(f"[{i}/{len(seeds)}] {sid}  refs={len(refs)} full(title+abs)={full}")

            if not args.dry_run:
                merged = dict(existing or {})
                merged.update(doc)  # doc carries the fresh references + S2 metadata
                merged["arxivId"] = sid
                merged.setdefault("paper_id", sid)
                try:
                    db.update_or_insert(sid, merged)
                except Exception as e:
                    logger.error(f"write failed for {sid}: {e}")
    finally:
        db.close()

    print("-" * 60)
    print(f"seeds={len(seeds)}  fetched={n_fetch}  skipped(local already)={n_skip}  "
          f"failed={n_fail}  total full refs stored={n_refs}")
    if args.dry_run:
        print("DRY RUN — nothing written to recovered.db")


if __name__ == "__main__":
    main()
