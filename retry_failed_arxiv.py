#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry downloading papers that previously failed due to 429 rate limiting.

Two modes:
  arxiv  — retry arXiv batch fetches (failed_arxiv_ids.log)
           fetches full metadata and writes to SQLite
  s2     — retry Semantic Scholar reference lookups (failed_s2_ids.log)
           fetches reference lists and writes all referenced papers to SQLite

Usage:
    python3 retry_failed_arxiv.py arxiv              # one pass, retry arXiv failures
    python3 retry_failed_arxiv.py arxiv --loop       # loop until all IDs stored or no progress
    python3 retry_failed_arxiv.py s2 --loop          # same for S2
    python3 retry_failed_arxiv.py arxiv --dry-run
    python3 retry_failed_arxiv.py arxiv --loop   # recommended; defaults are ToS-safe
                                                 # (batch-size 50, delay 3s, single connection)

--loop behaviour:
    Keeps running passes until the log is empty (all IDs stored) or two
    consecutive passes make zero progress (those IDs are likely 404/retracted
    and will never be fetchable). Permanently-unfetchable IDs are moved to
    <log>.permanent_failures for inspection and removed from the retry log.
"""

import sys
import os
import re
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api_web import (
    search_paper_from_arxiv_by_arxiv_id_bsz,
    get_doc_info_from_semantic_scholar_by_arxivid,
    FAILED_ARXIV_IDS_LOG,
    FAILED_S2_IDS_LOG,
)
from local_db_v2 import db_path, ArxivDatabase
from log import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Modern IDs: YYMM.NNNN[N] (4-5 digit seq), optional version.
# Old-style IDs: archive[.SS]/NNNNNNN, optional version (e.g. hep-ph/9606011).
# A bare 7-digit number like "9606011" is an old-style ID that lost its archive
# prefix — export.arxiv.org returns HTTP 400 for the whole batch if one is present.
_MODERN_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_OLD_ID = re.compile(r"^[a-z][a-z-]*(\.[A-Z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE)


def is_valid_arxiv_id(aid: str) -> bool:
    """True if aid is a well-formed arXiv identifier accepted by the API."""
    return bool(_MODERN_ID.match(aid) or _OLD_ID.match(aid))


def load_failed_ids(log_path: str) -> list:
    """Load unique IDs from log, preserving order."""
    if not os.path.exists(log_path):
        return []
    with open(log_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    seen = set()
    unique = []
    for _id in ids:
        if _id not in seen:
            seen.add(_id)
            unique.append(_id)
    return unique


def save_ids(log_path: str, ids: list):
    with open(log_path, "w") as f:
        for _id in ids:
            f.write(_id + "\n")


def ids_not_in_cache(ids: list) -> list:
    if not os.path.exists(db_path):
        return ids
    with ArxivDatabase(db_path) as db:
        return [_id for _id in ids if db.get(_id) is None]


def append_permanent_failures(log_path: str, ids: list):
    perm_path = log_path + ".permanent_failures"
    with open(perm_path, "a") as f:
        for _id in ids:
            f.write(_id + "\n")
    print(f"  {len(ids)} permanently-unfetchable IDs written to {perm_path}")


# ---------------------------------------------------------------------------
# One pass: arXiv mode
# ---------------------------------------------------------------------------

def _arxiv_pass(to_fetch: list, batch_size: int, delay: float, max_retries: int) -> tuple:
    """
    Run one pass over to_fetch.
    Returns (success_ids, still_failed_ids).
    """
    success_ids = []
    still_failed = []
    total_batches = (len(to_fetch) + batch_size - 1) // batch_size

    for i in range(0, len(to_fetch), batch_size):
        batch = to_fetch[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  [{batch_num}/{total_batches}] Fetching: {batch}")

        result = search_paper_from_arxiv_by_arxiv_id_bsz(batch, max_retries=max_retries)

        if result:
            with ArxivDatabase(db_path) as db:
                for _id, info in result.items():
                    db.update_or_insert(_id, info)
            fetched = list(result.keys())
            success_ids.extend(fetched)
            missing = [_id for _id in batch if _id not in result]
            still_failed.extend(missing)
            msg = f"    OK: {fetched}"
            if missing:
                msg += f"  MISSING: {missing}"
            print(msg)
        else:
            still_failed.extend(batch)
            print(f"    FAILED (all {len(batch)} IDs, likely 429)")

        if i + batch_size < len(to_fetch):
            print(f"    Sleeping {delay}s ...")
            time.sleep(delay)

    return success_ids, still_failed


def run_arxiv(log_path: str, batch_size: int, delay: float, max_retries: int,
              dry_run: bool, loop: bool, inter_pass_delay: float):

    if dry_run:
        ids = load_failed_ids(log_path)
        malformed = [i for i in ids if not is_valid_arxiv_id(i)]
        valid = [i for i in ids if is_valid_arxiv_id(i)]
        to_fetch = ids_not_in_cache(valid)
        print(f"[dry-run] {len(ids)} in log | {len(malformed)} malformed (would "
              f"quarantine) | {len(to_fetch)} valid & not yet cached")
        if malformed:
            print(f"  malformed examples: {malformed[:10]}")
        for i in range(0, len(to_fetch), batch_size):
            print(f"  batch {i//batch_size+1}: {to_fetch[i:i+batch_size]}")
        return

    no_progress_streak = 0
    pass_num = 0

    while True:
        all_ids = load_failed_ids(log_path)
        if not all_ids:
            print("Log is empty — all IDs have been stored. Done.")
            return

        # Quarantine malformed IDs first — they cause HTTP 400 for the whole
        # batch and can never be fetched (e.g. old-style IDs missing the prefix).
        malformed = [i for i in all_ids if not is_valid_arxiv_id(i)]
        if malformed:
            print(f"  Quarantining {len(malformed)} malformed IDs -> "
                  f"{os.path.basename(log_path)}.permanent_failures: "
                  f"{malformed[:5]}{' ...' if len(malformed) > 5 else ''}")
            append_permanent_failures(log_path, malformed)
            all_ids = [i for i in all_ids if is_valid_arxiv_id(i)]
            save_ids(log_path, all_ids)
            if not all_ids:
                print("Only malformed IDs remained. Log cleared. Done.")
                return

        to_fetch = ids_not_in_cache(all_ids)
        if not to_fetch:
            print("All IDs already in cache. Clearing log.")
            save_ids(log_path, [])
            return

        pass_num += 1
        print(f"\n{'='*55}")
        print(f"Pass {pass_num} | {len(to_fetch)} IDs to fetch")
        print(f"{'='*55}")

        success, still_failed = _arxiv_pass(to_fetch, batch_size, delay, max_retries)

        print(f"\n  Pass {pass_num} result: fetched={len(success)}  still_failed={len(still_failed)}")
        save_ids(log_path, still_failed)

        if not still_failed:
            print("All IDs fetched. Log cleared. Done.")
            return

        if not loop:
            print(f"{len(still_failed)} IDs remain. Run again (or use --loop) to retry.")
            return

        # Detect no-progress: same set of IDs failed as before
        if set(still_failed) == set(to_fetch):
            no_progress_streak += 1
            print(f"  No progress this pass (streak={no_progress_streak}/2)")
            if no_progress_streak >= 2:
                print(f"\nTwo consecutive passes with zero progress.")
                print(f"These {len(still_failed)} IDs are likely 404/retracted/permanently unavailable:")
                for _id in still_failed:
                    print(f"  {_id}")
                append_permanent_failures(log_path, still_failed)
                save_ids(log_path, [])
                print("Retry log cleared. Done.")
                return
        else:
            no_progress_streak = 0

        print(f"Waiting {inter_pass_delay}s before next pass ...")
        time.sleep(inter_pass_delay)


# ---------------------------------------------------------------------------
# One pass: S2 mode
# ---------------------------------------------------------------------------

def _s2_pass(to_process: list, delay: float, max_retries: int) -> tuple:
    """
    Run one S2 pass. Returns (success_ids, still_failed_ids).
    success means S2 responded (even if no refs); still_failed means S2 still 429.
    """
    success_ids = []
    still_failed = []

    for i, arxiv_id in enumerate(to_process):
        print(f"  [{i+1}/{len(to_process)}] S2 lookup: {arxiv_id}")
        doc_info = get_doc_info_from_semantic_scholar_by_arxivid(arxiv_id, try_num=max_retries)
        if doc_info is not None:
            refs = [r for r in doc_info.get("references", []) if r.get("paper_id") and r.get("title")]
            if refs:
                with ArxivDatabase(db_path) as db:
                    for ref in refs:
                        db.update_or_insert(ref["paper_id"], ref)
                print(f"    OK — {len(refs)} references written to SQLite")
            else:
                print(f"    OK — S2 responded, no full-metadata references")
            success_ids.append(arxiv_id)
        else:
            print(f"    FAILED — S2 still unavailable (429 or error)")
            still_failed.append(arxiv_id)

        if i + 1 < len(to_process):
            time.sleep(delay)

    return success_ids, still_failed


def run_s2(log_path: str, delay: float, max_retries: int,
           dry_run: bool, loop: bool, inter_pass_delay: float):

    if dry_run:
        ids = load_failed_ids(log_path)
        print(f"[dry-run] {len(ids)} parent arXiv IDs in S2 log:")
        for _id in ids:
            print(f"  {_id}")
        return

    no_progress_streak = 0
    pass_num = 0

    while True:
        all_ids = load_failed_ids(log_path)
        if not all_ids:
            print("S2 log is empty. Done.")
            return

        pass_num += 1
        print(f"\n{'='*55}")
        print(f"Pass {pass_num} | {len(all_ids)} parent IDs to resolve via S2")
        print(f"{'='*55}")

        success, still_failed = _s2_pass(all_ids, delay, max_retries)

        print(f"\n  Pass {pass_num} result: resolved={len(success)}  still_failed={len(still_failed)}")
        save_ids(log_path, still_failed)

        if not still_failed:
            print("All S2 lookups resolved. Log cleared. Done.")
            return

        if not loop:
            print(f"{len(still_failed)} IDs remain. Run again (or use --loop) to retry.")
            return

        if set(still_failed) == set(all_ids):
            no_progress_streak += 1
            print(f"  No progress this pass (streak={no_progress_streak}/2)")
            if no_progress_streak >= 2:
                print(f"\nTwo consecutive passes with zero progress.")
                print(f"These {len(still_failed)} parent IDs are permanently unreachable via S2:")
                for _id in still_failed:
                    print(f"  {_id}")
                append_permanent_failures(log_path, still_failed)
                save_ids(log_path, [])
                print("S2 retry log cleared. Done.")
                return
        else:
            no_progress_streak = 0

        print(f"Waiting {inter_pass_delay}s before next pass ...")
        time.sleep(inter_pass_delay)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retry fetching papers that failed due to 429 rate limiting"
    )
    parser.add_argument(
        "mode", nargs="?", default="arxiv", choices=["arxiv", "s2"],
        help="arxiv: retry arXiv batch fetches; s2: retry Semantic Scholar reference lookups"
    )
    parser.add_argument("--log", default=None, help="Override default log file path")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="IDs per arXiv API request (default: 50; one id_list "
                             "request returns up to 100, so larger = fewer requests)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Seconds between batches within a pass (default: 3, the "
                             "arXiv ToS minimum). The arxiv client also throttles "
                             "page requests internally at 3s.")
    parser.add_argument("--inter-pass-delay", type=float, default=300.0,
                        help="Seconds to wait between loop passes (default: 300)")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Retries per individual request (default: 5)")
    parser.add_argument("--loop", action="store_true",
                        help="Keep retrying passes until log is empty or no progress")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without writing anything")
    args = parser.parse_args()

    if args.mode == "s2":
        log_path = args.log or FAILED_S2_IDS_LOG
        run_s2(log_path, args.delay, args.max_retries, args.dry_run,
               args.loop, args.inter_pass_delay)
    else:
        log_path = args.log or FAILED_ARXIV_IDS_LOG
        run_arxiv(log_path, args.batch_size, args.delay, args.max_retries,
                  args.dry_run, args.loop, args.inter_pass_delay)


if __name__ == "__main__":
    main()
