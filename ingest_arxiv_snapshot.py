#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================
# [Descriptions] : Stream-ingest the arXiv full metadata snapshot
#   (Kaggle: Cornell-University/arxiv -> arxiv-metadata-oai-snapshot.json)
#   into the local SQLite cache, mapped to SPAR's doc schema.
#
# The snapshot is one JSON object per line (~2.6M papers, ~5GB uncompressed).
# This reads it line-by-line (constant memory) and batch-inserts.
#
# Usage:
#   # full offline coverage of all of arXiv:
#   python3 ingest_arxiv_snapshot.py arxiv-metadata-oai-snapshot.json --all
#
#   # only fill the IDs that previously failed (failed_arxiv_ids.log):
#   python3 ingest_arxiv_snapshot.py arxiv-metadata-oai-snapshot.json --only-failed
#
#   # .json / .json.gz / .zip all detected automatically; --limit N for a test run.
#
# Download the snapshot first (needs a Kaggle account + ~/.kaggle/kaggle.json):
#   pip install kaggle
#   kaggle datasets download -d Cornell-University/arxiv   # -> arxiv.zip (~1.5GB)
#   python3 ingest_arxiv_snapshot.py arxiv.zip --all       # feed the .zip directly
# ==================================================================

import argparse
import gzip
import io
import json
import os
import sys
import zipfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tqdm import tqdm

from local_db_v2 import db_path as DEFAULT_DB_PATH, ArxivDatabase
from api_web import FAILED_ARXIV_IDS_LOG
from log import logger


# ---------------------------------------------------------------------------
# Field mapping: Kaggle snapshot record -> SPAR doc dict
# ---------------------------------------------------------------------------

def _parse_pub_year(rec: dict) -> str:
    """v1 submission date -> YYYYMMDD (matches api_web's publicationYear semantics)."""
    versions = rec.get("versions") or []
    if versions:
        created = versions[0].get("created", "")
        try:
            return datetime.strptime(
                created, "%a, %d %b %Y %H:%M:%S %Z"
            ).strftime("%Y%m%d")
        except Exception:
            pass
    # fallback: last-update date "YYYY-MM-DD" -> "YYYYMMDD"
    return (rec.get("update_date") or "").replace("-", "")


def kaggle_to_doc(rec: dict) -> dict:
    aid = rec["id"]
    authors = [
        {"name": f"{first} {last}".strip()}
        for last, first, *_ in (rec.get("authors_parsed") or [])
    ]
    if not authors and rec.get("authors"):
        authors = [{"name": rec["authors"]}]
    return {
        "paper_id": aid,
        "arxivId": aid,
        "arxivUrl": f"http://arxiv.org/abs/{aid}",
        "title": (rec.get("title") or "").replace("\n", " ").strip(),
        "abstract": (rec.get("abstract") or "").replace("\n", " ").strip(),
        "authors": authors,
        "publicationYear": _parse_pub_year(rec),
        "fieldsOfStudy": ";".join((rec.get("categories") or "").split()),
        "doi": rec.get("doi") or "",
        "journalRef": rec.get("journal-ref") or "",
        "source": "Search From Arxiv",
    }


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def open_snapshot(path: str):
    """Open .json, .json.gz, or the Kaggle .zip transparently as a text line iterator.

    For .zip we stream the largest .json member directly (no full extraction),
    keeping a reference to the ZipFile so it isn't garbage-collected mid-read.
    """
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        json_members = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_members:
            raise ValueError(f"No .json member inside {path}: {zf.namelist()}")
        name = max(json_members, key=lambda n: zf.getinfo(n).file_size)
        wrapper = io.TextIOWrapper(zf.open(name, "r"), encoding="utf-8")
        wrapper._zipfile_ref = zf  # keep ZipFile alive for the wrapper's lifetime
        return wrapper
    return open(path, "r", encoding="utf-8")


def load_wanted_ids(log_path: str) -> set:
    if not os.path.exists(log_path):
        logger.warning(f"--only-failed: log not found: {log_path}")
        return set()
    with open(log_path, "r") as f:
        return {ln.strip() for ln in f if ln.strip()}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def ingest(snapshot: str, db_path: str, only_failed: bool,
           batch_size: int, limit: int, overwrite: bool):
    wanted = load_wanted_ids(FAILED_ARXIV_IDS_LOG) if only_failed else None
    if only_failed:
        print(f"only-failed mode: {len(wanted)} target IDs")
        if not wanted:
            print("Nothing to do (empty target set).")
            return

    # OR IGNORE (default): keep existing rows untouched — they may carry
    # OpenAlex/S2 enrichment (citationCount/references) the snapshot lacks.
    # OR REPLACE (--overwrite): clobber existing rows with snapshot data.
    verb = "INSERT OR REPLACE" if overwrite else "INSERT OR IGNORE"
    print(f"write mode: {verb}")

    inserted = bad = scanned = 0
    seen_wanted = set()
    batch = []

    with ArxivDatabase(db_path) as db, open_snapshot(snapshot) as fr:
        def flush():
            nonlocal batch
            if not batch:
                return
            db.cursor.executemany(
                f"{verb} INTO arxiv_docs (arxiv_id, data) VALUES (?, ?)",
                batch,
            )
            db.conn.commit()
            batch = []

        for line in tqdm(fr, desc="Ingesting", unit=" rec"):
            scanned += 1
            line = line.strip().rstrip(",")  # tolerate trailing commas
            if not line or line in ("[", "]"):
                continue
            try:
                rec = json.loads(line)
                aid = rec["id"]
            except Exception:
                bad += 1
                continue

            if wanted is not None:
                if aid not in wanted:
                    continue
                seen_wanted.add(aid)

            try:
                doc = kaggle_to_doc(rec)
                batch.append((aid, json.dumps(doc)))
                inserted += 1
            except Exception:
                bad += 1
                continue

            if len(batch) >= batch_size:
                flush()

            if limit and inserted >= limit:
                break

            # early exit once every wanted id is found
            if wanted is not None and len(seen_wanted) == len(wanted):
                break

        flush()

    print("\n=== Done ===")
    print(f"  scanned lines : {scanned}")
    print(f"  inserted/updated: {inserted}")
    print(f"  bad/skipped lines: {bad}")
    if wanted is not None:
        missing = wanted - seen_wanted
        print(f"  matched targets : {len(seen_wanted)}/{len(wanted)}")
        if missing:
            print(f"  NOT in snapshot ({len(missing)}): "
                  f"{sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
            print("  (these are likely non-arXiv IDs, e.g. S2/OpenAlex-only papers)")


def main():
    ap = argparse.ArgumentParser(
        description="Ingest the arXiv full metadata snapshot into the local SQLite cache"
    )
    ap.add_argument("snapshot", help="Path to arxiv-metadata-oai-snapshot.json[.gz]")
    ap.add_argument("--db", default=DEFAULT_DB_PATH,
                    help=f"SQLite path (default: {DEFAULT_DB_PATH})")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true",
                      help="Ingest every record (full offline coverage)")
    mode.add_argument("--only-failed", action="store_true",
                      help="Ingest only IDs listed in failed_arxiv_ids.log (default)")
    ap.add_argument("--batch-size", type=int, default=5000,
                    help="Rows per SQLite commit (default: 5000)")
    ap.add_argument("--limit", type=int, default=0,
                    help="Stop after N inserts (0 = no limit; for testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing rows (INSERT OR REPLACE). Default keeps "
                         "existing rows (INSERT OR IGNORE) to preserve enrichment.")
    args = ap.parse_args()

    if not os.path.exists(args.snapshot):
        sys.exit(f"Snapshot not found: {args.snapshot}")

    only_failed = not args.all  # default to only-failed unless --all is given
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    ingest(args.snapshot, args.db, only_failed, args.batch_size, args.limit,
           args.overwrite)


if __name__ == "__main__":
    main()
