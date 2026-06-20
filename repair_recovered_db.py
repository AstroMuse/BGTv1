# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
repair_recovered_db.py — de-nest bloated reference blobs in recovered.db.

A bug let stored docs accumulate references-of-references (each cached ref still
carried its own `references`), bloating single blobs toward SQLite's ~1 GB string
limit and blowing up memory. ArxivDatabase.update_or_insert now strips that on
write; this script repairs rows already bloated on disk.

For each row larger than --threshold, it strips the nested `references`/`citations`
off the top-level reference entries (which drops the entire nested subtree),
keeping the doc's own title/abstract + one level of refs (title/abstract) that
Phase-2 needs. Run with the benchmark / S1 stopped.

  python3 repair_recovered_db.py --dry-run        # report only
  python3 repair_recovered_db.py                  # repair
  sqlite3 database/recovered.db 'VACUUM;'         # reclaim file space afterwards
"""
import argparse
import json
import sqlite3

from local_db_v2 import db_path, ArxivDatabase
from log import logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=300_000,
                    help="only inspect blobs larger than this many bytes (default 300KB)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ro = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ids = [r[0] for r in ro.execute(
        "SELECT arxiv_id FROM arxiv_docs WHERE length(data) > ?", (args.threshold,))]
    ro.close()
    print(f"candidate rows > {args.threshold} bytes: {len(ids)}")
    if not ids:
        return

    db = ArxivDatabase(db_path)
    db.conn.execute("PRAGMA busy_timeout=60000")
    fixed = skipped = 0
    saved = 0
    try:
        for i, aid in enumerate(ids, 1):
            try:
                rec = db.get(aid)            # may briefly load a large blob into RAM
            except Exception as e:
                logger.error(f"get {aid} failed: {e}")
                continue
            if not isinstance(rec, dict):
                continue
            before = len(json.dumps(rec))
            slim = ArxivDatabase._slim_for_storage(rec)
            after = len(json.dumps(slim))
            if after < before:
                print(f"[{i}/{len(ids)}] {aid}: {before/1e6:8.2f}MB -> {after/1e6:6.3f}MB")
                saved += before - after
                fixed += 1
                if not args.dry_run:
                    db.update_or_insert(aid, slim)
            else:
                skipped += 1
            del rec, slim
    finally:
        db.close()

    print("-" * 60)
    print(f"rows inspected={len(ids)}  de-nested={fixed}  unchanged={skipped}  "
          f"reclaimed~{saved/1e6:.1f}MB" + ("  (dry-run)" if args.dry_run else ""))
    if not args.dry_run and fixed:
        print("Now run:  sqlite3 database/recovered.db 'VACUUM;'  to shrink the file")


if __name__ == "__main__":
    main()
