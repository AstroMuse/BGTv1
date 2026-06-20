# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
dump_citation_urls.py — dump FRESH Semantic Scholar `citations` pre-signed URLs
into an aria2c input file, for exactly the shards not yet downloaded.

Why: large shards (~700 MB+) get repeatedly cut off when streamed single-threaded
through a shared proxy node. aria2c (-x16 -s16) splits each file into 16 segments
with per-segment retry/resume — far more robust on a flaky proxy than requests.

Workflow (run from repo root, SPAR env):
  1. STOP the running build first (Ctrl-C) — don't let it parse half-written files.
  2. python3 dump_citation_urls.py          # writes dl/citations_aria2.txt
  3. run the printed aria2c command          # pre-signed URLs expire in hours —
                                             # run it promptly; if some 403, just
                                             # re-run this script + aria2c (resumes)
  4. python3 build_citation_store.py --stream-download   # all present -> parse only

The build's download is the SKIP path: any cit_NNN.gz already on disk is not
re-fetched, so aria2c and the build compose cleanly.
"""
import os

from build_citation_store import list_dataset_files
from global_config import PROXIES

DL = "./dl/citations"
INP = "./dl/citations_aria2.txt"


def main():
    os.makedirs(DL, exist_ok=True)
    files = list_dataset_files("citations", "latest")  # hardened retry handles 429

    lines, todo, partial = [], 0, 0
    for i, f in enumerate(files, 1):
        url = f if isinstance(f, str) else f.get("url") or f
        name = f"cit_{i:03d}.gz"
        gz = os.path.join(DL, name)
        ctrl = gz + ".aria2"
        # Skip ONLY fully-downloaded shards (.gz present AND no .aria2 control
        # file). A shard that is missing, OR partial (has a .aria2 sibling), is
        # re-listed with a FRESH pre-signed URL so aria2c -c resumes it — the
        # old URLs expire (temporary STS token), and the dump used to skip any
        # existing file, which orphaned partials.
        if os.path.exists(gz) and not os.path.exists(ctrl):
            continue
        if os.path.exists(ctrl):
            partial += 1
        # aria2c input format: URL line, then options indented on following lines.
        lines.append(url)
        lines.append(f"  out={name}")
        todo += 1

    with open(INP, "w") as fw:
        fw.write("\n".join(lines) + ("\n" if lines else ""))

    proxy = PROXIES["https"]
    print("-" * 70)
    print(f"wrote {INP}")
    print(f"  shards to fetch : {todo}  ({partial} partial-resume, "
          f"{todo-partial} new; {len(files)-todo} already complete)")
    print("-" * 70)
    if todo == 0:
        print("Nothing to fetch — all shards present. Just run the build to parse.")
        return
    print("# 1) install aria2c if missing:")
    print("#      sudo apt install -y aria2")
    print("# 2) leftover requests .part files are harmless; optional cleanup:")
    print(f"#      rm -f {DL}/*.part")
    print("# 3) download (16-segment, resumable, infinite retry, via proxy):")
    print(
        f"aria2c -i {INP} -d {DL} -x16 -s16 -j2 -c "
        f"--auto-file-renaming=false --max-tries=0 --retry-wait=5 --timeout=60 "
        f"--all-proxy={proxy}"
    )
    print("# 4) then resume the build (skips downloads, streams parse -> LMDB):")
    print("#      python3 build_citation_store.py --stream-download")


if __name__ == "__main__":
    main()
