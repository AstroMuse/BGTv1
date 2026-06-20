# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# Encode recovered.db arXiv abstracts with SPECTER2 and build a FAISS index,
# for hard-negative mining. Shard-based + resumable: each shard's embeddings/ids
# are saved to disk; re-running skips finished shards. After all shards, builds
# a flat inner-product index (vectors are L2-normalized -> IP == cosine).
#
#   python3 build_specter_index.py                 # full ~3M (hours, GPU)
#   python3 build_specter_index.py --limit 3000    # quick subset test
#   python3 build_specter_index.py --build-only     # just (re)build faiss from shards
# ==================================================================
import argparse
import json
import os
import sqlite3

import numpy as np
from tqdm import tqdm

from local_db_v2 import db_path as DEFAULT_DB
from specter_embed import embed, paper_text


def iter_shard(db, offset, limit):
    """Yield (paper_id, title, abstract) for a rowid-ordered page; stable for resume."""
    cur = db.execute(
        "SELECT arxiv_id, data FROM arxiv_docs ORDER BY rowid LIMIT ? OFFSET ?",
        (limit, offset),
    )
    for arxiv_id, data in cur:
        try:
            d = json.loads(data)
        except Exception:
            continue
        title, abstract = d.get("title"), d.get("abstract")
        if title and abstract:
            pid = d.get("paper_id") or d.get("arxivId") or arxiv_id
            yield pid, title, abstract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="bge/data/specter_index")
    ap.add_argument("--shard-size", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="cap total records (0=all)")
    ap.add_argument("--build-only", action="store_true", help="skip encoding, just build faiss")
    args = ap.parse_args()

    shards_dir = os.path.join(args.out, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    db = sqlite3.connect(args.db)
    total = db.execute("SELECT COUNT(*) FROM arxiv_docs").fetchone()[0]
    if args.limit:
        total = min(total, args.limit)
    n_shards = (total + args.shard_size - 1) // args.shard_size
    print(f"records: {total} | shard_size: {args.shard_size} | shards: {n_shards}")

    if not args.build_only:
        for s in range(n_shards):
            emb_f = os.path.join(shards_dir, f"shard_{s:05d}.npy")
            ids_f = os.path.join(shards_dir, f"shard_{s:05d}.ids.txt")
            if os.path.exists(emb_f) and os.path.exists(ids_f):
                continue  # resume: already done
            this_limit = min(args.shard_size, total - s * args.shard_size)
            rows = list(iter_shard(db, s * args.shard_size, this_limit))
            if not rows:
                continue
            ids = [r[0] for r in rows]
            texts = [paper_text(r[1], r[2]) for r in rows]
            print(f"[shard {s+1}/{n_shards}] encoding {len(texts)} docs ...")
            vecs = embed(texts, batch_size=args.batch_size, max_length=args.max_length, show=True)
            np.save(emb_f, vecs.astype("float16"))           # fp16 on disk
            with open(ids_f, "w", encoding="utf-8") as fw:
                fw.write("\n".join(ids))

    # ---- build flat IP index from all shards ----
    import faiss
    shard_files = sorted(f for f in os.listdir(shards_dir) if f.endswith(".npy"))
    if not shard_files:
        print("no shards to build from"); return
    # Add shard-by-shard (identical flat index to vstack+add, but ~3x less peak RAM:
    # no giant intermediate matrix, each shard is freed after add).
    index, all_ids = None, []
    for f in shard_files:
        m = np.load(os.path.join(shards_dir, f)).astype("float32")
        if index is None:
            index = faiss.IndexFlatIP(m.shape[1])
            print(f"building IndexFlatIP: dim {m.shape[1]}")
        index.add(m)
        del m
        with open(os.path.join(shards_dir, f.replace(".npy", ".ids.txt")), encoding="utf-8") as fr:
            all_ids.extend(fr.read().splitlines())
    print(f"added {index.ntotal} vectors from {len(shard_files)} shards")
    faiss.write_index(index, os.path.join(args.out, "corpus.faiss"))
    with open(os.path.join(args.out, "corpus_ids.txt"), "w", encoding="utf-8") as fw:
        fw.write("\n".join(all_ids))
    print(f"wrote {os.path.join(args.out, 'corpus.faiss')} ({len(all_ids)} vectors) "
          f"+ corpus_ids.txt")


if __name__ == "__main__":
    main()
