# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# Mine hard negatives for the bge cross-encoder from the SPECTER2 index.
# For each AutoScholarQuery query: embed the question, retrieve top-K from the
# corpus, drop the gold answers + a top `--margin` band (likely-unlabeled-positive
# guard), then take hard negatives from the top ranks and a few mid-rank ones.
# Positives = the gold papers (Title+Abstract looked up in recovered.db).
# Output = bge clean_natural format (passage = "Title: ..\nAbstract: ..").
#
#   python3 mine_hard_negatives.py                 # all queries
#   python3 mine_hard_negatives.py --limit 5       # quick test
# ==================================================================
import argparse
import json
import os
import random
import re
import sqlite3

import numpy as np

from local_db_v2 import db_path as DEFAULT_DB
from specter_embed import embed

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")


def norm_id(x):
    m = ARXIV_RE.search(str(x))
    return m.group(1) if m else str(x).strip()


def get_doc(db, pid):
    row = db.execute("SELECT data FROM arxiv_docs WHERE arxiv_id=?", (pid,)).fetchone()
    if not row:
        # gold ids may carry version/prefix; try base id
        row = db.execute("SELECT data FROM arxiv_docs WHERE arxiv_id=?", (norm_id(pid),)).fetchone()
    if not row:
        return None
    try:
        d = json.loads(row[0])
    except Exception:
        return None
    if d.get("title") and d.get("abstract"):
        return {"title": d["title"], "abstract": d["abstract"]}
    return None


def passage(doc):
    return f"Title: {doc['title']}\nAbstract: {doc['abstract']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="dataset/AutoScholarQuery/train.jsonl")
    ap.add_argument("--index-dir", default="bge/data/specter_index")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default="bge/data/specter_hardneg/train.jsonl")
    ap.add_argument("--topk", type=int, default=120)
    ap.add_argument("--margin", type=int, default=5, help="skip this many top ranks (false-neg guard)")
    ap.add_argument("--n-hard", type=int, default=8)
    ap.add_argument("--n-mid", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import faiss
    index = faiss.read_index(os.path.join(args.index_dir, "corpus.faiss"))
    with open(os.path.join(args.index_dir, "corpus_ids.txt"), encoding="utf-8") as fr:
        corpus_ids = fr.read().splitlines()
    print(f"index: {index.ntotal} vectors")

    rows = [json.loads(l) for l in open(args.queries, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    questions = [r["question"] for r in rows]
    print(f"queries: {len(questions)} | embedding ...")
    qv = embed(questions, batch_size=args.batch_size, show=True)

    _, idxs = index.search(qv, args.topk)        # (Q, topk) corpus positions

    db = sqlite3.connect(args.db)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_out = n_noskip_pos = 0
    neg_counts = []
    random.seed(123)
    with open(args.out, "w", encoding="utf-8") as fw:
        for r, ranked in zip(rows, idxs):
            q = r["question"]
            gold_ids = {norm_id(i) for i in (r.get("answer_arxiv_id") or [])}
            cand = [corpus_ids[j] for j in ranked if j >= 0]
            # drop gold + the top `margin` band, then split hard / mid
            kept = [c for c in cand if norm_id(c) not in gold_ids]
            kept = kept[args.margin:]
            hard = kept[:args.n_hard]
            mid_pool = kept[args.n_hard:]
            mid = random.sample(mid_pool, min(args.n_mid, len(mid_pool))) if mid_pool else []
            neg_ids = hard + mid
            neg_docs = [(nid, get_doc(db, nid)) for nid in neg_ids]
            negatives = [{"arxiv_id": norm_id(nid), "passage": passage(d)}
                         for nid, d in neg_docs if d]
            if not negatives:
                continue

            qid = r.get("qid", "asq_" + str(n_out))
            wrote = False
            for gid in (r.get("answer_arxiv_id") or []):
                gdoc = get_doc(db, gid)
                if not gdoc:
                    continue
                fw.write(json.dumps({
                    "qid": qid,
                    "rerank_query": q,
                    "positive": {"arxiv_id": norm_id(gid), "passage": passage(gdoc)},
                    "negatives": negatives,
                }, ensure_ascii=False) + "\n")
                n_out += 1
                wrote = True
            if wrote:
                n_noskip_pos += 1
                neg_counts.append(len(negatives))

    avg = sum(neg_counts) / len(neg_counts) if neg_counts else 0
    print(f"\nwrote {n_out} lines ({n_noskip_pos} queries with >=1 resolvable gold) -> {args.out}")
    print(f"avg negatives per positive: {avg:.1f}")


if __name__ == "__main__":
    main()
