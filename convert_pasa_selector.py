#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================
# Convert PASA sft_selector data (LLM chat format: query+paper -> True/False)
# into the bge cross-encoder training format used by bge/data/clean_natural/.
#
# Output line (one per positive, sharing the query's negatives):
#   {"qid","rerank_query","positive":{"arxiv_id","passage"},
#    "negatives":[{"arxiv_id","passage"}, ...]}
#   passage == "Title: {title}\nAbstract: {abstract}"
#
# Backs up the existing bge train.jsonl to train_old.json (only the first time,
# so re-running never clobbers the true original), then writes the new train.jsonl.
#
#   python3 convert_pasa_selector.py
#   python3 convert_pasa_selector.py --in dataset/sft_selector/train.jsonl \
#       --bge-dir bge/data/clean_natural
# ==================================================================
import argparse
import hashlib
import json
import os
import re
from collections import defaultdict

# --- parsers for the selector prompt/answer ---
RE_QUERY = re.compile(r"User Query:\s*(.*?)\s*\n\nOutput format:", re.DOTALL)
RE_QUERY_FALLBACK = re.compile(r"conducting research on (.+?)\.\s*Evaluate", re.DOTALL)
RE_BLOCK = re.compile(r"Searched Paper:\s*(.*?)\n\nUser Query:", re.DOTALL)
RE_TITLE = re.compile(r"Title:\s*(.*?)\nAbstract:", re.DOTALL)
RE_ABS = re.compile(r"Abstract:\s*(.*)", re.DOTALL)
RE_LABEL = re.compile(r"(?:Decision:\s*)?(True|False)", re.IGNORECASE)


def paper_id(title: str) -> str:
    """Stable synthetic id (PASA gives no arxiv_id); used for dedup + eval grouping."""
    norm = "".join(c for c in title if c.isalnum()).lower()
    return "pasa-" + hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


def query_id(q: str) -> str:
    return "pasa_q_" + hashlib.md5(q.encode("utf-8")).hexdigest()[:12]


def parse_record(rec: dict):
    """Return (query, title, abstract, is_positive) or None if unparseable."""
    msgs = rec.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    if not user or not asst:
        return None

    mq = RE_QUERY.search(user) or RE_QUERY_FALLBACK.search(user)
    mb = RE_BLOCK.search(user)
    ml = RE_LABEL.match(asst.strip())
    if not (mq and mb and ml):
        return None
    block = mb.group(1)
    mt = RE_TITLE.search(block)
    ma = RE_ABS.search(block)
    if not (mt and ma):
        return None

    query = mq.group(1).strip()
    title = mt.group(1).strip()
    abstract = ma.group(1).strip()
    is_pos = ml.group(1).lower() == "true"
    if not query or not title:
        return None
    return query, title, abstract, is_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="dataset/sft_selector/train.jsonl")
    ap.add_argument("--bge-dir", default="bge/data/clean_natural")
    args = ap.parse_args()

    # group by query -> {pos:{pid:passage}, neg:{pid:passage}}
    groups = defaultdict(lambda: {"pos": {}, "neg": {}})
    n_in = n_bad = 0
    with open(args.inp, "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            try:
                parsed = parse_record(json.loads(line))
            except Exception:
                parsed = None
            if parsed is None:
                n_bad += 1
                continue
            query, title, abstract, is_pos = parsed
            passage = f"Title: {title}\nAbstract: {abstract}"
            bucket = groups[query]["pos" if is_pos else "neg"]
            bucket[paper_id(title)] = passage

    # emit one bge record per positive, attaching the query's negatives
    out_lines = []
    q_kept = q_no_pos = 0
    neg_per_pos = []
    for query, g in groups.items():
        if not g["pos"]:
            q_no_pos += 1
            continue
        q_kept += 1
        qid = query_id(query)
        negatives = [{"arxiv_id": pid, "passage": p} for pid, p in g["neg"].items()]
        for pid, passage in g["pos"].items():
            out_lines.append(json.dumps({
                "qid": qid,
                "rerank_query": query,
                "positive": {"arxiv_id": pid, "passage": passage},
                "negatives": negatives,
            }, ensure_ascii=False))
            neg_per_pos.append(len(negatives))

    # --- file ops: back up original once, then write new train.jsonl ---
    os.makedirs(args.bge_dir, exist_ok=True)
    train_path = os.path.join(args.bge_dir, "train.jsonl")
    backup_path = os.path.join(args.bge_dir, "train_old.json")
    if os.path.exists(train_path) and not os.path.exists(backup_path):
        os.rename(train_path, backup_path)
        print(f"backed up original -> {backup_path}")
    elif os.path.exists(backup_path):
        print(f"backup already exists ({backup_path}); not re-backing-up")

    with open(train_path, "w", encoding="utf-8") as fw:
        fw.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    avg_neg = sum(neg_per_pos) / len(neg_per_pos) if neg_per_pos else 0
    print(f"\nread {n_in} records ({n_bad} unparseable)")
    print(f"unique queries: {len(groups)} | kept (>=1 positive): {q_kept} | "
          f"dropped (no positive): {q_no_pos}")
    print(f"wrote {len(out_lines)} lines -> {train_path}")
    print(f"avg negatives per positive: {avg_neg:.2f}")


if __name__ == "__main__":
    main()
