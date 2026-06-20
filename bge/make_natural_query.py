"""
生成"只留自然问句"的 query 变体数据集（用于消融对照实验）

基于已清洗的 data/clean/ ，把每条 rerank_query 中的
    Intent keywords: ...
    Key concepts: ...
等元字段行去掉，只保留开头的自然问句。passage 不动（已清洗）。

输出到 data/clean_natural/ ，不覆盖输入。

用法：
    python make_natural_query.py                      # 处理 data/clean/{train,val,test}.jsonl
    python make_natural_query.py data/clean/train.jsonl
"""

import os
import re
import sys
import json

# 识别需要剥离的元字段行（行首，大小写不敏感）
META_MARKER_RE = re.compile(r'^(Intent keywords|Key concepts)\s*:', re.IGNORECASE)


def natural_query(query: str) -> str:
    """保留首个元字段标记之前的所有内容，即自然问句部分"""
    lines = query.splitlines()
    kept = []
    for line in lines:
        if META_MARKER_RE.match(line.strip()):
            break  # 命中元字段，停止；其后内容全部丢弃
        kept.append(line)
    result = "\n".join(kept).strip()
    # 兜底：万一第一行就是元字段导致清空，则退回原始 query
    return result if result else query.strip()


def process_file(in_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    n_changed = 0
    sample_before = sample_after = None

    with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            old_q = item.get("rerank_query", "")
            new_q = natural_query(old_q)
            item["rerank_query"] = new_q

            if new_q != old_q:
                n_changed += 1
                if sample_before is None:
                    sample_before, sample_after = old_q, new_q

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"{in_path} -> {out_path}")
    print(f"  共 {n} 条 query，其中 {n_changed} 条剥离了元字段 "
          f"({100.0 * n_changed / max(n, 1):.1f}%)")
    if sample_before is not None:
        print("  示例（改前 -> 改后）：")
        print(f"    改前: {sample_before!r}")
        print(f"    改后: {sample_after!r}")
    print()


def main():
    inputs = sys.argv[1:] or [
        "data/clean/train.jsonl",
        "data/clean/val.jsonl",
        "data/clean/test.jsonl",
    ]
    out_dir = "data/clean_natural"
    for in_path in inputs:
        if not os.path.exists(in_path):
            print(f"跳过（文件不存在）: {in_path}\n")
            continue
        out_path = os.path.join(out_dir, os.path.basename(in_path))
        process_file(in_path, out_path)


if __name__ == "__main__":
    main()
