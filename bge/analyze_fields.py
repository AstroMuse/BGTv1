"""
数据集字段分布自检脚本

目的：验证正样本与负样本的 passage 是否存在"结构性捷径"——
即某些字段（Annotation / Fields / Year）在正负样本中出现比例严重不对称，
若不对称，模型可能学到"看字段是否存在"这一非语义捷径，导致泛化失败。

用法：
    python analyze_fields.py                       # 默认分析 data/train.jsonl
    python analyze_fields.py data/val.jsonl        # 指定文件
    python analyze_fields.py data/train.jsonl data/val.jsonl data/test.jsonl
"""

import sys
import json
from collections import Counter

# 要检测的字段标记（passage 内以 "字段名:" 的形式出现）
FIELD_MARKERS = ["Title:", "Year:", "Abstract:", "Fields:", "Annotation:"]


def has_field(passage: str, marker: str) -> bool:
    return marker in passage


def analyze_file(path: str) -> dict:
    n_query = 0
    n_pos = 0
    n_neg = 0
    pos_field_counts = Counter()   # 正样本中各字段出现次数
    neg_field_counts = Counter()   # 负样本中各字段出现次数
    pos_lens = []                  # 正样本 passage 字符长度
    neg_lens = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            n_query += 1

            # ---- 正样本 ----
            pos_passage = item.get("positive", {}).get("passage", "")
            n_pos += 1
            pos_lens.append(len(pos_passage))
            for marker in FIELD_MARKERS:
                if has_field(pos_passage, marker):
                    pos_field_counts[marker] += 1

            # ---- 负样本 ----
            for neg in item.get("negatives", []):
                neg_passage = neg.get("passage", "")
                n_neg += 1
                neg_lens.append(len(neg_passage))
                for marker in FIELD_MARKERS:
                    if has_field(neg_passage, marker):
                        neg_field_counts[marker] += 1

    return {
        "path": path,
        "n_query": n_query,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_field_counts": pos_field_counts,
        "neg_field_counts": neg_field_counts,
        "pos_lens": pos_lens,
        "neg_lens": neg_lens,
    }


def pct(count: int, total: int) -> float:
    return 100.0 * count / total if total > 0 else 0.0


def avg(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def report(stats: dict):
    path = stats["path"]
    n_pos = stats["n_pos"]
    n_neg = stats["n_neg"]

    print("=" * 72)
    print(f"文件: {path}")
    print(f"query 数: {stats['n_query']}  |  正样本: {n_pos}  |  负样本: {n_neg}"
          f"  |  正负比 ≈ 1:{n_neg / max(n_pos, 1):.1f}")
    print("-" * 72)
    print(f"{'字段':<14}{'正样本含有':>14}{'负样本含有':>14}{'差异(绝对)':>14}")
    print("-" * 72)

    max_gap = 0.0
    gap_field = None
    for marker in FIELD_MARKERS:
        pos_p = pct(stats["pos_field_counts"][marker], n_pos)
        neg_p = pct(stats["neg_field_counts"][marker], n_neg)
        gap = abs(pos_p - neg_p)
        if gap > max_gap:
            max_gap = gap
            gap_field = marker
        flag = "  <== 严重不对称" if gap >= 30 else ("  <- 注意" if gap >= 10 else "")
        print(f"{marker:<14}{pos_p:>12.1f}%{neg_p:>13.1f}%{gap:>12.1f}%{flag}")

    print("-" * 72)
    print(f"passage 平均长度(字符)  正样本: {avg(stats['pos_lens']):.0f}  "
          f"负样本: {avg(stats['neg_lens']):.0f}")
    print("-" * 72)

    # ---- 结论 ----
    if max_gap >= 30:
        print(f"⚠️  结论: 字段 `{gap_field}` 在正负样本中出现比例相差 {max_gap:.1f}%，"
              f"存在明显结构性捷径，模型极可能学到非语义特征。")
        print("    建议: 统一正负样本 passage 格式（如只保留 Title + Abstract）。")
    elif max_gap >= 10:
        print(f"提示: 字段 `{gap_field}` 存在 {max_gap:.1f}% 的不对称，建议关注。")
    else:
        print("✓ 结论: 各字段在正负样本中分布较均衡，未见明显结构性捷径。")
    print()


def main():
    paths = sys.argv[1:] or ["data/train.jsonl"]
    for path in paths:
        try:
            stats = analyze_file(path)
        except FileNotFoundError:
            print(f"跳过（文件不存在）: {path}\n")
            continue
        report(stats)


if __name__ == "__main__":
    main()
