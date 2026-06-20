"""
数据清洗脚本：消除正负样本的格式捷径

将每个 passage（正样本 + 全部负样本）重新格式化为仅保留语义内容：
    Title: <标题>
    Abstract: <摘要>
去掉 Year / Fields / Annotation 等会造成格式泄漏的元字段，
使正负样本结构完全一致，消除模型走捷径的可能。

- 不覆盖原始数据，输出到 data/clean/ 下的同名文件
- 完成后自动调用 analyze_fields.py 复测清洗结果

用法：
    python clean_data.py                 # 处理 data/train.jsonl / val.jsonl / test.jsonl
    python clean_data.py data/train.jsonl  # 仅处理指定文件
"""

import os
import re
import sys
import json
import subprocess

# 识别 passage 中以 "字段名:" 开头的行（行首锚定，避免误伤摘要正文里的冒号）
MARKER_RE = re.compile(r'^(Title|Year|Abstract|Fields|Annotation):[ \t]*', re.MULTILINE)


def parse_passage(passage: str) -> dict:
    """把 passage 按字段标记切成 {字段名: 内容}"""
    matches = list(MARKER_RE.finditer(passage))
    sections = {}
    for i, m in enumerate(matches):
        key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(passage)
        sections[key] = passage[start:end].strip()
    return sections


def clean_passage(passage: str) -> str:
    """仅保留 Title 与 Abstract，统一格式"""
    s = parse_passage(passage)
    title = s.get("Title", "").strip()
    abstract = s.get("Abstract", "").strip()

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if abstract:
        parts.append(f"Abstract: {abstract}")
    # 兜底：万一某条 passage 完全没有可识别字段，则原样保留，避免清空
    return "\n".join(parts) if parts else passage.strip()


def clean_doc(doc: dict) -> dict:
    """清洗单个文档对象的 passage 字段，其余字段（arxiv_id 等）保持不变"""
    if isinstance(doc, dict) and "passage" in doc:
        doc = dict(doc)
        doc["passage"] = clean_passage(doc["passage"])
    return doc


def clean_file(in_path: str, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
            open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if "positive" in item:
                item["positive"] = clean_doc(item["positive"])
            if "negatives" in item:
                item["negatives"] = [clean_doc(neg) for neg in item["negatives"]]

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    inputs = sys.argv[1:] or [
        "data/train.jsonl",
        "data/val.jsonl",
        "data/test.jsonl",
    ]

    out_dir = "data/clean"
    out_paths = []
    for in_path in inputs:
        if not os.path.exists(in_path):
            print(f"跳过（文件不存在）: {in_path}")
            continue
        out_path = os.path.join(out_dir, os.path.basename(in_path))
        n = clean_file(in_path, out_path)
        print(f"清洗 {in_path} -> {out_path}  ({n} 条 query)")
        out_paths.append(out_path)

    if not out_paths:
        print("没有可处理的文件。")
        return

    # ---- 自动复测 ----
    print("\n" + "=" * 72)
    print("复测清洗结果（各字段差异应降为 0，仅 Title/Abstract 为 100%）")
    print("=" * 72)
    subprocess.run([sys.executable, "analyze_fields.py", *out_paths])


if __name__ == "__main__":
    main()
