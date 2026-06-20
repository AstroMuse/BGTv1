"""
bge-reranker-v2-m3 LoRA 微调后的推理与评测脚本
支持：
1. 单条 query-doc 打分
2. 批量对测试集进行评测
3. 与基座模型的效果对比
"""

import os

# 禁用 HuggingFace 远程请求，模型从本地加载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig


def load_model_and_tokenizer(
    base_model_path: str = "models/bge-reranker-v2-m3",
    lora_model_path: str = None,
    device: str = None,
):
    """
    加载模型和分词器
    如果指定了 lora_model_path，则加载 LoRA adapter
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_path,
        num_labels=1,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    )

    if lora_model_path and os.path.exists(lora_model_path):
        print(f"加载 LoRA adapter: {lora_model_path}")
        model = PeftModel.from_pretrained(base_model, lora_model_path)
        model = model.merge_and_unload()  # 合并权重，加速推理
        print("LoRA 权重已合并")
    else:
        model = base_model
        print("使用基座模型（无 LoRA）")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    return model, tokenizer, device


@torch.no_grad()
def compute_score(
    model,
    tokenizer,
    query: str,
    doc_text: str,
    max_length: int = 512,
    device: str = "cuda",
) -> float:
    """
    计算单个 query-document pair 的相关性分数
    """
    inputs = tokenizer(
        query,
        doc_text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    logits = model(**inputs).logits
    # logits 是单个值，直接返回（未经 sigmoid，值域为实数）
    score = logits.cpu().item()
    return score


@torch.no_grad()
def compute_scores_batch(
    model,
    tokenizer,
    pairs: List[Tuple[str, str]],
    max_length: int = 512,
    device: str = "cuda",
    batch_size: int = 32,
) -> List[float]:
    """
    批量计算 query-document pair 的相关性分数
    """
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size), desc="批量打分"):
        batch_pairs = pairs[i:i + batch_size]
        queries, docs = zip(*batch_pairs)

        inputs = tokenizer(
            list(queries),
            list(docs),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        logits = model(**inputs).logits
        scores.extend(logits.squeeze(-1).cpu().tolist())

    return scores


def format_doc(doc: Dict) -> str:
    """格式化文档文本：直接返回 passage（已预格式化 Title/Year/Abstract），由 tokenizer max_length 截断"""
    return doc.get("passage", "")


def evaluate(
    model,
    tokenizer,
    data_path: str,
    device: str = "cuda",
    max_length: int = 512,
    batch_size: int = 32,
    threshold: float = 0.0,
) -> Dict:
    """
    在测试集上评测模型（按 query 分组，阈值分类）

    流程：
    1. 按 qid 将同一 query 的多行数据聚合
    2. 每个 query 下的候选文档按 arxiv_id 去重
    3. 模型对所有候选打分
    4. 分数 >= threshold 的预测为"正样本"
    5. 与真实正样本集合比对，计算 Precision / Recall / F1

    指标：
    - Macro Precision / Recall / F1（各 query 等权平均）
    - Micro Precision / Recall / F1（按总 TP/FP/FN 计算）
    - 正负样本分数分布统计
    """
    # ---- 1. 加载数据，按 qid 分组 ----
    # query_groups[qid] = {
    #     "query_text": str,
    #     "pos_ids": set of arxiv_id,
    #     "candidates": {arxiv_id: formatted_doc_text},
    # }
    query_groups: Dict[str, dict] = defaultdict(lambda: {
        "query_text": "",
        "pos_ids": set(),
        "candidates": {},
    })

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            qid = item.get("qid", item["rerank_query"])
            group = query_groups[qid]
            group["query_text"] = item["rerank_query"]

            # 正样本
            pos_doc = item["positive"]
            pos_id = pos_doc.get("arxiv_id", format_doc(pos_doc))
            group["pos_ids"].add(pos_id)
            if pos_id not in group["candidates"]:
                group["candidates"][pos_id] = format_doc(pos_doc)

            # 负样本（只添加未出现过的 doc，避免重复）
            for neg in item.get("negatives", []):
                neg_id = neg.get("arxiv_id", format_doc(neg))
                if neg_id not in group["candidates"]:
                    group["candidates"][neg_id] = format_doc(neg)

    print(f"评测数据: {len(query_groups)} 个唯一 query"
          f"（原始 {sum(1 for _ in open(data_path))} 行）")

    total_pos = sum(len(g["pos_ids"]) for g in query_groups.values())
    total_cand = sum(len(g["candidates"]) for g in query_groups.values())
    print(f"总正样本(去重): {total_pos}, 总候选文档(去重): {total_cand}")

    # ---- 2. 按 query 逐组打分 ----
    query_results = []

    for qid, group in tqdm(query_groups.items(), desc="评测中"):
        query_text = group["query_text"]
        pos_ids = group["pos_ids"]
        candidates = group["candidates"]

        # 保持 doc_ids 与 pairs 顺序一致
        doc_ids = list(candidates.keys())
        doc_texts = [candidates[did] for did in doc_ids]
        pairs = [(query_text, dt) for dt in doc_texts]

        scores = compute_scores_batch(
            model, tokenizer, pairs,
            max_length=max_length, device=device, batch_size=batch_size,
        )

        # 阈值分类 + 收集正负样本分数
        pred_pos_ids = set()
        pos_scores = []
        neg_scores = []
        for did, score in zip(doc_ids, scores):
            if did in pos_ids:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
            if score >= threshold:
                pred_pos_ids.add(did)

        tp = len(pred_pos_ids & pos_ids)
        fp = len(pred_pos_ids - pos_ids)
        fn = len(pos_ids - pred_pos_ids)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        query_results.append({
            "qid": qid,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pos_scores": pos_scores,
            "neg_scores": neg_scores,
        })

    # ---- 3. 汇总指标 ----
    # Macro 平均：各 query 等权
    macro_precision = np.mean([r["precision"] for r in query_results])
    macro_recall = np.mean([r["recall"] for r in query_results])
    macro_f1 = np.mean([r["f1"] for r in query_results])

    # Micro 平均：汇总所有 query 的 TP/FP/FN 后统一计算
    sum_tp = sum(r["tp"] for r in query_results)
    sum_fp = sum(r["fp"] for r in query_results)
    sum_fn = sum(r["fn"] for r in query_results)
    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall /
                (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0)

    # Score statistics
    all_pos_scores = []
    all_neg_scores = []
    for r in query_results:
        all_pos_scores.extend(r["pos_scores"])
        all_neg_scores.extend(r["neg_scores"])

    pos_mean = np.mean(all_pos_scores) if all_pos_scores else 0.0
    neg_mean = np.mean(all_neg_scores) if all_neg_scores else 0.0
    pos_std = np.std(all_pos_scores) if all_pos_scores else 0.0
    neg_std = np.std(all_neg_scores) if all_neg_scores else 0.0

    results = {
        "num_queries": len(query_groups),
        "num_candidates": total_cand,
        "num_positives": total_pos,
        "threshold": threshold,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "pos_mean_score": pos_mean,
        "neg_mean_score": neg_mean,
        "pos_std": pos_std,
        "neg_std": neg_std,
        "score_margin": pos_mean - neg_mean,
        "per_query": query_results,
    }

    return results


def compare_models(
    base_model_path: str,
    lora_model_path: str,
    test_data_path: str,
    device: str = "cuda",
    max_length: int = 512,
    threshold: float = 0.0,
):
    """
    对比基座模型和 LoRA 微调模型的效果
    """
    print("\n" + "=" * 60)
    print("基座模型 vs LoRA 微调模型 对比评测")
    print("=" * 60)

    # ---- 评测基座模型 ----
    print("\n[1/2] 评测基座模型...")
    base_model, tokenizer, device = load_model_and_tokenizer(
        base_model_path=base_model_path,
        lora_model_path=None,  # 不加载 LoRA
        device=device,
    )
    base_results = evaluate(
        base_model, tokenizer, test_data_path,
        device=device, max_length=max_length, threshold=threshold,
    )
    del base_model
    torch.cuda.empty_cache()

    # ---- 评测 LoRA 模型 ----
    print("\n[2/2] 评测 LoRA 微调模型...")
    lora_model, _, _ = load_model_and_tokenizer(
        base_model_path=base_model_path,
        lora_model_path=lora_model_path,
        device=device,
    )
    lora_results = evaluate(
        lora_model, tokenizer, test_data_path,
        device=device, max_length=max_length, threshold=threshold,
    )
    del lora_model
    torch.cuda.empty_cache()

    # ---- 对比结果 ----
    print("\n" + "=" * 60)
    print(f"评测结果对比（阈值 = {threshold}）")
    print("=" * 60)
    print(f"{'指标':<25} {'基座模型':>15} {'LoRA微调':>15} {'提升':>15}")
    print("-" * 70)

    metrics_display = [
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("micro_precision", "Micro Precision"),
        ("micro_recall", "Micro Recall"),
        ("micro_f1", "Micro F1"),
        ("score_margin", "Score Margin"),
        ("pos_mean_score", "Pos Mean Score"),
        ("neg_mean_score", "Neg Mean Score"),
    ]

    for metric_key, metric_name in metrics_display:
        base_val = base_results[metric_key]
        lora_val = lora_results[metric_key]
        delta = lora_val - base_val
        delta_str = f"{delta:+.4f}"
        print(f"{metric_name:<25} {base_val:>15.4f} {lora_val:>15.4f} {delta_str:>15}")

    return base_results, lora_results


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="bge-reranker-v2-m3 推理与评测")
    parser.add_argument("--base_model", type=str, default="models/bge-reranker-v2-m3",
                        help="基座模型路径")
    parser.add_argument("--lora_model", type=str, default="output/lora-bge-reranker/best_model",
                        help="LoRA adapter 路径")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl",
                        help="测试数据路径")
    parser.add_argument("--mode", type=str, default="compare",
                        choices=["compare", "base_only", "lora_only", "interactive"],
                        help="评测模式: compare(对比), base_only, lora_only, interactive(交互打分)")
    parser.add_argument("--query", type=str, default=None,
                        help="交互模式下的查询文本")
    parser.add_argument("--doc", type=str, default=None,
                        help="交互模式下的文档文本")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default=None,
                        help="设备 (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="正样本分类阈值，logit >= threshold 预测为正（默认 0.0）")

    args = parser.parse_args()

    if args.mode == "compare":
        compare_models(
            base_model_path=args.base_model,
            lora_model_path=args.lora_model,
            test_data_path=args.test_data,
            device=args.device,
            max_length=args.max_length,
            threshold=args.threshold,
        )

    elif args.mode in ("base_only", "lora_only"):
        lora_path = args.lora_model if args.mode == "lora_only" else None
        if args.mode == "lora_only" and not os.path.exists(lora_path):
            print(f"错误: LoRA 模型路径不存在: {lora_path}")
            exit(1)

        model, tokenizer, device = load_model_and_tokenizer(
            base_model_path=args.base_model,
            lora_model_path=lora_path,
            device=args.device,
        )

        if args.query and args.doc:
            # 单条打分
            score = compute_score(
                model, tokenizer, args.query, args.doc,
                max_length=args.max_length, device=device,
            )
            print(f"\nQuery: {args.query[:100]}...")
            print(f"Doc: {args.doc[:100]}...")
            print(f"Relevance Score: {score:.4f}")
        else:
            # 测试集评测
            results = evaluate(
                model, tokenizer, args.test_data,
                device=device, max_length=args.max_length,
                threshold=args.threshold,
            )
            print(f"\n评测结果（阈值 = {results['threshold']}）:")
            metric_keys = [
                "num_queries", "num_candidates", "num_positives",
                "macro_precision", "macro_recall", "macro_f1",
                "micro_precision", "micro_recall", "micro_f1",
                "score_margin", "pos_mean_score", "neg_mean_score",
            ]
            for k in metric_keys:
                if k in results:
                    print(f"  {k}: {results[k]:.4f}")

    elif args.mode == "interactive":
        lora_path = args.lora_model if os.path.exists(args.lora_model) else None
        model, tokenizer, device = load_model_and_tokenizer(
            base_model_path=args.base_model,
            lora_model_path=lora_path,
            device=args.device,
        )

        print("\n交互打分模式（输入 'quit' 退出）")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() == "quit":
                break
            doc = input("Doc: ").strip()
            if doc.lower() == "quit":
                break

            score = compute_score(
                model, tokenizer, query, doc,
                max_length=args.max_length, device=device,
            )
            print(f"→ Relevance Score: {score:.4f}")
