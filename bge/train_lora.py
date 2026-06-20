"""
bge-reranker-v2-m3 LoRA 微调脚本
基于正负样本对的精排模型微调

数据格式：JSONL，每行包含 qid, rerank_query, positive, negatives
训练策略：将 query-doc 拼接，正样本 label=1，负样本 label=0，使用 BCE loss
"""

import os

# 禁用 HuggingFace 远程请求，模型从本地加载
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    EarlyStoppingCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
)
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm


# ============================================================================
# 配置
# ============================================================================

class Config:
    # 模型路径
    model_name_or_path = "models/bge-reranker-v2-m3"
    tokenizer_path = "models/bge-reranker-v2-m3"

    # 数据路径
    train_data_path = "data/train.jsonl"
    val_data_path = "data/val.jsonl"
    test_data_path = "data/test.jsonl"

    # 输出路径
    output_dir = "output/lora-bge-reranker"
    logging_dir = "output/logs"

    # 训练参数
    max_length = 512  # query + document 最大 token 数
    per_device_train_batch_size = 8
    per_device_eval_batch_size = 16
    gradient_accumulation_steps = 2
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_ratio = 0.1
    num_train_epochs = 3
    max_negatives_per_query = -1  # -1 = 使用全部负样本；>0 = 限制最大负样本数
    seed = 42
    fp16 = torch.cuda.is_available()
    save_steps = 100
    eval_steps = 100           # 每 100 步验证一次
    logging_steps = 50
    early_stopping_patience = 5  # 连续 5 次 F1 不上升则停止

    # 混合精度：优先 bf16（XLM-R-large 用 fp16 易出 NaN / 梯度被跳过）
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16  # 仅在不支持 bf16 时回退 fp16

    gradient_checkpointing = True  # 长序列必备，大幅节省激活值显存

    # LoRA 参数
    lora_r = 16
    lora_alpha = 64  # alpha/r = 2，LoRA 推荐比例
    lora_dropout = 0.1
    # self.* 精确匹配 attention 子层，避开 classifier（peft 0.10 用子串匹配）
    lora_target_modules = ["self.query", "self.value", "self.key", "attention.output.dense"]


# ============================================================================
# 数据集
# ============================================================================

class RerankerDataset(Dataset):
    """
    将 JSONL 正负样本对转换为 query-doc pair 分类数据集。

    数据格式：
      {"qid": "...", "rerank_query": "...",
       "positive": {"arxiv_id": "...", "passage": "..."},
       "negatives": [{"arxiv_id": "...", "passage": "..."}, ...]}

    核心设计：__init__ 阶段将所有正负样本展开为扁平列表，每个 epoch 遍历全部负样本。
    DataLoader 配合 shuffle=True，每个 epoch 以不同顺序遍历所有样本，
    既能保证负样本不浪费，又能通过 shuffle 打乱顺序避免模型记忆样本顺序。

    展开结构（每个 query 展开为 1 个正样本 + 全部 N 个负样本）：
      query_0: [(q0, pos, 1), (q0, neg_0, 0), (q0, neg_1, 0), ...]
      query_1: [(q1, pos, 1), (q1, neg_0, 0), (q1, neg_1, 0), ...]
      ...
    全部展开后 → 打平为一条长列表 → DataLoader shuffle 后混合训练。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        max_negatives: int = -1,  # -1 表示使用全部负样本，>0 表示限制最大负样本数
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # ---- 一次性展开：正样本 + 全部负样本 ----
        # 每个样本记录 qid 和 arxiv_id，供 query 级评估使用
        self.samples: List[Dict] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                qid = item["qid"]
                query = item["rerank_query"]
                pos_id = item["positive"]["arxiv_id"]

                # 正样本
                pos_text = format_doc(item["positive"])
                self.samples.append({
                    "query": query,
                    "doc": pos_text,
                    "label": 1.0,
                    "qid": qid,
                    "arxiv_id": pos_id,
                })

                # 全部负样本（不做采样，全部参与训练）
                negatives = item.get("negatives", [])
                if max_negatives > 0:
                    negatives = negatives[:max_negatives]

                for neg in negatives:
                    self.samples.append({
                        "query": query,
                        "doc": format_doc(neg),
                        "label": 0.0,
                        "qid": qid,
                        "arxiv_id": neg["arxiv_id"],
                    })

        total_pos = sum(1 for s in self.samples if s["label"] == 1)
        total_neg = sum(1 for s in self.samples if s["label"] == 0)
        print(f"从 {data_path} 展开 {len(self.samples)} 条样本 "
              f"(正样本: {total_pos}, 负样本: {total_neg}, "
              f"正负比 ≈ 1:{total_neg / max(total_pos, 1):.1f})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        encoded = self.tokenizer(
            sample["query"],
            sample["doc"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )

        encoded["labels"] = torch.tensor(sample["label"], dtype=torch.float32)
        return encoded

    def get_qids(self) -> List[str]:
        """返回所有样本的 qid 列表（与 __getitem__ 同序），供 query 级评估使用"""
        return [s["qid"] for s in self.samples]

    def get_doc_ids(self) -> List[str]:
        """返回所有样本的 arxiv_id 列表（与 __getitem__ 同序），供 query 级评估去重"""
        return [s["arxiv_id"] for s in self.samples]


def format_doc(doc: Dict) -> str:
    """格式化文档文本：直接返回 passage（已预格式化 Title/Year/Abstract），由 tokenizer max_length 截断"""
    return doc.get("passage", "")


# ============================================================================
# 自定义 Trainer（使用 BCEWithLogitsLoss）
# ============================================================================

class RerankerTrainer(Trainer):
    """
    自定义 Trainer，使用 BCEWithLogitsLoss 替代默认的 MSELoss
    bge-reranker-v2-m3 使用 num_labels=1，默认是 regression + MSE，
    但精排任务应使用 BCE loss：正样本→高分，负样本→低分

    评估：覆盖 evaluate() 以支持 query 级 Precision / Recall / F1
    """
    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_qids: List[str] = []
        self._eval_doc_ids: List[str] = []
        # 正样本权重：抵消负样本占比大的 bias，加速正样本分数过零
        self._pos_weight = torch.tensor([pos_weight]) if pos_weight else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()  # (batch_size,)
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)    # (batch_size,)
        pos_weight = self._pos_weight.to(logits.device) if self._pos_weight is not None else None
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """覆盖 evaluate：在调用父类前捕获 qid/doc_id 顺序，供 compute_metrics 使用"""
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(ds, RerankerDataset):
            self._eval_qids = ds.get_qids()
            self._eval_doc_ids = ds.get_doc_ids()
        return super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

    def _compute_metrics(self, eval_pred):
        """
        Query 级评估指标：按 qid 分组，阈值分类，计算 P / R / F1

        逻辑与 inference.py evaluate() 完全一致：
        1. 按 qid 将样本分组
        2. 每组内 logit >= 0 的 doc 预测为正
        3. 与真实正样本集合比对，计算 P/R/F1
        4. Macro 平均（各 query 等权）

        另含与部署(top-K 排序闸门)对齐的排序指标 hit@1 / MRR——选 best model 用这个,
        阈值类 P/R/F1 仅作参考(pos_weight 会整体抬高 logit,threshold=0 不是工作点)。
        """
        logits = eval_pred.predictions.squeeze(-1)          # (n_samples,)
        labels = eval_pred.label_ids.astype(int)             # (n_samples,)
        predictions = (logits > 0).astype(int)               # 阈值 = 0

        # ---- Query 级分组 ----
        # query_groups[qid] = {"pos_ids": set, "pred_pos_ids": set}
        qid_groups: Dict[str, dict] = defaultdict(lambda: {"pos_ids": set(), "pred_pos_ids": set()})

        for qid, did, pred, label in zip(self._eval_qids, self._eval_doc_ids, predictions, labels):
            group = qid_groups[qid]
            if label == 1:
                group["pos_ids"].add(did)
            if pred == 1:
                group["pred_pos_ids"].add(did)

        # ---- 每 query 算 P / R / F1 ----
        per_q_f1 = []
        per_q_precision = []
        per_q_recall = []
        sum_tp = sum_fp = sum_fn = 0
        for qid, g in qid_groups.items():
            tp = len(g["pred_pos_ids"] & g["pos_ids"])
            fp = len(g["pred_pos_ids"] - g["pos_ids"])
            fn = len(g["pos_ids"] - g["pred_pos_ids"])
            sum_tp += tp
            sum_fp += fp
            sum_fn += fn
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_q_precision.append(p)
            per_q_recall.append(r)
            per_q_f1.append(f1)

        # Macro 平均
        macro_precision = np.mean(per_q_precision) if per_q_precision else 0.0
        macro_recall = np.mean(per_q_recall) if per_q_recall else 0.0
        macro_f1 = np.mean(per_q_f1) if per_q_f1 else 0.0

        # Micro 平均
        micro_p = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
        micro_r = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

        # ---- Query 级排序指标(与部署对齐:prefilter 按 logit 降序取 top-K,不用阈值)----
        # hit@1: 排第一的 doc 是正样本的 query 占比;MRR: 首个正样本排名倒数的均值
        qid_scores: Dict[str, list] = defaultdict(list)  # qid -> [(logit, label)]
        for qid, logit, label in zip(self._eval_qids, logits, labels):
            qid_scores[qid].append((float(logit), int(label)))

        hit1_list, mrr_list = [], []
        for g in qid_scores.values():
            if not any(lbl == 1 for _, lbl in g):
                continue  # 无正样本的 query 不计入排序指标
            g.sort(key=lambda x: x[0], reverse=True)
            first_pos_rank = next(i for i, (_, lbl) in enumerate(g, 1) if lbl == 1)
            hit1_list.append(1.0 if first_pos_rank == 1 else 0.0)
            mrr_list.append(1.0 / first_pos_rank)

        hit1 = float(np.mean(hit1_list)) if hit1_list else 0.0
        mrr = float(np.mean(mrr_list)) if mrr_list else 0.0

        # 分数分布统计
        pos_scores = logits[labels == 1]
        neg_scores = logits[labels == 0]
        pos_mean = pos_scores.mean().item() if len(pos_scores) > 0 else 0.0
        neg_mean = neg_scores.mean().item() if len(neg_scores) > 0 else 0.0

        return {
            "mrr": mrr,
            "hit1": hit1,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "pos_score": pos_mean,
            "neg_score": neg_mean,
            "margin": pos_mean - neg_mean,
        }


# ============================================================================
# Data Collator
# ============================================================================

class RerankerDataCollator:
    """自定义 DataCollator，处理 labels"""

    def __init__(self, tokenizer, padding=True, max_length=None):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        labels = [f.pop("labels") for f in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        batch["labels"] = torch.stack(labels)
        return batch


# ============================================================================
# 训练
# ============================================================================

def train():
    config = Config()

    # 设置随机种子
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    print("=" * 60)
    print("bge-reranker-v2-m3 LoRA 微调")
    print("=" * 60)
    print(f"GPU 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    # ---- 加载 Tokenizer 和模型 ----
    print("加载 Tokenizer 和模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_path,
        use_fast=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name_or_path,
        num_labels=1,  # 单输出 logit，配合 BCEWithLogitsLoss
        trust_remote_code=True,
    )

    # 设置 pad_token（如果未设置）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # ---- 配置 LoRA ----
    print("\n配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        modules_to_save=["classifier"],  # 分类头全参数训练，适配新任务
    )

    model = get_peft_model(model, lora_config)

    # 关键：梯度检查点 + 冻结基座时，必须显式让输入 embedding 产生梯度，
    # 否则被 checkpoint 的层内 LoRA 参数梯度为 None，loss 完全不下降。
    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    model.print_trainable_parameters()

    # ---- 加载数据集 ----
    print("\n加载数据集...")
    train_dataset = RerankerDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_negatives=config.max_negatives_per_query,
    )
    val_dataset = RerankerDataset(
        data_path=config.val_data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_negatives=config.max_negatives_per_query,
    )

    # ---- Training Arguments ----
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        logging_dir=config.logging_dir,
        logging_steps=config.logging_steps,
        eval_strategy="steps",  # renamed from evaluation_strategy in transformers 4.41+
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        fp16=config.fp16,
        bf16=config.bf16,
        load_best_model_at_end=True,
        # MRR 与部署一致(prefilter 按 logit 取 top-K),且比 threshold-F1 平滑、比 margin 有界
        metric_for_best_model="mrr",
        greater_is_better=True,
        report_to="none",  # 不上报 wandb/tensorboard，用 logging_dir 记录即可
        dataloader_num_workers=0,
        seed=config.seed,
        remove_unused_columns=False,
        gradient_checkpointing=config.gradient_checkpointing,
        # use_reentrant=False 与冻结基座的梯度检查点兼容性更好
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ---- Trainer ----
    data_collator = RerankerDataCollator(tokenizer=tokenizer)

    # 计算正样本权重 = 负样本数/正样本数，缓解样本不平衡导致的全部分数偏向负值
    train_pos = sum(1 for s in train_dataset.samples if s["label"] == 1)
    train_neg = sum(1 for s in train_dataset.samples if s["label"] == 0)
    pos_weight = train_neg / train_pos if train_pos > 0 else 1.0
    print(f"\nBCE pos_weight: {pos_weight:.1f} (neg:pos = {train_neg}:{train_pos})")

    trainer = RerankerTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience,
                early_stopping_threshold=0.0,
            ),
        ],
    )
    # 挂载 query 级评估（在 __init__ 之后设置，因 Trainer 构造函数会覆盖 compute_metrics）
    trainer.compute_metrics = trainer._compute_metrics

    # ---- 开始训练 ----
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)

    trainer.train()

    # ---- 保存最优模型 ----
    print("\n保存最佳 LoRA 模型...")
    best_model_path = os.path.join(config.output_dir, "best_model")
    trainer.save_model(best_model_path)
    tokenizer.save_pretrained(best_model_path)
    print(f"模型已保存至: {best_model_path}")

    # ---- 测试集评估 ----
    print("\n在测试集上评估...")
    test_dataset = RerankerDataset(
        data_path=config.test_data_path,
        tokenizer=tokenizer,
        max_length=config.max_length,
        max_negatives=config.max_negatives_per_query,
    )
    test_results = trainer.evaluate(test_dataset)
    print("\n测试集结果:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}")

    return trainer


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA 微调 bge-reranker-v2-m3")
    parser.add_argument("--model_path", type=str, default=None,
                        help="基座模型路径，默认 models/bge-reranker-v2-m3")
    parser.add_argument("--train_data", type=str, default=None,
                        help="训练数据路径")
    parser.add_argument("--val_data", type=str, default=None,
                        help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--max_negatives", type=int, default=None,
                        help="每个 query 最多使用的负样本数，-1=全部使用（默认-1）")
    parser.add_argument("--no_fp16", action="store_true",
                        help="禁用 fp16 混合精度")

    args = parser.parse_args()

    # 覆盖配置
    if args.model_path:
        Config.model_name_or_path = args.model_path
        Config.tokenizer_path = args.model_path
    if args.train_data:
        Config.train_data_path = args.train_data
    if args.val_data:
        Config.val_data_path = args.val_data
    if args.output_dir:
        Config.output_dir = args.output_dir
    if args.epochs:
        Config.num_train_epochs = args.epochs
    if args.batch_size:
        Config.per_device_train_batch_size = args.batch_size
    if args.lr:
        Config.learning_rate = args.lr
    if args.max_length:
        Config.max_length = args.max_length
    if args.lora_r:
        Config.lora_r = args.lora_r
    if args.lora_alpha:
        Config.lora_alpha = args.lora_alpha
    if args.max_negatives is not None:
        Config.max_negatives_per_query = args.max_negatives
    if args.no_fp16:
        Config.fp16 = False
        Config.bf16 = False  # 全精度 fp32，便于排查数值问题

    train()
