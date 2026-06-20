# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# BGE cross-encoder reranker wrapper.
#
# Wraps the fine-tuned bge-reranker-v2-m3 (+ LoRA) under ./bge/ as a relevance
# reranker for the SPAR pipeline. Loads lazily and degrades gracefully: if the
# deps (torch/transformers/peft), the model files, or a device are unavailable,
# scoring returns None and the caller keeps the original ranking.
#
# Requires (in whatever env runs this): torch, transformers, peft, sentencepiece.
# These live in the bge/ project env; the default SPAR env may lack `peft`.
# ==================================================================
import os
import importlib.util
import traceback
from typing import List, Dict, Optional

from log import logger
from global_config import (
    BGE_BASE_MODEL_PATH,
    BGE_LORA_PATH,
    RERANK_BATCH_SIZE,
    RERANK_MAX_LENGTH,
    RERANK_DEVICE,
)

_BGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bge")


def _load_bge_inference_module():
    """Import ./bge/inference.py as a uniquely-named module (avoids name clashes)."""
    path = os.path.join(_BGE_DIR, "inference.py")
    spec = importlib.util.spec_from_file_location("bge_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class BgeReranker:
    """Lazy cross-encoder reranker. `score(query, docs)` -> list of logits or None."""

    def __init__(self, base_path: str = None, lora_path: str = None):
        self.base_path = base_path or BGE_BASE_MODEL_PATH
        self.lora_path = lora_path or BGE_LORA_PATH
        self._inf = None
        self._model = None
        self._tok = None
        self._device = None
        self._failed = False  # don't retry a hard load failure every query

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._failed:
            return False
        try:
            # CPU mode: hide the GPU BEFORE torch is imported, so peft's device
            # probe (infer_device) doesn't try a CUDA alloc. Works only if torch
            # isn't imported yet (true in the SPAR pipeline). To force it from the
            # shell regardless, prefix the run with CUDA_VISIBLE_DEVICES="".
            if RERANK_DEVICE == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            self._inf = _load_bge_inference_module()
            lora = self.lora_path if self.lora_path and os.path.exists(self.lora_path) else None
            self._model, self._tok, self._device = self._inf.load_model_and_tokenizer(
                base_model_path=self.base_path,
                lora_model_path=lora,
                device=RERANK_DEVICE,
            )
            logger.info(
                f"BGE reranker loaded (device={self._device}, "
                f"lora={'yes' if lora else 'no'}, base={self.base_path})"
            )
            return True
        except Exception:
            self._failed = True
            logger.error(
                "BGE reranker load failed (missing deps/model/GPU?); "
                f"rerank will be skipped: {traceback.format_exc()}"
            )
            return False

    @staticmethod
    def passage(doc: Dict) -> str:
        """Match the training passage format: 'Title: {title}\\nAbstract: {abstract}'."""
        return f"Title: {doc.get('title','')}\nAbstract: {doc.get('abstract','')}"

    def score(self, query: str, docs: List[Dict]) -> Optional[List[float]]:
        """Cross-encoder logits for (query, passage) pairs, aligned with `docs`."""
        if not docs or not self._ensure_loaded():
            return None
        pairs = [(query, self.passage(d)) for d in docs]
        try:
            return self._inf.compute_scores_batch(
                self._model, self._tok, pairs,
                max_length=RERANK_MAX_LENGTH,
                device=self._device,
                batch_size=RERANK_BATCH_SIZE,
            )
        except Exception:
            logger.error(f"BGE scoring failed: {traceback.format_exc()}")
            return None
