# !/usr/bin/env python
# -*- coding:utf-8 -*-
# SPECTER2 encoder (shared by build_specter_index.py / mine_hard_negatives.py).
# Papers are encoded as "Title{SEP}Abstract"; queries as the raw question text.
import numpy as np
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

_STATE = [None]


def load(base="allenai/specter2_base", adapter="allenai/specter2"):
    if _STATE[0] is None:
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoAdapterModel.from_pretrained(base)
        model.load_adapter(adapter, source="hf", load_as="proximity", set_active=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev).eval()
        _STATE[0] = (tok, model, dev)
    return _STATE[0]


def paper_text(title: str, abstract: str) -> str:
    tok, _, _ = load()
    return f"{title or ''}{tok.sep_token}{abstract or ''}"


@torch.no_grad()
def embed(texts, batch_size=256, max_length=512, show=False):
    """Return L2-normalized float32 embeddings (N, 768) for a list of strings."""
    tok, model, dev = load()
    out = []
    rng = range(0, len(texts), batch_size)
    if show:
        from tqdm import tqdm
        rng = tqdm(rng, desc="embed")
    for i in rng:
        b = texts[i:i + batch_size]
        inp = tok(b, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(dev)
        e = model(**inp).last_hidden_state[:, 0, :]          # CLS pooling
        e = torch.nn.functional.normalize(e, dim=1)
        out.append(e.cpu().numpy().astype("float32"))
    return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")
