#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] : Global configuration settings for Scholar Paper Agent Retrieval
# ==================================================================
import os
import json
import arxiv
from typing import Dict, List, Any

# Load secrets from a local .env (never committed) into os.environ, so the keys
# below resolve via os.getenv. No-op if python-dotenv or .env is absent — keys
# can also come straight from real environment variables.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Debug mode
DEBUG = False

# =============================================================================
# API CREDENTIALS
# =============================================================================
# DashScope (Qwen3-32B via Aliyun)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_DEPLOYMENT = "qwen3-32b"

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
# Dedicated key for Agent B's thinking-mode model (deepseek-v4-pro). Kept separate
# from the shared DEEPSEEK_API_KEY so Agent B's reasoning calls bill independently.
DEEPSEEK_API_KEY_AGENT_B = os.getenv("DEEPSEEK_API_KEY_AGENT_B", "")

# Legacy aliases kept for backward compatibility
API_KEY = DEEPSEEK_API_KEY
ENDPOINT = DEEPSEEK_ENDPOINT
DEPLOYMENT_NAME = "deepseek-v4-flash"

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================
SAVE_ID2DOCS = True
RELEVANCE_SCORE = 0.5
WEB_RETRY_NUM = 1

# Query threshold settings
QUERY_LOW_THRESHOLD = 0.2
QUERY_HIGH_THRESHOLD = 0.8
CORRECT_SCORE_THRESHOLD = 0.8
EXPAND_SCORE_THRESHOLD = 0.85
QUERY_TO_SEARCH_THRESHOLD = 0.85

# Generation settings
LENGTH_GEN_QUERY_FROM_CITATION = 12288

# =============================================================================
# WEB API CONFIGURATION
# =============================================================================
TRY_COUNT = 4
LLM_TRY_COUNT = 4
LLM_PARALLEL_NUM = 4
LLM_MODEL_NAME = os.getenv("SPAR_MODEL", "DeepSeek-V4")


API_TRY_COUNT = 4
API_PARALLEL_REQUEST = 1

SLEEP_TIME_LLM = 2.0

# =============================================================================
# SEARCH HYPERPARAMETERS
# =============================================================================
DO_FUSION_JUDGE = True
FUSION_TEMPLATE = "AUTOMATIC"  # Options: "WITHEXPLAIN", "AUTOMATIC"

# Query processing settings
QUERY_NUM_PRUNED = 2  # Number of queries to use for search
RETRIEVAL_QUERY_BATCH_SIZE = 6  # Batch size for query processing to avoid excessive searching

# Document processing settings
DOCS_TO_EXPAND = 40
REFERENCE_DOC_PRUNED = 20  # Number of references to extract from each relevant document
REFERENCE_OCCUR_FREQUENCY = 0.6
REFERENCE_DOC_NUM_TO_GEN_NEW_QUERY = 2  # Number of reference docs used to generate new queries

# Similarity thresholds
REFERENCE_DOC_SIM_THRESHOLD = 0.6
BEGIN_SIM_THRESHOLD = 0.5
PASS_SIM_THRESHOLD = 0.5

# Search routes configuration
# SEARCH_ROUTES: List[str] = ["arxiv", "openalex","semantic"]
SEARCH_ROUTES: List[str] = ["arxiv", "openalex"]

# =============================================================================
# EXTERNAL API KEYS
# =============================================================================
# Register at: https://google.serper.dev/search
GOOGLE_SERPER_KEY = os.getenv("GOOGLE_SERPER_KEY", "")

# Semantic Scholar API key (https://www.semanticscholar.org/product/api)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
# =============================================================================
# SEARCH FEATURES
# =============================================================================
# Toggle reference-based search (Phase 2). Off by default; enable per-run with
# run_spr_agent.py --reference (sets SPAR_DO_REFERENCE).
DO_REFERENCE_SEARCH = os.getenv("SPAR_DO_REFERENCE", "").lower() in ("1", "true", "yes", "on")

# Toggle Phase 3 (context-driven query generation -> deeper BFS levels). Off by
# default: the tree stops after depth 1; enable per-run with run_spr_agent.py
# --phase3 (sets SPAR_DO_PHASE3).
DO_PHASE3 = os.getenv("SPAR_DO_PHASE3", "").lower() in ("1", "true", "yes", "on")

# Early-stop toggle. ON (default) = current behaviour: once depth-1 has more than
# max_docs high-relevance (sim>0.75) docs, the BFS stops and Phase 2 reference
# expansion is SKIPPED. OFF (--no-early-stop): never stop on the high-rel count
# (only depth still bounds the tree), so Phase 2 always runs at depth 1. When OFF
# and high-rel > max_docs, Phase 2 expands only the top-max_docs high-rel docs;
# when high-rel <= max_docs it expands the original DOCS_TO_EXPAND set.
ENABLE_EARLY_STOP = os.getenv("SPAR_EARLY_STOP", "1").lower() not in ("0", "false", "no", "off")

# Which API supplies a paper's reference list in Phase 2: "openalex" (default,
# free / no per-second throttle / abstracts included) or "s2" (Semantic Scholar,
# 1 RPS serialized) or "local" (offline arXiv citation LMDB built by
# build_citation_store.py; ref ids from the local graph, metadata from
# recovered.db, zero latency / no throttle, falls back to s2 then openalex on a
# miss). Selectable per-run with run_spr_agent.py --ref-api.
REFERENCE_API = os.getenv("SPAR_REF_API", "openalex").lower()
# Path to the local forward-citation LMDB (used when REFERENCE_API == "local").
CITATION_STORE_PATH = os.getenv("SPAR_CITATION_STORE", "./database/citations.lmdb")

# Phase 2 reference pruning — caps how many references get LLM-scored per query
# (the dominant Phase 2 cost). Selectable with run_spr_agent.py --ref-prune.
#   "freq"     : aggregate refs across all expanded docs, dedup, rank by
#                co-occurrence count (then citationCount), keep global top-N (150).
#   "citation" : dedup only, rank by citationCount desc, keep global top-N (200).
REFERENCE_PRUNE_STRATEGY = os.getenv("SPAR_REF_PRUNE", "freq").lower()  # freq | citation
# 0 -> use the per-strategy default (freq=150, citation=200); >0 overrides both.
REFERENCE_PRUNE_TOPN = int(os.getenv("SPAR_REF_TOPN", "0"))

RERANK = os.getenv("DO_RERANK", "true").lower() not in ("0", "false", "no", "off")

# Idea 1: multi-level domain tree (build broad -> prune to query). Costs 2 extra
# LLM calls per query; set False to skip entirely.
BUILD_DOMAIN_TREE = True
DOMAIN_TREE_DEPTH = 2  # levels below root: L1(3 nodes) + L2(9 nodes) = 13 total

KEY_WORDS_NUM =3
# Concurrent LLM requests within a single ScoringAgent.score() call (batch chunks
# / Phase-2 reference pool). DeepSeek tolerates several concurrent requests.
LLM_PARREL_NUM = int(os.getenv("SPAR_LLM_PARALLEL", "6"))
# Concurrent query-nodes scored in parallel in Phase 1 (each node ≈ one batched
# LLM call). The dominant Phase-1 speedup: the ~6 expanded queries are scored
# concurrently instead of one-after-another. Total in-flight LLM requests stay
# bounded because each ≤50-doc node is a single call.
PHASE1_NODE_PARALLEL = int(os.getenv("SPAR_PHASE1_PARALLEL", "6"))
# =============================================================================
# NETWORK CONFIGURATION
# =============================================================================
# Proxy port exposed by the Clash "Allow LAN" listener on the Windows host.
PROXY_PORT = os.getenv("PROXY_PORT", "7890")


def _default_proxy() -> str:
    """Build the proxy URL for the Windows host.

    Under WSL2 NAT mode the Windows host is reachable at the default gateway
    (not localhost), and that IP can change across WSL restarts, so detect it
    at runtime. Falls back to localhost when not running under WSL2 NAT.
    """
    try:
        import subprocess

        out = subprocess.check_output(["ip", "route", "show", "default"], text=True)
        gateway = out.split()[2]  # "default via <gateway> dev ..."
        return f"http://{gateway}:{PROXY_PORT}"
    except Exception:
        return f"http://localhost:{PROXY_PORT}"


# HTTP_PROXY / HTTPS_PROXY env vars override the autodetected host gateway.
_proxy = os.getenv("HTTP_PROXY") or _default_proxy()
PROXIES: Dict[str, str] = {
    "http": _proxy,
    "https": os.getenv("HTTPS_PROXY") or _proxy,
}

# ArXiv client configuration.
# arXiv API ToS: <=1 request / 3s on a single connection. delay_seconds is the
# library's built-in throttle between page requests; num_retries handles 429s
# with backoff. Do NOT lower delay_seconds below 3.0 or fan out concurrently.
ARXIV_CLIENT = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)

# =============================================================================
# RERANKING CONFIGURATION
# =============================================================================
# BGE cross-encoder rerank of B_raw after collection. Enable per-run with
# run_spr_agent.py --rerank (sets SPAR_RERANK). Requires torch/transformers/peft
# (the bge/ project env; the default SPAR env may lack peft).
ENABLE_RERANK = os.getenv("SPAR_RERANK", "").lower() in ("1", "true", "yes", "on")
RERANK_MODEL = "Qwen3-32B"  # used only by the legacy LLM Reranker in rerank.py
BGE_BASE_MODEL_PATH = os.getenv("BGE_BASE_MODEL_PATH", "./bge/bge-reranker-v2-m3")
BGE_LORA_PATH = os.getenv("BGE_LORA_PATH", "./bge/output/lora-natural/best_model")
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "32"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "512"))
# Blend with citation authority: final = (1-w)*bge_norm + w*cite_norm. 0 = pure bge.
RERANK_CITATION_WEIGHT = float(os.getenv("RERANK_CITATION_WEIGHT", "0.0"))
# Force the reranker device: "cpu" / "cuda" / "" (auto). Use "cpu" if the GPU is
# busy or CUDA is flaky under WSL (spurious "out of memory" on tiny allocs).
RERANK_DEVICE = os.getenv("SPAR_RERANK_DEVICE", "") or None

# =============================================================================
# BGE prefilter: "coarse filter (BGE cross-encoder) -> fine rank (LLM)" cascade
# for the relevance scoring step (Phase 1 retrieval scoring + Phase 2 reference
# scoring, both via ScoringAgent.score). Default OFF -> keeps the original
# LLM-scores-every-doc behavior. When ON, each scoring batch is first ranked by
# the fine-tuned bge-reranker; only the top-K survive to LLM scoring, the rest
# are dropped (kept out of B_raw, with their bge score recorded for provenance).
# Reuses BGE_BASE_MODEL_PATH / BGE_LORA_PATH / RERANK_DEVICE above. Set per run
# with run_spr_agent.py --bge-prefilter [--bge-prefilter-topk N].
# Degrades gracefully: if the reranker can't load, falls back to full LLM scoring.
# LLM fine-rank prompt: "intent" (default; paper-core-intent vs query-intent
# alignment, weighted 0.5/0.3/0.2) or "rubric" (legacy topic/contextual/depth
# relevance, equal-weight mean). Selectable per-run with --score-prompt.
SCORING_PROMPT = os.getenv("SPAR_SCORING_PROMPT", "intent").lower()

BGE_PREFILTER = os.getenv("SPAR_BGE_PREFILTER", "").lower() in ("1", "true", "yes", "on")
# Per-scoring-batch cap kept for LLM fine scoring. Keep this generous (>= where
# gold typically sits in the bge ranking) so coarse filtering doesn't drop recall.
BGE_PREFILTER_TOPK = int(os.getenv("SPAR_BGE_PREFILTER_TOPK", "40"))

# =============================================================================
# Batched LLM relevance scoring (default path; replaces the BGE coarse filter)
# =============================================================================
# Score up to SCORE_BATCH_SIZE documents against the query in ONE LLM call
# (query_intent stated once; per-paper core_intent + 3 intent-alignment scores)
# instead of one call per doc. Index-anchored + per-doc fallback re-score, so a
# batch parse failure never silently drops a doc to sim_score=-1. Set to 1 to
# fall back to the legacy per-doc scoring. Only the "intent" SCORING_PROMPT is
# batched; "rubric" always uses the per-doc path. Override per run with
# run_spr_agent.py --score-batch-size N (env SPAR_SCORE_BATCH_SIZE).
SCORE_BATCH_SIZE = int(os.getenv("SPAR_SCORE_BATCH_SIZE", "50"))

# =============================================================================
# Parallel candidate agents (Agent B / Agent C) — alternative recall routes that
# run BESIDE the main retrieval at tree depth 1 and inject their candidate docs
# into the SAME Phase-1 scoring pool (B_raw). Both default OFF; enable per-run
# with run_spr_agent.py --agent-b / --agent-c.
# =============================================================================
# Agent B — LLM "memory" recall + existence verification: ask the LLM for the
# papers it believes best match the query (title + arXiv id), verify each id is a
# REAL paper (recovered.db first, then Google Serper), then a second LLM check
# confirms the resolved metadata matches the recommendation. Survivors join B_raw.
AGENT_B_ENABLED = os.getenv("SPAR_AGENT_B", "").lower() in ("1", "true", "yes", "on")
AGENT_B_RECALL_NUM = int(os.getenv("SPAR_AGENT_B_RECALL_NUM", "10"))
# Agent B uses a dedicated thinking-mode model (deepseek-v4-pro, reasoning_effort
# high, no temperature — sampling params are unsupported in thinking mode) so its
# "recall from memory" benefits from deep reasoning. See MODEL_CONFIGS entry of
# the same name in local_request_v2.py. Override per-run with SPAR_AGENT_B_MODEL.
AGENT_B_MODEL = os.getenv("SPAR_AGENT_B_MODEL", "DeepSeek-V4-Pro-AgentB")
# Round-2 existence verification only checks "is this a real paper that matches?"
# — a cheap true/false judgement that does not need reasoning. Use the fast
# non-thinking flash model so verification adds negligible latency.
AGENT_B_VERIFY_MODEL = os.getenv("SPAR_AGENT_B_VERIFY_MODEL", "DeepSeek-V4")

# Agent C — multi-dimensional intent-constraint decomposition: LLM decomposes the
# query into hard/soft constraints (time/domain/lit-type/keywords/+extensions),
# emits parametrizable hard constraints + a rewritten NL query, then searches
# Semantic Scholar + Google(arXiv) + OpenAlex, drops low-impact papers, and merges
# the survivors (E_raw) into B_raw (skipping ids already present).
AGENT_C_ENABLED = os.getenv("SPAR_AGENT_C", "").lower() in ("1", "true", "yes", "on")
# Drop a candidate whose citation count is KNOWN and below this threshold.
# Papers with no citation info (typically arXiv-resolved) are kept — they cannot
# be judged low-impact, and dropping them would nullify the arXiv source.
AGENT_C_MIN_CITATION = int(os.getenv("SPAR_AGENT_C_MIN_CITATION", "20"))

# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================
def validate_config() -> bool:
    """
    Validate essential configuration settings.

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_keys = [API_KEY, ENDPOINT]

    if not all(key and key != "your_openai_api_key_here" for key in required_keys):
        print("Warning: OpenAI API configuration is incomplete")
        return False

    if QUERY_LOW_THRESHOLD >= QUERY_HIGH_THRESHOLD:
        print("Error: QUERY_LOW_THRESHOLD must be less than QUERY_HIGH_THRESHOLD")
        return False

    return True

# Validate configuration on import
if __name__ == "__main__":
    if validate_config():
        print("Configuration validation passed")
