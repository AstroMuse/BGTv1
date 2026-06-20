#!/usr/bin/env bash
# Run SPAR on AstaBench PaperFindingBench via the bridge solver.
#
# Usage:
#   ./run_paperfinding.sh validation     # iterate (default)
#   ./run_paperfinding.sh test           # leaderboard run (1 submission / 24h)
#
# Run with bash (./run_paperfinding.sh or bash run_paperfinding.sh), NOT `sh`.
# Grader (gpt-4o-2024-11-20) is routed through OpenRouter; the SPAR pipeline
# still uses DeepSeek internally. Edit the keys below or export them first.

# Re-exec under bash if launched via `sh` (dash lacks `set -o pipefail`).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail

SPLIT="${1:-validation}"
ASTA_DIR="${ASTA_DIR:-$HOME/AI/LLM_Academic_agent/asta-bench}"
SPAR_DIR="${SPAR_DIR:-$HOME/AI/LLM_Academic_agent/SPAR}"
UV="${UV:-$HOME/.local/bin/uv}"
PROXY="${PROXY:-http://172.22.48.1:7890}"

# --- Grader via OpenRouter (inspect's openai provider, base-url override) ---
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
# litellm bundled price table (online fetch fails behind the proxy).
export LITELLM_LOCAL_MODEL_COST_MAP="True"

# --- HuggingFace gated dataset token (read scope) ---
# HF_ACCESS_TOKEN: task's load_dataset(). HF_TOKEN: the scorer's hf_hub_download()
# for normalizer_reference.json (it passes no token, so it needs HF_TOKEN). Set both.
export HF_ACCESS_TOKEN="${HF_ACCESS_TOKEN:-}"
export HF_TOKEN="${HF_TOKEN:-$HF_ACCESS_TOKEN}"
if [[ -z "${HF_ACCESS_TOKEN}" ]]; then
  echo "WARNING: HF_ACCESS_TOKEN is empty; gated dataset load + scoring will fail." >&2
  echo "  export HF_ACCESS_TOKEN=hf_xxx  (read token from your HF account)" >&2
fi

# --- Proxy for the uv-env side (grader + HF download); WSL2 NAT needs it ---
export HTTPS_PROXY="$PROXY"
export HTTP_PROXY="$PROXY"

# --- Bridge knobs (consumed by spar_solver.py / spar_query_cli.py) ---
export SPAR_DIR
export SPAR_PYTHON="${SPAR_PYTHON:-$HOME/miniconda3/envs/SPAR/bin/python}"
export BGE_LORA_PATH="${BGE_LORA_PATH:-./bge/output/lora-specter-hardneg/best_model}"

LOG_DIR="${LOG_DIR:-logs/spar-${SPLIT}/}"

# Optional: LIMIT=N runs only the first N samples (validation is semantic-first,
# so LIMIT=10 == the first 10 natural-language queries). SPAR_TOPN / SPAR_MIN_SIM
# are read by spar_solver.py from the environment.
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then LIMIT_ARGS=(--limit "${LIMIT}"); fi

cd "$ASTA_DIR"
echo "Running paper_finder_${SPLIT} -> ${LOG_DIR}  (LIMIT=${LIMIT:-all}, SPAR_TOPN=${SPAR_TOPN:-50}, SPAR_MIN_SIM=${SPAR_MIN_SIM:-none})"
"$UV" run inspect eval "astabench/paper_finder_${SPLIT}" \
  -T with_search_tools=false \
  --solver "${SPAR_DIR}/astabench_bridge/spar_solver.py" \
  --model openai/gpt-4o-2024-11-20 \
  --max-samples 1 "${LIMIT_ARGS[@]}" \
  --log-dir "$LOG_DIR"

# eval_config.json is required by `astabench score` and by leaderboard submission.
echo "--- generating eval_config.json ---"
"$UV" run astabench eval --config-only --log-dir "$LOG_DIR" \
  --config-path astabench/config/v1.0.0.yml --split "$SPLIT" --ignore-git

echo "--- scoring ---"
"$UV" run astabench score "$LOG_DIR"
