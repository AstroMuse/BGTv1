# SPAR → AstaBench PaperFindingBench bridge

Runs the local SPAR pipeline as an inspect-AI solver for
`astabench/paper_finder_validation` / `_test`, then packages the logs for the
AstaBench leaderboard.

## Layout

- `spar_query_cli.py` — runs SPAR on ONE query (config: `--reference
  --bge-prefilter --phase3`, adapter `lora-specter-hardneg`), maps results to
  Semantic Scholar `corpus_id` via the S2 batch API, writes
  `{"results":[{"paper_id","markdown_evidence"}]}`. Runs in the **SPAR conda
  env** (Python 3.10).
- `spar_solver.py` — inspect solver. Calls the CLI as a subprocess (one per
  sample) and puts the JSON into `state.output.completion` in the format
  `score_paper_finder()` expects.

Two envs on purpose: SPAR is pinned to Python 3.10; astabench needs >=3.11.
The solver bridges them via subprocess, so they never share an interpreter.

## Prerequisites

1. `asta-bench` cloned and synced at `~/AI/LLM_Academic_agent/asta-bench`
   (`uv sync`). `uv` at `~/.local/bin/uv`.
2. **Grader model** (`gpt-4o-2024-11-20`) — instantiated at task-import time to
   judge semantic-query relevance; the task will not load without it. Routed
   through **OpenRouter** by pointing inspect's openai provider at it:
   `OPENAI_BASE_URL=https://openrouter.ai/api/v1` + `OPENAI_API_KEY=sk-or-...`.
   OpenRouter resolves the bare name `gpt-4o-2024-11-20` to
   `openai/gpt-4o-2024-11-20` (verified), so no task-code change is needed. A
   real `OPENAI_API_KEY` works too — just drop the base-url override.
3. **`HF_ACCESS_TOKEN`** — the `allenai/asta-bench` dataset is gated. Accept the
   terms once on the dataset page, then set `HF_ACCESS_TOKEN=hf_...` (a read
   token). The eval calls `load_dataset(...)` which downloads on demand — no
   manual `hf download` needed.
4. SPAR's own keys (DeepSeek, Serper) already in `global_config.py`, plus a
   funded Serper account and a free GPU for the BGE prefilter.

## Run (validation split)

One-shot via the helper (sets all env, runs eval + score):

```bash
export HF_ACCESS_TOKEN=hf_...        # your gated-dataset read token
~/AI/LLM_Academic_agent/SPAR/astabench_bridge/run_paperfinding.sh validation
```

Or manually:

```bash
cd ~/AI/LLM_Academic_agent/asta-bench
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-or-...
export HF_ACCESS_TOKEN=hf_...
export HTTPS_PROXY=http://172.22.48.1:7890 HTTP_PROXY=http://172.22.48.1:7890

~/.local/bin/uv run inspect eval astabench/paper_finder_validation \
    -T with_search_tools=false \
    --solver /home/richard/AI/LLM_Academic_agent/SPAR/astabench_bridge/spar_solver.py \
    --model openai/gpt-4o-2024-11-20 \
    --max-samples 1 \
    --log-dir logs/spar-val/
```

- `-T with_search_tools=false`: SPAR searches on its own; this skips
  astabench's MCP search tools (which our solver ignores anyway).
- `--max-samples 1`: **mandatory.** SPAR can't run samples concurrently
  (single GPU for BGE, arXiv 1-connection ToS, Serper rate/credit limits).
- `--model` here only names the grader; the SPAR pipeline still uses DeepSeek
  internally.

### Agent cost reporting

SPAR's DeepSeek calls bypass inspect's model API, so by default the leaderboard
would show ~$0 agent cost. To fix that the bridge now reports real spend:
`local_request_v2` accumulates prompt/completion tokens from actual DeepSeek
responses (cache hits don't count); `spar_query_cli` writes them into the result
JSON's `usage` field; `spar_solver` replays them as a single `ModelEvent`
(via `mockllm`, tagged `deepseek/deepseek-chat`) so agenteval prices it as
**agent** (solver) cost — not grader cost, which agenteval excludes by design.
Verify it priced after a run: the eval log's solver-span ModelUsage is non-zero.

Per-query knobs (env): `SPAR_TIMEOUT_S` (default 5400), `SPAR_TOPN` (default
50), `SPAR_PYTHON`, `SPAR_DIR`.

## Score + submit

```bash
# local scores + cost
~/.local/bin/uv run astabench score logs/spar-val/

# generate eval_config.json for the leaderboard
~/.local/bin/uv run astabench eval --config-only --log-dir logs/spar-val/ \
    --config-path astabench/config/v1.0.0.yml --split validation --ignore-git

tar czfv spar-paperfinding-val.tar.gz logs/spar-val/
# upload at https://allenai-asta-bench-leaderboard.hf.space/submit
```

Submitting only paper_finder logs is allowed: you appear in the
PaperFindingBench column; the Literature-Understanding aggregate counts the
missing tasks as empty. Use `validation` to iterate; `test` is rate-limited to
one submission / 24h.

## Quick single-query sanity check (SPAR env)

```bash
cd ~/AI/LLM_Academic_agent/SPAR
/home/richard/miniconda3/envs/SPAR/bin/python astabench_bridge/spar_query_cli.py \
    --query "papers using GANs for text-to-image generation" \
    --end-date 2025-06-01 --topn 20 --out /tmp/pf_test.json
cat /tmp/pf_test.json
```
