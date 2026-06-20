"""Inspect-AI solver wrapping the SPAR pipeline for astabench/paper_finder_*.

Runs astabench_bridge/spar_query_cli.py as a subprocess in the SPAR conda env
(separate from the asta-bench uv env: SPAR is pinned to Python 3.10 while
astabench needs >=3.11). One subprocess per sample.

IMPORTANT: run inspect with --max-samples 1. SPAR cannot run samples
concurrently (single GPU for the BGE prefilter, arXiv single-connection ToS,
Serper rate/credit limits).

Usage:
    uv run inspect eval astabench/paper_finder_validation \
        -T with_search_tools=false \
        --solver /home/richard/AI/LLM_Academic_agent/SPAR/astabench_bridge/spar_solver.py \
        --model openai/gpt-4o-2024-11-20 --max-samples 1 --log-dir logs/spar-val/

SPAR's per-query logs are written to SPAR_SUB_LOG_DIR (default
<SPAR_DIR>/logs/spar_subprocess/), one file per sample, plus a `latest.log`
symlink — so during an `inspect eval` run you can watch live progress with:
    tail -f <SPAR_DIR>/logs/spar_subprocess/latest.log

Env overrides: SPAR_DIR, SPAR_PYTHON, SPAR_TIMEOUT_S, SPAR_TOPN, SPAR_SUB_LOG_DIR.
"""
import asyncio
import json
import os
import re
import tempfile
import time

from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.solver import Generate, TaskState, solver

SPAR_DIR = os.getenv("SPAR_DIR", "/home/richard/AI/LLM_Academic_agent/SPAR")
SPAR_PYTHON = os.getenv(
    "SPAR_PYTHON", "/home/richard/miniconda3/envs/SPAR/bin/python"
)
CLI = os.path.join(SPAR_DIR, "astabench_bridge", "spar_query_cli.py")
TIMEOUT_S = int(os.getenv("SPAR_TIMEOUT_S", "5400"))  # 90 min per query
TOPN = int(os.getenv("SPAR_TOPN", "50"))
MIN_SIM = os.getenv("SPAR_MIN_SIM")  # e.g. "0.55" -> return A high-rel pool
SUB_LOG_DIR = os.getenv(
    "SPAR_SUB_LOG_DIR", os.path.join(SPAR_DIR, "logs", "spar_subprocess")
)

# The asta-bench side exports HTTP(S)_PROXY for the grader / HF downloads, but
# SPAR must NOT inherit them: it routes DeepSeek/Serper through global_config's
# auto-detected PROXIES explicitly and deliberately hits OpenAlex/arXiv DIRECT
# (api_web passes no proxies= there). Leaking the proxy env forces every
# OpenAlex reference fetch through the Clash proxy -> per-request SSL handshakes
# -> Phase 2 crawls for tens of minutes. Strip them for the subprocess.
_PROXY_ENV_KEYS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


def _spar_env():
    env = {k: v for k, v in os.environ.items() if k not in _PROXY_ENV_KEYS}
    return env


def _litellm_model_name(spar_model: str) -> str:
    """Map SPAR's model label to a litellm-priceable name for cost accounting."""
    m = (spar_model or "").lower()
    if "deepseek" in m:
        return "deepseek/deepseek-chat"
    if "qwen" in m:
        return "openrouter/qwen/qwen-2.5-72b-instruct"
    return spar_model or "deepseek/deepseek-chat"


async def _record_agent_usage(usage: dict) -> None:
    """Emit a ModelEvent carrying SPAR's real token spend so agenteval counts it
    as agent (solver) cost. SPAR's DeepSeek calls bypass inspect's model API, so
    without this the leaderboard would show ~$0 agent cost. Best-effort.
    """
    try:
        it = int(usage.get("input_tokens") or 0)
        ot = int(usage.get("output_tokens") or 0)
        if it == 0 and ot == 0:
            return
        model_name = _litellm_model_name(usage.get("model", ""))
        out = ModelOutput.from_content(model=model_name, content="")
        out.usage = ModelUsage(
            input_tokens=it, output_tokens=ot, total_tokens=it + ot
        )
        m = get_model("mockllm/model", custom_outputs=[out])
        await m.generate(input="record-spar-agent-usage")
    except Exception as e:  # never let cost bookkeeping break the eval
        print(f"[spar_solver] usage record skipped: {e}")


@solver
def spar_solver(topn: int = TOPN):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        query = (state.metadata or {}).get("raw_query") or state.input_text
        end_date = (state.metadata or {}).get("insertion_date") or ""

        fd, out_path = tempfile.mkstemp(suffix=".json", prefix="spar_pf_")
        os.close(fd)

        # Per-sample SPAR log file (+ latest.log pointer) so the run is
        # observable live via `tail -f .../spar_subprocess/latest.log`.
        os.makedirs(SUB_LOG_DIR, exist_ok=True)
        sid = re.sub(r"[^A-Za-z0-9_-]", "_", str(state.sample_id or "sample"))[:40]
        log_path = os.path.join(SUB_LOG_DIR, f"{int(time.time())}_{sid}.log")
        latest = os.path.join(SUB_LOG_DIR, "latest.log")
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(log_path), latest)
        except OSError:
            pass

        try:
            with open(log_path, "wb") as logf:
                cli_args = [
                    SPAR_PYTHON, CLI,
                    "--query", query,
                    "--end-date", end_date,
                    "--topn", str(topn),
                    "--out", out_path,
                ]
                if MIN_SIM is not None:
                    cli_args += ["--min-sim", MIN_SIM]
                proc = await asyncio.create_subprocess_exec(
                    *cli_args,
                    cwd=SPAR_DIR,
                    env=_spar_env(),
                    stdout=logf,
                    stderr=asyncio.subprocess.STDOUT,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=TIMEOUT_S)
                except asyncio.TimeoutError:
                    proc.kill()
                    raise RuntimeError(
                        f"SPAR timed out after {TIMEOUT_S}s: {query[:60]} "
                        f"(see {log_path})"
                    )
            if proc.returncode != 0:
                with open(log_path, "r", encoding="utf-8", errors="replace") as fr:
                    tail = fr.read()[-2000:]
                raise RuntimeError(
                    f"SPAR exited {proc.returncode} (see {log_path}):\n{tail}"
                )
            with open(out_path, encoding="utf-8") as fr:
                out_data = json.load(fr)
            results = out_data.get("results", [])
            usage = out_data.get("usage", {})
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass

        # Surface SPAR's real DeepSeek spend as agent cost (best-effort).
        await _record_agent_usage(usage)

        state.output.completion = json.dumps(
            {"output": {"results": results}}, ensure_ascii=False
        )
        return state

    return solve
