#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================
# Aggregate per-phase / end-to-end timings across a run folder.
#
#   python3 phase_timing_report.py ./gen_result/<run_folder>
#
# Reads extra.timing from each <md5>.json (written by OrchestratorAgent.search
# + run_spr_agent.py). Prints count, mean, median, p90, min, max, sum per field.
# ==================================================================
import glob
import json
import os
import statistics as st
import sys

PHASES = [
    "phase0_query_expansion",
    "phase1_retrieval_scoring",
    "phase2_reference",
    "phase3_context_query",
    "collect_results",
    "search_total",
    "end_to_end",
]

# Phase 2 sub-step breakdown (extra.timing_phase2_detail), seconds and counts.
DETAIL_FIELDS = [
    "ref_fetch_s", "ref_score_s",
    "n_docs_expanded", "n_refs_fetched_unique",
    "n_refs_pruned", "n_refs_scored", "n_relevant",
]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 phase_timing_report.py <run_folder>")
    run_dir = sys.argv[1]
    files = [f for f in glob.glob(os.path.join(run_dir, "*.json"))]

    series = {p: [] for p in PHASES}
    detail = {k: [] for k in DETAIL_FIELDS}
    n_samples = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fr:
                data = json.load(fr)
        except Exception:
            continue
        t = (data.get("extra") or {}).get("timing")
        if not t:
            continue
        n_samples += 1
        for p in PHASES:
            if p in t and t[p] is not None:
                series[p].append(float(t[p]))
        d = (data.get("extra") or {}).get("timing_phase2_detail")
        if d:
            for k in DETAIL_FIELDS:
                if k in d and d[k] is not None:
                    detail[k].append(float(d[k]))

    if n_samples == 0:
        sys.exit(f"No samples with extra.timing found in {run_dir}\n"
                 "(only NEW runs made after the timing instrumentation have it).")

    def p90(xs):
        s = sorted(xs)
        return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]

    print(f"run: {run_dir}")
    print(f"samples with timing: {n_samples}\n")
    hdr = f"{'phase':28} {'n':>4} {'mean':>8} {'median':>8} {'p90':>8} {'min':>7} {'max':>8} {'sum':>9}"
    print(hdr)
    print("-" * len(hdr))
    for p in PHASES:
        xs = series[p]
        if not xs:
            print(f"{p:28} {0:>4}  (no data)")
            continue
        print(f"{p:28} {len(xs):>4} {st.mean(xs):>8.2f} {st.median(xs):>8.2f} "
              f"{p90(xs):>8.2f} {min(xs):>7.2f} {max(xs):>8.2f} {sum(xs):>9.1f}")

    # share of search_total (mean) per phase — where the time goes
    tot = series.get("search_total") or []
    if tot:
        mean_tot = st.mean(tot)
        print("\nmean share of search_total:")
        for p in ["phase0_query_expansion", "phase1_retrieval_scoring",
                  "phase2_reference", "phase3_context_query", "collect_results"]:
            if series[p]:
                print(f"  {p:28} {100*st.mean(series[p])/mean_tot:5.1f}%")

    # Phase 2 sub-step breakdown (fetch vs LLM-score) + work volume
    if any(detail[k] for k in DETAIL_FIELDS):
        print("\nPhase 2 sub-steps (mean per sample):")
        for k in DETAIL_FIELDS:
            xs = detail[k]
            if xs:
                unit = "s" if k.endswith("_s") else ""
                print(f"  {k:24} {st.mean(xs):>10.2f}{unit}")
        fetch = detail["ref_fetch_s"]
        score = detail["ref_score_s"]
        if fetch and score:
            mf, ms = st.mean(fetch), st.mean(score)
            tot2 = mf + ms or 1
            print(f"  -> fetch {100*mf/tot2:.1f}%  |  score {100*ms/tot2:.1f}%  of (fetch+score)")


if __name__ == "__main__":
    main()
