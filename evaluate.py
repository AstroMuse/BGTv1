#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# Evaluate SPAR retrieval results saved under gen_result/<run>/*.json.
#
# Gold answers come from the benchmark file (matched to each result by the
# query string), which carries BOTH gold titles (`answer`) and gold arxiv ids
# (`answer_arxiv_id`). Results are scored against two retrieval sets:
#
#   Set A = high-relevance docs  -> extra.hight_relevance_docs (sim filtered)
#   Set B = raw searched docs    -> extra.searched_docs        (all recalled)
#   Set S = LLM-scored pool      -> searched_docs with sim_score >= 0
#           (in a --bge-prefilter run this is exactly the coarse-filter
#            survivors; gold outside S can never reach A or top-k)
#
# Three matching modes are reported side by side:
#   title_exact : keep_letters-normalized full title, exact equality
#   title_loose : normalized containment (either direction, len>=12) OR
#                 word-token Jaccard >= 0.7
#   arxiv_id    : base arxiv id (\d{4}\.\d+, version stripped), exact equality
#
# Per query: Recall / Precision / F1. Reported as macro-average over queries
# that have a gold answer for that mode.
#
# Usage:
#   python3 evaluate.py                       # all run dirs under gen_result/
#   python3 evaluate.py <run_dir>             # one run dir
#   python3 evaluate.py <run_dir> --verbose   # also print per-query (arxiv mode)
# ==================================================================
import glob
import json
import math
import os
import re
import sys

BENCHMARK_FILES = [
    "benchmark/AutoScholarQuery_test.jsonl",
    "benchmark/spar_bench.jsonl",
]

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
TOKEN_RE = re.compile(r"[a-z0-9]+")


def keep_letters(s):
    """Normalize a title: keep only letters, lowercase. Matches utils.keep_letters."""
    return "".join(c for c in str(s) if c.isalpha()).lower()


def norm_arxiv(x):
    """Extract the base arxiv id (drop version/prefix); None if not arxiv-like."""
    m = ARXIV_RE.search(str(x))
    return m.group(1) if m else None


def tokens(t):
    return set(TOKEN_RE.findall(str(t).lower()))


def loose_match(g, r):
    """Loose title match between raw titles g (gold) and r (retrieved)."""
    ng, nr = keep_letters(g), keep_letters(r)
    if not ng or not nr:
        return False
    if ng == nr:
        return True
    if len(ng) >= 12 and (ng in nr or nr in ng):
        return True
    tg, tr = tokens(g), tokens(r)
    if tg and tr and len(tg & tr) / len(tg | tr) >= 0.7:
        return True
    return False


# ----------------------------------------------------------------------------
# Gold lookup from the benchmark files: question -> {titles:[...], ids:set()}
# ----------------------------------------------------------------------------
def load_gold():
    gold = {}
    for path in BENCHMARK_FILES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                q = d.get("question") or d.get("query")
                if not q:
                    continue
                titles = d.get("answer", []) or []
                if isinstance(titles, str):
                    titles = [titles]
                ids = d.get("answer_arxiv_id", []) or []
                if isinstance(ids, str):
                    ids = [ids]
                gold[q] = {
                    "titles": [t for t in titles if t],
                    "ids": {norm_arxiv(i) for i in ids if norm_arxiv(i)},
                }
    return gold


# ----------------------------------------------------------------------------
# Retrieval sets from one result JSON
# ----------------------------------------------------------------------------
def retrieval_sets(extra):
    searched = extra.get("searched_docs", {})
    # Set B: all recalled docs (B-raw)
    b_titles = [d.get("title", "") for d in searched.values()]
    b_ids = {norm_arxiv(pid) for pid in searched.keys() if norm_arxiv(pid)}
    b_count = len(searched)
    # Set A: high-relevance ids resolved against searched_docs (sim_score > 0.75)
    a_ids_raw = extra.get("hight_relevance_docs", [])
    a_titles = []
    for pid in a_ids_raw:
        doc = searched.get(pid)
        if doc:
            a_titles.append(doc.get("title", ""))
    a_ids = {norm_arxiv(pid) for pid in a_ids_raw if norm_arxiv(pid)}
    a_count = len(a_ids_raw)
    # Set S: docs that actually received an LLM sim_score (>= 0). With
    # --bge-prefilter these are the coarse-filter survivors (dropped docs
    # keep sim_score = -1 and stay in B only).
    s_titles = [d.get("title", "") for d in searched.values()
                if d.get("sim_score", -1) >= 0]
    s_ids = {norm_arxiv(pid) for pid, d in searched.items()
             if d.get("sim_score", -1) >= 0 and norm_arxiv(pid)}
    s_count = sum(1 for d in searched.values() if d.get("sim_score", -1) >= 0)
    # Set C: collected top-k (extra.ahp_topk; AHP or cite-gate+sim depending on run)
    ahp_topk = extra.get("ahp_topk", {})
    c_titles = [d.get("title", "") for d in ahp_topk.values()]
    c_ids = {norm_arxiv(pid) for pid in ahp_topk.keys() if norm_arxiv(pid)}
    c_count = len(ahp_topk)
    return {
        "A": {"titles": a_titles, "ids": a_ids, "count": a_count},
        "B": {"titles": b_titles, "ids": b_ids, "count": b_count},
        "S": {"titles": s_titles, "ids": s_ids, "count": s_count},
        "C": {"titles": c_titles, "ids": c_ids, "count": c_count},
    }


def prf(n_matched, n_gold, n_retrieved):
    recall = n_matched / n_gold if n_gold else 0.0
    precision = n_matched / n_retrieved if n_retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return recall, precision, f1


def rank_key(doc):
    """Ranking signal for B_raw: BGE rerank score if present, else AHP, else sim_score."""
    for k in ("bge_score", "ahp_score", "sim_score"):
        v = doc.get(k)
        if v is not None:
            return v
    return -1


def ranked_topk(extra, k):
    """Top-k docs from searched_docs ranked by rank_key, as a retr dict {ids,titles,count}."""
    searched = extra.get("searched_docs", {})
    ranked = sorted(searched.values(), key=rank_key, reverse=True)[: max(k, 0)]
    return {
        "ids": {norm_arxiv(d.get("paper_id", d.get("arxivId", ""))) for d in ranked
                if norm_arxiv(d.get("paper_id", d.get("arxivId", "")))},
        "titles": [d.get("title", "") for d in ranked],
        "count": len(ranked),
    }


FUNNEL_KS = (5, 10, 40)

# Set-A threshold sweep: recompute A(t) = {scored docs with sim_score > t}
# directly from searched_docs (the stored hight_relevance_docs is the t=0.75
# snapshot the orchestrator wrote at run time).
A_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75)


def sim_thresh_set(extra, t):
    """Docs with sim_score > t, as a retr dict {ids, titles, count}."""
    searched = extra.get("searched_docs", {})
    sel = {pid: d for pid, d in searched.items() if (d.get("sim_score") or -1) > t}
    return {
        "ids": {norm_arxiv(pid) for pid in sel.keys() if norm_arxiv(pid)},
        "titles": [d.get("title", "") for d in sel.values()],
        "count": len(sel),
    }


def sim_ranked_topk(extra, k):
    """Top-k docs from searched_docs ranked by sim_score only (the LLM ordering)."""
    searched = extra.get("searched_docs", {})
    ranked = sorted(
        searched.values(), key=lambda d: d.get("sim_score") or -1, reverse=True
    )[: max(k, 0)]
    return {
        "ids": {norm_arxiv(d.get("paper_id", d.get("arxivId", ""))) for d in ranked
                if norm_arxiv(d.get("paper_id", d.get("arxivId", "")))},
        "titles": [d.get("title", "") for d in ranked],
        "count": len(ranked),
    }


def ndcg_at_k(mode, gold, extra, k):
    """Binary nDCG@k over the sim_score ranking (AstaBench-style).

    Gain 1 the first time a gold item is matched (a gold already matched by an
    earlier position contributes nothing again — mirrors the set semantics of
    count_matches). IDCG places min(n_gold, k) hits at the top.
    """
    searched = extra.get("searched_docs", {})
    ranked = sorted(
        searched.values(), key=lambda d: d.get("sim_score") or -1, reverse=True
    )[: max(k, 0)]

    if mode == "arxiv_id":
        g = set(gold["ids"])
        if not g:
            return None
        n_gold = len(g)

        def take(doc):
            na = norm_arxiv(doc.get("paper_id", doc.get("arxivId", "")))
            if na and na in g:
                g.remove(na)
                return True
            return False
    elif mode == "title_exact":
        g = {keep_letters(t) for t in gold["titles"] if keep_letters(t)}
        if not g:
            return None
        n_gold = len(g)

        def take(doc):
            nt = keep_letters(doc.get("title", ""))
            if nt in g:
                g.remove(nt)
                return True
            return False
    else:  # title_loose
        remaining = [t for t in gold["titles"] if keep_letters(t)]
        if not remaining:
            return None
        n_gold = len({keep_letters(t) for t in remaining})

        def take(doc):
            rt = doc.get("title", "")
            for i, gt in enumerate(remaining):
                if loose_match(gt, rt):
                    remaining.pop(i)
                    return True
            return False

    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked) if take(d))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(n_gold, k)))
    return dcg / idcg if idcg else None


def gold_size(mode, gold):
    """Number of gold items for a mode (denominator of recall), or None if no gold."""
    if mode == "arxiv_id":
        return len(gold["ids"]) or None
    g_norm = {keep_letters(t) for t in gold["titles"] if keep_letters(t)}
    return len(g_norm) or None


def resolve_k(k_arg, n_gold):
    """k_arg is 'ngold2' (-> n_gold+2) or an int string (fixed cutoff)."""
    return n_gold + 2 if k_arg == "ngold2" else int(k_arg)


def count_matches(mode, gold, retr):
    """Return (n_matched, n_gold) for a given matching mode, or None if no gold."""
    if mode == "arxiv_id":
        g = gold["ids"]
        if not g:
            return None
        return len(g & retr["ids"]), len(g)

    # title modes need normalized gold titles
    gold_titles = gold["titles"]
    g_norm = {keep_letters(t) for t in gold_titles if keep_letters(t)}
    if not g_norm:
        return None

    if mode == "title_exact":
        r_norm = {keep_letters(t) for t in retr["titles"] if keep_letters(t)}
        return len(g_norm & r_norm), len(g_norm)

    if mode == "title_loose":
        matched = sum(
            1 for gt in gold_titles if any(loose_match(gt, rt) for rt in retr["titles"])
        )
        return matched, len(g_norm)

    raise ValueError(mode)


MODES = ["title_exact", "title_loose", "arxiv_id"]


def evaluate_run(run_dir, gold_map, k_arg="ngold2"):
    files = sorted(glob.glob(os.path.join(run_dir, "*.json")))
    # results[mode][set] = list of (recall, precision, f1)
    results = {m: {"A": [], "B": [], "S": [], "C": []} for m in MODES}
    # ranked[mode] = {"cut": [(r,p,f)...] at k=resolve_k, "rprec": [float...] at k=n_gold}
    ranked = {m: {"cut": [], "rprec": []} for m in MODES}
    # funnel[mode][k] = [(r, p, f1, nP, nDCG)...] at cutoff k, sim_score ranking;
    # nP@k = matched / min(k, n_gold) (per-query ceiling-normalized P@k)
    funnel = {m: {k: [] for k in FUNNEL_KS} for m in MODES}
    # asweep[mode][t] = [(r, p, f1)...] for A(t) = sim_score > t
    asweep = {m: {t: [] for t in A_THRESHOLDS} for m in MODES}
    asweep_counts = {t: [] for t in A_THRESHOLDS}  # |A(t)| per query
    set_counts = {"A": [], "B": [], "S": [], "C": []}  # pool sizes per query
    ks_used = []  # actual k applied (for reporting mean k)
    per_query = []  # for verbose arxiv view
    matched_gold = unmatched_gold = 0
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fr:
                data = json.load(fr)
        except Exception:
            continue
        if not isinstance(data, dict) or "search_query" not in data:
            continue  # skip non-result JSON (e.g. select files, lists)
        query = data.get("search_query", "")
        extra = data.get("extra", {})
        gold = gold_map.get(query)
        if gold is None:
            # fall back to the titles stored inside the result JSON (no ids)
            ans = extra.get("answer", [])
            if isinstance(ans, str):
                ans = [ans]
            gold = {"titles": [t for t in ans if t], "ids": set()}
            unmatched_gold += 1
        else:
            matched_gold += 1
        sets = retrieval_sets(extra)
        for sk in set_counts:
            set_counts[sk].append(sets[sk]["count"])
        sets_by_t = {t: sim_thresh_set(extra, t) for t in A_THRESHOLDS}
        for t in A_THRESHOLDS:
            asweep_counts[t].append(sets_by_t[t]["count"])
        row = {"query": query}
        for mode in MODES:
            for sk in ("A", "B", "S", "C"):
                cm = count_matches(mode, gold, sets[sk])
                if cm is None:
                    continue
                n_matched, n_gold = cm
                r, p, f = prf(n_matched, n_gold, sets[sk]["count"])
                results[mode][sk].append((r, p, f))
                if mode == "arxiv_id":
                    row[sk] = (r, p, f)

            # --- Set-A threshold sweep ---
            for t in A_THRESHOLDS:
                cm_t = count_matches(mode, gold, sets_by_t[t])
                if cm_t is not None:
                    n_m, n_g = cm_t
                    asweep[mode][t].append(prf(n_m, n_g, sets_by_t[t]["count"]))

            # --- ranked B_raw cutoff metrics (P@k / R@k / F1@k) ---
            ng = gold_size(mode, gold)
            if not ng:
                continue
            k = resolve_k(k_arg, ng)
            cut = ranked_topk(extra, k)
            cm = count_matches(mode, gold, cut)
            if cm is not None:
                m_cut, _ = cm
                r, p, f = prf(m_cut, ng, cut["count"])
                ranked[mode]["cut"].append((r, p, f))
                if mode == "arxiv_id":
                    ks_used.append(k)
            # R-precision: cutoff at k=n_gold (P==R==matched/n_gold)
            rprec_set = ranked_topk(extra, ng)
            cm_r = count_matches(mode, gold, rprec_set)
            if cm_r is not None:
                ranked[mode]["rprec"].append(cm_r[0] / ng if ng else 0.0)
            # --- funnel: R/P/F1/nP/nDCG@k ranked purely by sim_score ---
            for fk in FUNNEL_KS:
                cut_f = sim_ranked_topk(extra, fk)
                cm_f = count_matches(mode, gold, cut_f)
                if cm_f is not None:
                    m_f, _ = cm_f
                    r, p, f = prf(m_f, ng, cut_f["count"])
                    np_k = m_f / min(fk, ng)
                    nd = ndcg_at_k(mode, gold, extra, fk)
                    funnel[mode][fk].append((r, p, f, np_k, nd or 0.0))
        per_query.append(row)
    return (results, ranked, funnel, asweep, asweep_counts, set_counts, ks_used,
            per_query, len(files), matched_gold, unmatched_gold)


def macro(triples):
    if not triples:
        return (0.0, 0.0, 0.0, 0)
    n = len(triples)
    return (
        sum(t[0] for t in triples) / n,
        sum(t[1] for t in triples) / n,
        sum(t[2] for t in triples) / n,
        n,
    )


def print_report(run_dir, results, ranked, funnel, asweep, asweep_counts,
                 set_counts, ks_used, k_arg, per_query, total, matched_gold,
                 unmatched_gold, verbose):
    print("=" * 84)
    print(f"RUN: {run_dir}")
    print(
        f"json files: {total} | gold via benchmark: {matched_gold} | "
        f"gold via result-file fallback (no ids): {unmatched_gold}"
    )
    pool = {sk: (sum(v) / len(v) if v else 0.0) for sk, v in set_counts.items()}
    print(
        f"mean pool size: B raw {pool['B']:.0f} | S scored {pool['S']:.0f} | "
        f"A high-rel {pool['A']:.1f} | C topk {pool['C']:.1f}"
    )

    if verbose:
        print("-" * 84)
        print("Per-query (arxiv_id mode)  R/P/F1")
        print(f"{'query':50.50} {'A (high-rel)':>14} {'B (raw)':>14} {'C (AHP-topk)':>14}")
        for row in per_query:
            a = row.get("A")
            b = row.get("B")
            c = row.get("C")
            a_s = f"{a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f}" if a else "  (no gold id)"
            b_s = f"{b[0]:.2f}/{b[1]:.2f}/{b[2]:.2f}" if b else "  (no gold id)"
            c_s = f"{c[0]:.2f}/{c[1]:.2f}/{c[2]:.2f}" if c else "  (no gold id)"
            print(f"{row['query']:50.50} {a_s:>14} {b_s:>14} {c_s:>14}")

    print("-" * 84)
    print(f"{'mode':14} {'set':18} {'Recall':>8} {'Precision':>10} {'F1':>8} {'#q':>5}")
    for mode in MODES:
        for sk, label in (("A", "A high-rel"), ("B", "B raw"),
                          ("S", "S scored-pool"), ("C", "C topk")):
            r, p, f, n = macro(results[mode][sk])
            print(f"{mode:14} {label:18} {r:>8.4f} {p:>10.4f} {f:>8.4f} {n:>5}")
        print()

    # --- Set-A threshold sweep ---
    print("-" * 84)
    print("A high-rel threshold sweep (A(t) = scored docs with sim_score > t; "
          "stored A row uses t=0.75)")
    print(f"{'mode':14} {'t':>5} {'Recall':>8} {'Precision':>10} {'F1':>8} "
          f"{'mean|A|':>8} {'#q':>5}")
    for mode in MODES:
        for t in A_THRESHOLDS:
            r, p, f, n = macro(asweep[mode][t])
            cnts = asweep_counts[t]
            mean_a = sum(cnts) / len(cnts) if cnts else 0.0
            print(f"{mode:14} {t:>5.2f} {r:>8.4f} {p:>10.4f} {f:>8.4f} "
                  f"{mean_a:>8.1f} {n:>5}")
        print()

    # --- Ranked B_raw cutoff metrics ---
    k_label = "n_gold+2" if k_arg == "ngold2" else f"k={k_arg}"
    mean_k = (sum(ks_used) / len(ks_used)) if ks_used else 0
    print("-" * 84)
    print(f"Ranked B_raw @ cutoff (rank by ahp_score->sim_score; cutoff = {k_label}"
          + (f", mean k={mean_k:.1f}" if k_arg == 'ngold2' else "") + ")")
    print(f"{'mode':14} {'P@k':>8} {'R@k':>8} {'F1@k':>8} {'R-prec':>8} {'#q':>5}")
    for mode in MODES:
        r, p, f, n = macro(ranked[mode]["cut"])
        rprec_list = ranked[mode]["rprec"]
        rprec = sum(rprec_list) / len(rprec_list) if rprec_list else 0.0
        print(f"{mode:14} {p:>8.4f} {r:>8.4f} {f:>8.4f} {rprec:>8.4f} {n:>5}")

    # --- Cascade funnel: fixed cutoffs over the sim_score ranking ---
    print("-" * 84)
    print("Cascade funnel (rank by sim_score = LLM ordering; S-recall above = "
          "coarse-filter ceiling)")
    print("  nP@k = matched/min(k, n_gold): ceiling-normalized P@k, upper bound 1; "
          "nDCG@k binary, dedup-matched")
    print(f"{'mode':14} {'k':>4} {'R@k':>8} {'P@k':>8} {'F1@k':>8} "
          f"{'nP@k':>8} {'nDCG@k':>8} {'#q':>5}")
    for mode in MODES:
        for fk in FUNNEL_KS:
            vals = funnel[mode][fk]
            n = len(vals)
            means = [sum(v[i] for v in vals) / n if n else 0.0 for i in range(5)]
            print(f"{mode:14} {fk:>4} "
                  + " ".join(f"{m:>8.4f}" for m in means) + f" {n:>5}")
        print()
    print("=" * 84)


def main():
    args = [a for a in sys.argv[1:]]
    verbose = "--verbose" in args
    args = [a for a in args if a != "--verbose"]
    # --k <ngold2|5|10|50|...> : ranked B_raw cutoff (default n_gold+2)
    k_arg = "ngold2"
    rest = []
    i = 0
    while i < len(args):
        if args[i] == "--k" and i + 1 < len(args):
            k_arg = args[i + 1]
            i += 2
        elif args[i].startswith("--k="):
            k_arg = args[i].split("=", 1)[1]
            i += 1
        else:
            rest.append(args[i])
            i += 1
    args = rest

    gold_map = load_gold()
    if not gold_map:
        print("WARNING: no benchmark gold loaded; arxiv_id mode will be empty.")

    base = "gen_result"
    if args:
        run_dirs = [args[0].rstrip("/\\")]
    else:
        run_dirs = sorted(d for d in glob.glob(os.path.join(base, "*")) if os.path.isdir(d))
    if not run_dirs:
        print(f"No run directories found under {base}/")
        return

    for run_dir in run_dirs:
        (results, ranked, funnel, asweep, asweep_counts, set_counts, ks_used,
         per_query, total, mg, ug) = evaluate_run(run_dir, gold_map, k_arg)
        print_report(run_dir, results, ranked, funnel, asweep, asweep_counts,
                     set_counts, ks_used, k_arg, per_query, total, mg, ug, verbose)


if __name__ == "__main__":
    main()
