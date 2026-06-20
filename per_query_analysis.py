"""
Per-query breakdown:
  B_total  : total papers in searched_docs (full recall pool)
  B_gold   : how many gold papers are in searched_docs
  A_total  : papers in hight_relevance_docs (after LLM scoring filter)
  A_gold   : gold papers that passed the filter
  ranks    : position of each gold paper in hight_relevance_docs (1-based)
             or score if found in B but filtered out
"""
import json, glob, re, sys

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
BENCHMARK_FILES = ["benchmark/AutoScholarQuery_test.jsonl", "benchmark/spar_bench.jsonl"]

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    "gen_result/AutoScholarQuery_Qwen3-32B_30_msearch_arxiv-openalex_depth2_"
    "do_reference_False_query_judge_True_fusion_AUTOMATIC_domaintreeD2_"
    "enddate_no_autocorrect_pasa_score_0.5"
)

def norm_id(x):
    m = ARXIV_RE.search(str(x))
    return m.group(1) if m else None

# Load gold
gold_map = {}
for path in BENCHMARK_FILES:
    try:
        with open(path) as f:
            for line in f:
                d = json.loads(line.strip())
                q = d.get("question") or d.get("query")
                if not q:
                    continue
                ids = {norm_id(i) for i in d.get("answer_arxiv_id", []) if norm_id(i)}
                if ids:
                    gold_map[q] = ids
    except FileNotFoundError:
        pass

files = sorted(glob.glob(f"{RUN_DIR}/*.json"))
print(f"{'#':>3} {'qid/query':40} {'B_tot':>6} {'B_gld':>6} {'A_tot':>6} {'A_gld':>6}  ranks_in_A  (filtered_out_score)")
print("-" * 120)

for idx, fp in enumerate(files, 1):
    with open(fp) as f:
        d = json.load(f)
    query = d.get("search_query", "")
    extra = d.get("extra", {})

    searched_docs = extra.get("searched_docs", {})
    hrd = extra.get("hight_relevance_docs", [])  # ordered list of paper_ids

    # Normalize IDs in searched_docs
    b_id_map = {}  # norm_id -> sim_score
    for pid, doc in searched_docs.items():
        nid = norm_id(pid)
        if nid:
            b_id_map[nid] = doc.get("sim_score", 0)

    # Normalize hrd
    a_ordered = [norm_id(x) for x in hrd if norm_id(x)]

    gold_ids = gold_map.get(query, set())

    b_gold = [g for g in gold_ids if g in b_id_map]
    a_gold = [g for g in gold_ids if g in set(a_ordered)]

    # Ranks in A (1-based)
    rank_strs = []
    for g in gold_ids:
        if g in set(a_ordered):
            rank = a_ordered.index(g) + 1
            rank_strs.append(f"{g}@{rank}")
        elif g in b_id_map:
            score = b_id_map[g]
            rank_strs.append(f"{g}(filtered,score={score:.2f})")
        else:
            rank_strs.append(f"{g}(not_recalled)")

    q_display = (query[:38] + "..") if len(query) > 40 else query
    print(f"{idx:>3} {q_display:40} {len(searched_docs):>6} {len(b_gold):>6} {len(a_ordered):>6} {len(a_gold):>6}  {' | '.join(rank_strs)}")

print("-" * 120)
# Summary
total_b_gold = 0; total_a_gold = 0; total_gold = 0; total_b = 0; total_a = 0
for fp in files:
    with open(fp) as f:
        d = json.load(f)
    query = d.get("search_query", "")
    extra = d.get("extra", {})
    searched_docs = extra.get("searched_docs", {})
    hrd = extra.get("hight_relevance_docs", [])
    b_id_map = {norm_id(pid): doc.get("sim_score", 0) for pid, doc in searched_docs.items() if norm_id(pid)}
    a_ordered = [norm_id(x) for x in hrd if norm_id(x)]
    gold_ids = gold_map.get(query, set())
    b_gold = [g for g in gold_ids if g in b_id_map]
    a_gold = [g for g in gold_ids if g in set(a_ordered)]
    total_b += len(searched_docs); total_a += len(a_ordered)
    total_b_gold += len(b_gold); total_a_gold += len(a_gold); total_gold += len(gold_ids)

n = len(files)
print(f"{'AVG':>3} {'':40} {total_b/n:>6.1f} {total_b_gold/n:>6.2f} {total_a/n:>6.1f} {total_a_gold/n:>6.2f}  total_gold={total_gold}, recall_B={total_b_gold/total_gold:.3f}, recall_A={total_a_gold/total_gold:.3f}")
