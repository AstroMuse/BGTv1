"""
Verify: in old run (no date constraint), did after-cutoff papers in searched_docs
reference the gold papers (enabling RefChain to find them)?

Steps:
1. Find the diffusion query file in the old run
2. Get its gold IDs from benchmark
3. Classify searched_docs as before/after cutoff
4. For after-cutoff papers: check if their stored references include gold IDs
5. For gold papers found: trace HOW they got into searched_docs (direct search vs refchain)
"""
import json, glob, re, os
from datetime import datetime

ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5})")
OLD_RUN = "gen_result/AutoScholarQuery_20_msearch_arxiv-openalex_depth2_do_reference_True_query_judge_True_fusion_AUTOMATIC_enddate_no_autocorrect_pasa_score_0.5"
BENCHMARK = "benchmark/AutoScholarQuery_test.jsonl"

def norm_id(x):
    m = ARXIV_RE.search(str(x))
    return m.group(1) if m else None

def id_to_date(aid):
    m = re.match(r"(\d{2})(\d{2})\.\d+", str(aid))
    if not m:
        return None
    yy, mm = int(m.group(1)), int(m.group(2))
    try:
        return datetime(2000 + yy, mm, 1)
    except:
        return None

# Load benchmark gold map
gold_map = {}
pub_time_map = {}
with open(BENCHMARK) as f:
    for line in f:
        d = json.loads(line.strip())
        q = d.get("question", "")
        ids = {norm_id(i) for i in d.get("answer_arxiv_id", []) if norm_id(i)}
        pt = d.get("published_time") or d.get("source_meta", {}).get("published_time", "")
        if ids:
            gold_map[q] = ids
        if pt:
            pub_time_map[q] = pt

# Find diffusion file
diffusion_file = None
diffusion_data = None
for fp in glob.glob(f"{OLD_RUN}/*.json"):
    with open(fp) as f:
        d = json.load(f)
    q = d.get("search_query", "")
    if "diffus" in q.lower():
        diffusion_file = fp
        diffusion_data = d
        diffusion_query = q
        break

if not diffusion_data:
    print("Diffusion query not found in old run!")
    exit()

print(f"Query: {diffusion_query}")
gold_ids = gold_map.get(diffusion_query, set())
pub_time = pub_time_map.get(diffusion_query, "")
print(f"Gold IDs: {gold_ids}")
print(f"published_time from benchmark: '{pub_time}'")

extra = diffusion_data.get("extra", {})
searched_docs = extra.get("searched_docs", {})
hrd = extra.get("hight_relevance_docs", [])
hrd_norm = [norm_id(x) for x in hrd if norm_id(x)]

# Use published_time from benchmark (old run had end_date="" so cutoff was ignored)
if pub_time:
    cutoff_dt = datetime.strptime(pub_time, "%Y%m%d")
else:
    cutoff_dt = None
    print("WARNING: No published_time found for this query - can't classify by date")

print(f"\nTotal searched_docs: {len(searched_docs)}")

before_cutoff = {}
after_cutoff = {}
unknown = {}

for pid, doc in searched_docs.items():
    nid = norm_id(pid)
    if not nid:
        unknown[pid] = doc
        continue
    dt = id_to_date(nid)
    if dt is None:
        py = doc.get("publicationYear", "")
        try:
            dt = datetime(int(str(py)[:4]), 1, 1)
        except:
            unknown[nid] = doc
            continue
    if cutoff_dt and dt > cutoff_dt:
        after_cutoff[nid] = doc
    else:
        before_cutoff[nid] = doc

print(f"  Before cutoff ({pub_time}): {len(before_cutoff)}")
print(f"  After  cutoff            : {len(after_cutoff)}")
print(f"  Unknown date             : {len(unknown)}")

# Gold paper status
print(f"\n{'='*70}")
print("GOLD PAPER STATUS")
print(f"{'='*70}")
for gid in sorted(gold_ids):
    if gid in before_cutoff:
        loc = "before_cutoff"
        score = before_cutoff[gid].get("sim_score", "?")
        title = before_cutoff[gid].get("title", "")[:60]
    elif gid in after_cutoff:
        loc = "after_cutoff"
        score = after_cutoff[gid].get("sim_score", "?")
        title = after_cutoff[gid].get("title", "")[:60]
    elif gid in unknown:
        loc = "unknown_date"
        score = unknown[gid].get("sim_score", "?")
        title = unknown[gid].get("title", "")[:60]
    else:
        loc = "NOT_RECALLED"
        score = "-"
        title = ""
    rank = (hrd_norm.index(gid) + 1) if gid in hrd_norm else "-"
    print(f"  [{gid}] {loc}  score={score}  rank={rank}")
    if title:
        print(f"         \"{title}\"")

# Key question: for gold papers that ARE in searched_docs,
# check doc["source"] to see if they came from direct search or refchain
print(f"\n{'='*70}")
print("HOW DID GOLD PAPERS GET INTO searched_docs? (source field)")
print(f"{'='*70}")
for gid in sorted(gold_ids):
    doc = (before_cutoff.get(gid) or after_cutoff.get(gid) or unknown.get(gid))
    if doc:
        src = doc.get("source", "unknown")
        print(f"  [{gid}] source='{src}'")

# Now the core question: do after-cutoff papers reference the gold papers?
print(f"\n{'='*70}")
print("DO AFTER-CUTOFF PAPERS REFERENCE GOLD PAPERS? (RefChain hypothesis)")
print(f"{'='*70}")
gold_cited_by = {gid: [] for gid in gold_ids}

for nid, doc in after_cutoff.items():
    refs = doc.get("references", [])
    for ref in refs:
        ref_id = norm_id(ref.get("paper_id", "") or ref.get("arxivId", ""))
        if ref_id and ref_id in gold_ids:
            gold_cited_by[ref_id].append(nid)

for gid, citers in gold_cited_by.items():
    print(f"  Gold [{gid}] cited by {len(citers)} after-cutoff papers in searched_docs")
    for citer in citers[:5]:
        doc = after_cutoff[citer]
        print(f"    <- [{citer}] score={doc.get('sim_score','?')}  \"{doc.get('title','')[:60]}\"")

# Also check before-cutoff papers referencing gold (RefChain from in-scope papers)
print(f"\n{'='*70}")
print("BEFORE-CUTOFF PAPERS REFERENCING GOLD (normal refchain)")
print(f"{'='*70}")
gold_cited_by_before = {gid: [] for gid in gold_ids}
for nid, doc in before_cutoff.items():
    refs = doc.get("references", [])
    for ref in refs:
        ref_id = norm_id(ref.get("paper_id", "") or ref.get("arxivId", ""))
        if ref_id and ref_id in gold_ids:
            gold_cited_by_before[ref_id].append(nid)

for gid, citers in gold_cited_by_before.items():
    print(f"  Gold [{gid}] cited by {len(citers)} before-cutoff papers")
    for citer in citers[:3]:
        doc = before_cutoff[citer]
        print(f"    <- [{citer}] score={doc.get('sim_score','?')}  \"{doc.get('title','')[:60]}\"")
