# !/usr/bin/env python
# -*- coding:utf-8 -*-
# ==================================================================
# [Author]       : shixiaofeng
# [Descriptions] :
# ==================================================================

# Parse --model BEFORE importing global_config so SPAR_MODEL env var is in place
import sys
import os
import time
import argparse

_parser = argparse.ArgumentParser(description="SPAR benchmark runner")
_parser.add_argument("benchmark_name", help="AutoScholarQuery | OwnBenchmark")
_parser.add_argument(
    "--model", default=None,
    help="Model key: DeepSeek-V4 (default) | DeepSeek-V4-Pro | Qwen3-32B"
)
_parser.add_argument(
    "--qids", default=None,
    help="Comma-separated qids to run (skips random sampling), e.g. AutoScholarQuery_test_1"
)
_parser.add_argument(
    "--no-date", action="store_true", default=False,
    help="Ignore published_time constraint (set end_date='' for all queries)"
)
_parser.add_argument(
    "--reference", action="store_true", default=False,
    help="Enable Phase 2 reference (citation) expansion"
)
_parser.add_argument(
    "--ref-api", default=None, choices=["openalex", "s2", "local"],
    help="Reference source: openalex (default) | s2 | local (offline arXiv "
         "citation LMDB from build_citation_store.py; falls back to s2/openalex)"
)
_parser.add_argument(
    "--ref-prune", default=None, choices=["freq", "citation"],
    help="Phase 2 reference pruning before LLM scoring: "
         "freq (co-occurrence+citation, top-150, default) | citation (top-200 by citationCount)"
)
_parser.add_argument(
    "--ref-topn", type=int, default=None,
    help="Override the pruning top-N (default: freq=150, citation=200)"
)
_parser.add_argument(
    "--phase3", action="store_true", default=False,
    help="Enable Phase 3 context-driven query generation (BFS depth>1). "
         "Default off = tree stops after depth 1."
)
_parser.add_argument(
    "--rerank", action="store_true", default=False,
    help="BGE cross-encoder rerank of B_raw after collection (needs peft/torch; bge env)"
)
_parser.add_argument(
    "--bge-prefilter", action="store_true", default=False,
    help="Relevance scoring as 'BGE coarse filter -> LLM fine rank' cascade "
         "(Phase 1 + Phase 2). Default off = original LLM-scores-every-doc."
)
_parser.add_argument(
    "--bge-prefilter-topk", type=int, default=None,
    help="Top-K kept for LLM scoring per batch after BGE coarse filtering (default 40)"
)
_parser.add_argument(
    "--score-prompt", default=None, choices=["intent", "rubric"],
    help="LLM fine-rank prompt: intent (default; core-intent vs query-intent "
         "alignment) | rubric (legacy relevance rubric)"
)
_parser.add_argument(
    "--score-batch-size", type=int, default=None,
    help="LLM relevance-scoring batch size: docs scored per LLM call "
         "(default 50; set 1 for legacy per-doc scoring). Intent prompt only."
)
_parser.add_argument(
    "--agent-b", action="store_true", default=False,
    help="Enable Agent B (LLM memory recall + existence verification): LLM-named "
         "papers are verified real (recovered.db + Serper) then injected into B_raw."
)
_parser.add_argument(
    "--agent-c", action="store_true", default=False,
    help="Enable Agent C (intent-constraint decomposition): hard/soft constraint "
         "split -> 3-source search -> citation filter -> merged into B_raw."
)
_parser.add_argument(
    "--no-early-stop", action="store_true", default=False,
    help="Disable the high-rel early stop so Phase 2 reference expansion always "
         "runs at depth 1 (expands top-max_docs high-rel docs when high-rel>max_docs)."
)
args = _parser.parse_args()

if args.model:
    os.environ["SPAR_MODEL"] = args.model  # picked up by global_config.LLM_MODEL_NAME
if args.reference:
    os.environ["SPAR_DO_REFERENCE"] = "1"  # picked up by global_config.DO_REFERENCE_SEARCH
if args.ref_api:
    os.environ["SPAR_REF_API"] = args.ref_api  # picked up by global_config.REFERENCE_API
if args.ref_prune:
    os.environ["SPAR_REF_PRUNE"] = args.ref_prune  # -> global_config.REFERENCE_PRUNE_STRATEGY
if args.ref_topn is not None:
    os.environ["SPAR_REF_TOPN"] = str(args.ref_topn)  # -> global_config.REFERENCE_PRUNE_TOPN
if args.phase3:
    os.environ["SPAR_DO_PHASE3"] = "1"  # -> global_config.DO_PHASE3
if args.rerank:
    os.environ["SPAR_RERANK"] = "1"  # -> global_config.ENABLE_RERANK
if args.bge_prefilter:
    os.environ["SPAR_BGE_PREFILTER"] = "1"  # -> global_config.BGE_PREFILTER
if args.bge_prefilter_topk is not None:
    os.environ["SPAR_BGE_PREFILTER_TOPK"] = str(args.bge_prefilter_topk)  # -> BGE_PREFILTER_TOPK
if args.score_prompt:
    os.environ["SPAR_SCORING_PROMPT"] = args.score_prompt  # -> global_config.SCORING_PROMPT
if args.score_batch_size is not None:
    os.environ["SPAR_SCORE_BATCH_SIZE"] = str(args.score_batch_size)  # -> SCORE_BATCH_SIZE
if args.agent_b:
    os.environ["SPAR_AGENT_B"] = "1"  # -> global_config.AGENT_B_ENABLED
if args.agent_c:
    os.environ["SPAR_AGENT_C"] = "1"  # -> global_config.AGENT_C_ENABLED
if args.no_early_stop:
    os.environ["SPAR_EARLY_STOP"] = "0"  # -> global_config.ENABLE_EARLY_STOP

import json
from tqdm import tqdm
import traceback
import random
from global_config import (
    LLM_MODEL_NAME,
    DO_REFERENCE_SEARCH,
    DO_PHASE3,
    REFERENCE_API,
    REFERENCE_PRUNE_STRATEGY,
    REFERENCE_PRUNE_TOPN,
    ENABLE_RERANK,
    BGE_PREFILTER,
    BGE_PREFILTER_TOPK,
    SCORING_PROMPT,
    SCORE_BATCH_SIZE,
    AGENT_B_ENABLED,
    AGENT_C_ENABLED,
    ENABLE_EARLY_STOP,
    DO_FUSION_JUDGE,
    FUSION_TEMPLATE,
    SEARCH_ROUTES,
    BUILD_DOMAIN_TREE,
    DOMAIN_TREE_DEPTH,
)
import glob
from utils import get_md5
import shutil
from pipeline_spar import AcademicSearchTree

file_lst = [
    "./global_config.py",
    "./instruction.py",
    "./run_spr_agent.py",
    "./search_engine.py",
    "./api_web.py",
    "./pipeline_spar.py",
] + glob.glob("./agents/*.py")

sample_num = 30
score_thresh = 0.5
max_depth = 2
relevance_doc_num = 10

benchmark_map = {
    "AutoScholarQuery": {
        "src_file": "./benchmark/AutoScholarQuery_test.jsonl",
        "select_file": f"./benchmark/AutoScholarQuery_test_select_{sample_num}.jsonl",
    },
    "OwnBenchmark": {
        "src_file": "./benchmark/spar_bench.jsonl",
        "select_file": f"./benchmark/spar_bench_select_{sample_num}.jsonl"

    },
}

benchmark_name = args.benchmark_name
target_qids = set(args.qids.split(",")) if args.qids else None

src_file = benchmark_map[benchmark_name]["src_file"]
select_file = benchmark_map[benchmark_name]["select_file"]
print(f"select_file: {select_file}")


domaintree_tag = f"domaintreeD{DOMAIN_TREE_DEPTH}" if BUILD_DOMAIN_TREE else "domaintreeOff"

# Folder tag: qid-mode keeps runs separate; otherwise it's the sample count.
if target_qids:
    _qtag = "-".join(sorted(target_qids))
    # The rest of the folder name is ~205 chars; ext4's per-component limit is
    # 255, so keep the qid tag short — hash as soon as it would risk overflow
    # (e.g. 2+ full qids). Single qid stays readable.
    run_tag = f"qids_{_qtag}" if len(_qtag) <= 25 else f"qids_{len(target_qids)}q_{get_md5(_qtag)[:8]}"
else:
    run_tag = str(sample_num)

_date_tag = "nodate" if args.no_date else "enddate"
if DO_REFERENCE_SEARCH:
    if BGE_PREFILTER:
        # full ref pool goes through the BGE coarse filter (freq prune bypassed;
        # falls back to freq at runtime only if the reranker fails to load)
        _ref_tag = f"{REFERENCE_API}-bgepool"
    else:
        _topn = REFERENCE_PRUNE_TOPN or (150 if REFERENCE_PRUNE_STRATEGY == "freq" else 200)
        _ref_tag = f"{REFERENCE_API}-{REFERENCE_PRUNE_STRATEGY}{_topn}"
else:
    _ref_tag = "False"
_rerank_tag = "rerankBGE" if ENABLE_RERANK else "rerankOff"
_prefilter_tag = f"prefilterBGE{BGE_PREFILTER_TOPK}" if BGE_PREFILTER else "prefilterOff"
_phase3_tag = "phase3On" if DO_PHASE3 else "phase3Off"
_prompt_tag = f"prompt{SCORING_PROMPT.capitalize()}"  # promptIntent | promptRubric
_scorebatch_tag = f"scoreBatch{SCORE_BATCH_SIZE}" if SCORE_BATCH_SIZE > 1 else "scorePerDoc"
_agents_tag = "agents" + ("B" if AGENT_B_ENABLED else "") + ("C" if AGENT_C_ENABLED else "")
if _agents_tag == "agents":
    _agents_tag = "agentsOff"
_earlystop_tag = "earlystopOn" if ENABLE_EARLY_STOP else "earlystopOff"
# collectCite70sim: Set C = A_raw -> keep top-70% by citation -> rank by sim (replaced AHP)
_folder_name = f"{benchmark_name}_{LLM_MODEL_NAME}_{run_tag}_msearch_{'-'.join(SEARCH_ROUTES)}_depth{max_depth}_{_phase3_tag}_do_reference_{_ref_tag}_query_judge_{DO_FUSION_JUDGE}_fusion_{FUSION_TEMPLATE}_{domaintree_tag}_{_date_tag}_{_rerank_tag}_{_prefilter_tag}_{_prompt_tag}_{_scorebatch_tag}_{_agents_tag}_{_earlystop_tag}_collectCite70sim_no_autocorrect_pasa_score_{score_thresh}"
# ext4 limits a single path component to 255 bytes; the full tag set is ~252,
# so a longer run_tag (multi-qid) or model name overflows. Cap deterministically.
if len(_folder_name) > 254:
    _folder_name = _folder_name[:245] + "_" + get_md5(_folder_name)[:8]
output_folder = f"./gen_result/{_folder_name}"

print(f"output_folder: {output_folder}")

os.makedirs(output_folder, exist_ok=True)


search_agent = AcademicSearchTree(
    max_depth=max_depth, max_docs=relevance_doc_num, similarity_threshold=score_thresh
)


for one in file_lst:
    shutil.copy2(one, output_folder)

already = {}
for one in glob.glob(f"{output_folder}/*.json"):
    with open(one, "r") as fr:
        info = json.load(fr)
    question = info["search_query"]
    already[question] = one

with open(src_file, "r") as f:
    if src_file.endswith(".jsonl"):
        lines = f.readlines()
        if target_qids:
            selected = []
            for ln in lines:
                try:
                    if json.loads(ln).get("qid") in target_qids:
                        selected.append(ln)
                except Exception:
                    continue
            lines = selected
            print(f"Selected {len(lines)} lines by qid: {sorted(target_qids)}")
        else:
            random.seed(123)
            random.shuffle(lines)
            lines = lines[:sample_num]
    elif src_file.endswith(".json"):
        lines = json.load(f)
    print(f"lines: {len(lines)}")

    with open(select_file,"w") as fw:
        for one in lines:
            fw.write(one.strip() + "\n")


    for idx, line in tqdm(enumerate(lines), total=len(lines), desc="Processing lines"):
        try:
            if isinstance(line, str):
                data = json.loads(line)
                question = data["question"]
            elif isinstance(line, dict):
                data = line
                question = data["query"]
            else:
                data = {}
                question = line

            end_date = "" if args.no_date else (
                data.get("published_time")
                or data.get("source_meta", {}).get("published_time", "")
            )
            if question in already:
                print(f"pass: {already[question]}")
                continue

            dest_name = get_md5(question)
            dest_file = os.path.join(output_folder, f"{dest_name}.json")

            _t_e2e = time.time()
            sorted_docs = search_agent.search(question, end_date=end_date)

            # Persist AHP top-k so evaluate.py can score Set C
            search_agent.root.extra["ahp_topk"] = sorted_docs

            # End-to-end per sample (search + post-processing, before JSON dump /
            # Graphviz render). Stored alongside the per-phase timings.
            search_agent.root.extra.setdefault("timing", {})["end_to_end"] = round(
                time.time() - _t_e2e, 3
            )

            if "answer" in data:
                search_agent.root.extra["answer"] = data["answer"]

            elif "data_result_add_score" in data:
                search_agent.root.extra["answer"] = [
                    one["title"] for one in data["data_result_add_score"]
                ]

            if output_folder != "":
                res = search_agent.root.convert_to_dict()
                with open(dest_file, "w") as fw:
                    json.dump(res, fw, indent=2)

            try:
                print("draw search tree")
                search_agent.visualize_tree(f"{output_folder}/{dest_name}")
            except:
                traceback.print_exc()
                pass

            # break

        except:
            traceback.print_exc()
            pass

print(f"output_folder: {output_folder}")
