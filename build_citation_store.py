# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
build_citation_store.py — build the local arXiv->arXiv forward-citation LMDB
(citation_store.CitationStore) from the Semantic Scholar bulk Datasets API.

Pipeline (S2 "arxiv subset bulk"):
  1. stream the `papers` dataset -> corpusid -> arXiv id map (arXiv papers only)
  2. stream the `citations` dataset -> keep edges whose BOTH endpoints are arXiv
     -> accumulate citing_arxiv -> [cited_arxiv, ...]
  3. write the adjacency into LMDB (edges only; metadata stays in recovered.db)

Disk: only the final LMDB (~hundreds of MB) is kept; with --stream-download each
dataset file is downloaded, parsed, and deleted in turn, so peak disk stays low.
RAM: corpusid->arxiv map (~0.4GB) + adjacency (~1-3GB) for the full arXiv subset.

Usage (run from repo root, SPAR conda env, ideally overnight):
  # full build, streaming each remote file then deleting it
  python3 build_citation_store.py --stream-download

  # build from already-downloaded .gz files
  python3 build_citation_store.py --papers-glob 'dl/papers/*.gz' \
                                  --citations-glob 'dl/citations/*.gz'

  # tiny smoke test on local synthetic/sample files, custom output
  python3 build_citation_store.py --papers-glob sample/papers*.gz \
        --citations-glob sample/cit*.gz --out ./database/citations_test.lmdb
"""
import argparse
import glob
import gzip
import json
import os
import random
import time

import requests

from citation_store import CitationWriter, CitationStore, DEFAULT_PATH
from global_config import SEMANTIC_SCHOLAR_API_KEY, PROXIES
from log import logger

DATASETS_API = "https://api.semanticscholar.org/datasets/v1"


# ---------------------------------------------------------------------------
# remote dataset file listing / download
# ---------------------------------------------------------------------------
def _headers():
    return {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}


def _api_get(url: str, timeout: int = 120, tries: int = 15):
    """GET a Datasets-API metadata endpoint with jittered backoff on 429/5xx.

    These endpoints (release/latest, dataset file listing) are needed only ONCE
    each, but the proxy's shared exit IP is rate-limited by S2 (~1 req/s, and
    OTHER users of the same VPN node consume the same per-IP budget), so any
    single request 429s often (~40-60% even at 2.5s spacing). The cure is to keep
    retrying with >=3s spacing and RANDOM jitter — the jitter desyncs our retries
    from the other traffic on the node so we eventually land in a free window.
    """
    for attempt in range(tries):
        # >=3s floor + random jitter (avoid lock-stepping with co-tenants on the IP)
        wait = round(min(30, 3 * (attempt + 1)) + random.uniform(0, 2.5), 1)
        try:
            r = requests.get(url, headers=_headers(), timeout=timeout, proxies=PROXIES)
            if r.status_code == 200:
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                logger.warning(f"{r.status_code} on {url} — retry {attempt+1}/{tries} in {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"request error on {url}: {e} — retry {attempt+1}/{tries} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(
        f"Datasets API GET failed after {tries} tries (shared proxy IP "
        f"rate-limited by S2 — try a less-busy Clash node): {url}"
    )


def list_dataset_files(dataset: str, release: str = "latest") -> list:
    if release == "latest":
        release = _api_get(f"{DATASETS_API}/release/latest", timeout=60).json()["release_id"]
    r = _api_get(f"{DATASETS_API}/release/{release}/dataset/{dataset}", timeout=120)
    files = r.json().get("files", [])
    logger.info(f"[{dataset}] release {release}: {len(files)} files")
    return files


def download(url: str, dest: str, chunk: int = 1 << 20, tries: int = 6, timeout: int = 600):
    """Download to dest with retry + HTTP Range resume of a partial .part file.

    - Skips entirely if dest already exists (a completed prior download) — this
      is what makes a crashed/interrupted build re-runnable: completed files are
      not re-fetched.
    - On a transient failure mid-file, resumes from the bytes already in .part
      via a Range request (S3 pre-signed URLs support ranged GETs).
    """
    if os.path.exists(dest):
        logger.info(f"skip (already downloaded): {os.path.basename(dest)}")
        return
    tmp = dest + ".part"
    for attempt in range(tries):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        headers, mode = {}, "wb"
        if have:
            headers["Range"] = f"bytes={have}-"
            mode = "ab"
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              proxies=PROXIES, headers=headers) as r:
                if have and r.status_code == 200:
                    mode, have = "wb", 0          # server ignored Range -> restart
                elif r.status_code not in (200, 206):
                    r.raise_for_status()
                with open(tmp, mode) as fw:
                    for c in r.iter_content(chunk_size=chunk):
                        if c:
                            fw.write(c)
            os.replace(tmp, dest)
            return
        except requests.exceptions.RequestException as e:
            wait = min(120, 5 * (attempt + 1))
            logger.warning(f"download {os.path.basename(dest)} failed ({e}); "
                           f"retry {attempt+1}/{tries} in {wait}s (resume from {have}B)")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {tries} tries: {url}")


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def iter_jsonl_gz(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _arxiv_of(rec) -> str:
    ext = rec.get("externalids") or rec.get("externalIds") or {}
    return ext.get("ArXiv") or ext.get("arxiv") or ""


def papers_to_cid2arx(paper_paths, normalize) -> dict:
    cid2arx = {}
    for p in paper_paths:
        n0 = len(cid2arx)
        for rec in iter_jsonl_gz(p):
            ax = _arxiv_of(rec)
            if not ax:
                continue
            cid = rec.get("corpusid", rec.get("corpusId"))
            if cid is None:
                continue
            cid2arx[int(cid)] = normalize(ax)
        logger.info(f"papers {os.path.basename(p)}: +{len(cid2arx) - n0} arXiv "
                    f"(total {len(cid2arx)})")
    return cid2arx


def stream_shard_to_writer(path, cid2arx, writer) -> int:
    """Stream one citations shard's arXiv-arXiv edges STRAIGHT into the LMDB writer.

    No in-memory adjacency: each kept edge is appended via writer.add() (dupsort
    dedups), so RAM stays O(1) regardless of how many shards are processed.
    Returns the number of arXiv-arXiv edges seen in this shard.
    """
    kept = 0
    for rec in iter_jsonl_gz(path):
        c = rec.get("citingcorpusid", rec.get("citingCorpusId"))
        d = rec.get("citedcorpusid", rec.get("citedCorpusId"))
        if c is None or d is None:
            continue
        ca = cid2arx.get(int(c))
        da = cid2arx.get(int(d))
        if ca and da:
            writer.add(ca, da)
            kept += 1
    return kept


def _parsed_path(out_path: str) -> str:
    """Checkpoint file recording which shards have been parsed INTO the store."""
    return out_path.rstrip("/") + ".parsed.json"


def _load_parsed(out_path: str) -> set:
    p = _parsed_path(out_path)
    if os.path.exists(p):
        try:
            with open(p) as fr:
                return set(json.load(fr))
        except Exception:
            logger.warning(f"could not read parse checkpoint {p}; starting fresh")
    return set()


def _save_parsed(out_path: str, parsed: set) -> None:
    p = _parsed_path(out_path)
    tmp = p + ".tmp"
    with open(tmp, "w") as fw:
        json.dump(sorted(parsed), fw)
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build local arXiv citation LMDB from S2 bulk")
    ap.add_argument("--out", default=DEFAULT_PATH)
    ap.add_argument("--papers-glob", default=None, help="local papers .gz files")
    ap.add_argument("--citations-glob", default=None, help="local citations .gz files")
    ap.add_argument("--stream-download", action="store_true",
                    help="download each remote file, parse it, then delete it")
    ap.add_argument("--dl-dir", default="./dl", help="scratch dir for --stream-download")
    ap.add_argument("--release", default="latest")
    ap.add_argument("--no-normalize", action="store_true",
                    help="store raw arXiv ids (default: strip version via normalize_arxiv_id)")
    ap.add_argument("--keep", action="store_true",
                    help="keep papers files after Stage 1 (default: delete them, "
                         "Stage 1 is checkpointed so they aren't needed again)")
    ap.add_argument("--cleanup", action="store_true",
                    help="delete the whole --dl-dir scratch tree after a successful build")
    args = ap.parse_args()

    if args.no_normalize:
        normalize = lambda x: x
    else:
        from api_web import normalize_arxiv_id
        normalize = lambda x: normalize_arxiv_id(x) or x

    t0 = time.time()

    # ---- Stage 1: papers -> corpusid->arxiv (checkpointed) ----
    import pickle
    import shutil
    ckpt = os.path.join(args.dl_dir, "cid2arx.pkl")
    if os.path.exists(ckpt):
        # The checkpoint is the authoritative corpusid->arXiv map. Load it in
        # EITHER mode: Stage 1 deletes the papers .gz once checkpointed, so the
        # --citations-glob (offline parse) path has no papers to glob and MUST
        # reuse the checkpoint.
        with open(ckpt, "rb") as fr:
            cid2arx = pickle.load(fr)
        logger.info(f"Stage 1: loaded checkpoint ({len(cid2arx)} arXiv papers)")
    elif args.stream_download:
        cid2arx = {}
        pdir = os.path.join(args.dl_dir, "papers")
        os.makedirs(pdir, exist_ok=True)
        files = list_dataset_files("papers", args.release)
        for i, f in enumerate(files, 1):
            url = f if isinstance(f, str) else f.get("url") or f
            dest = os.path.join(pdir, f"papers_{i:03d}.gz")
            logger.info(f"[papers {i}/{len(files)}]")
            download(url, dest)   # skips if already present (resume)
            cid2arx.update(papers_to_cid2arx([dest], normalize))
        with open(ckpt, "wb") as fw:
            pickle.dump(cid2arx, fw)
        logger.info(f"Stage 1 checkpoint saved: {ckpt}")
        if not args.keep:
            shutil.rmtree(pdir, ignore_errors=True)  # papers no longer needed
    else:
        ppaths = sorted(glob.glob(args.papers_glob)) if args.papers_glob else []
        if not ppaths:
            ap.error("no papers files (use --papers-glob, --stream-download, "
                     "or an existing dl/cid2arx.pkl checkpoint)")
        cid2arx = papers_to_cid2arx(ppaths, normalize)
    logger.info(f"Stage 1 done: {len(cid2arx)} arXiv papers mapped "
                f"({time.time()-t0:.0f}s)")

    # ---- Stage 2: citations -> LMDB edges (streamed, checkpointed) ----
    # Each shard is parsed and its edges are written STRAIGHT into the dupsort
    # LMDB (O(1) RAM), then the shard is recorded in a parse checkpoint so a
    # restart skips shards already in the store (download AND re-parse). dupsort
    # makes a re-parsed shard idempotent, so the checkpoint is an optimisation,
    # not a correctness requirement.
    parsed = _load_parsed(args.out)
    if parsed:
        logger.info(f"Stage 2: resume — {len(parsed)} shards already in the store")
    writer = CitationWriter(args.out)
    n_edges = 0
    try:
        if args.stream_download:
            cdir = os.path.join(args.dl_dir, "citations")
            os.makedirs(cdir, exist_ok=True)
            files = list_dataset_files("citations", args.release)
            shard_list = []
            for i, f in enumerate(files, 1):
                url = f if isinstance(f, str) else f.get("url") or f
                shard_list.append((f"cit_{i:03d}.gz", url))
        else:
            cpaths = sorted(glob.glob(args.citations_glob)) if args.citations_glob else []
            if not cpaths:
                ap.error("no citations files (use --citations-glob or --stream-download)")
            shard_list = [(os.path.basename(p), p) for p in cpaths]

        for idx, (name, url_or_path) in enumerate(shard_list, 1):
            logger.info(f"[citations {idx}/{len(shard_list)}] {name}")
            if name in parsed:
                logger.info(f"  already parsed into store — skip")
                continue
            if args.stream_download:
                dest = os.path.join(cdir, name)
                download(url_or_path, dest)   # skips if already present (resume)
            else:
                dest = url_or_path
            kept = stream_shard_to_writer(dest, cid2arx, writer)
            writer.commit()                   # durable BEFORE marking parsed
            parsed.add(name)
            _save_parsed(args.out, parsed)
            n_edges += kept
            logger.info(f"  +{kept} arXiv-arXiv edges (session total {n_edges})")
    finally:
        writer.close()
    logger.info(f"Stage 2/3 done: streamed {n_edges} edges this session "
                f"({time.time()-t0:.0f}s)")

    # Authoritative unique-edge total from the dupsort DB (dedup-aware).
    try:
        total_edges = CitationStore(args.out).stat().get("entries", -1)
    except Exception:
        total_edges = -1

    print("-" * 60)
    print(f"citation store -> {args.out}")
    print(f"arXiv papers mapped : {len(cid2arx)}")
    print(f"shards parsed       : {len(parsed)}/{len(shard_list)}")
    print(f"edges parsed (sess) : {n_edges}")
    print(f"unique edges stored : {total_edges}")
    print(f"elapsed             : {time.time()-t0:.0f}s")

    if args.stream_download and args.cleanup:
        shutil.rmtree(args.dl_dir, ignore_errors=True)
        print(f"cleaned scratch dir : {args.dl_dir}")


if __name__ == "__main__":
    main()
