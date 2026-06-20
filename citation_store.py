# !/usr/bin/env python
# -*- coding:utf-8 -*-
"""
CitationStore — compact local arXiv->arXiv forward-citation graph (LMDB).

Stores ONLY edges, in a DUPSORT sub-database "edges": each forward edge is one
record key=citing arXiv id, value=cited arXiv id (a citing paper therefore has
ONE record per cited paper, grouped under its key). Paper metadata
(title/abstract) is NOT duplicated here — it is resolved on demand from
recovered.db (the full arXiv snapshot). This keeps the store compact.

Why DUPSORT (vs the old "key -> newline-joined list, written once"): with dupsort
each edge is appended independently and LMDB natively rejects a duplicate
(citing, cited) pair, so the builder can STREAM edges straight to disk shard by
shard — no need to accumulate the whole adjacency in RAM before writing, and
re-parsing a shard on resume is idempotent. `references()` reads identically.

Built by build_citation_store.py from the Semantic Scholar bulk Datasets API
(papers + citations). Read by agents/reference_agent.py for `--ref-api local`.

Lookup is a single LMDB dup-cursor walk (memory-mapped, ~microseconds):
    store.references("2310.06825") -> ["1706.03762", "1810.04805", ...]
"""
import os
from typing import List

import lmdb

from log import logger

DEFAULT_PATH = "./database/citations.lmdb"
# LMDB map_size is the MAX size the file may grow to (sparse on disk; only the
# used pages occupy space). 32 GiB headroom for the full arXiv subset + growth.
DEFAULT_MAP_SIZE = 32 << 30
# Named sub-database holding the edges (dupsort requires a named DB, not main).
EDGES_DB = b"edges"


class CitationStore:
    """Read API over the LMDB forward-citation graph (lazy-opened, read-only)."""

    def __init__(self, path: str = DEFAULT_PATH, map_size: int = DEFAULT_MAP_SIZE):
        self.path = path
        self.map_size = map_size
        self._env = None
        self._db = None

    @staticmethod
    def exists(path: str = DEFAULT_PATH) -> bool:
        return os.path.isdir(path) and os.path.exists(os.path.join(path, "data.mdb"))

    @property
    def env(self):
        if self._env is None:
            self._env = lmdb.open(
                self.path,
                map_size=self.map_size,
                subdir=True,
                readonly=True,
                lock=False,        # readers don't need the writer lock
                readahead=False,
                max_dbs=1,         # the named "edges" dupsort sub-db
            )
            # create=False: the sub-db must already exist (built by the writer).
            self._db = self._env.open_db(EDGES_DB, dupsort=True, create=False)
        return self._env

    def references(self, arxiv_id: str) -> List[str]:
        """Cited arXiv ids for a citing arXiv paper (forward edges); [] if absent."""
        if not arxiv_id:
            return []
        try:
            with self.env.begin(buffers=False) as txn:
                cur = txn.cursor(self._db)
                if not cur.set_key(arxiv_id.encode("utf-8")):
                    return []
                out = [cur.value().decode("utf-8")]
                while cur.next_dup():
                    out.append(cur.value().decode("utf-8"))
                return out
        except Exception:
            logger.error(f"CitationStore lookup failed for {arxiv_id}")
            return []

    def __contains__(self, arxiv_id: str) -> bool:
        try:
            with self.env.begin() as txn:
                return txn.cursor(self._db).set_key(arxiv_id.encode("utf-8"))
        except Exception:
            return False

    def stat(self) -> dict:
        with self.env.begin() as txn:
            return dict(txn.stat(self._db))

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._db = None


class CitationWriter:
    """Streaming DUPSORT writer used by build_citation_store.py.

    `add(citing, cited)` appends ONE edge; LMDB+dupsort silently drops a repeat of
    an identical (citing, cited) pair, so callers can stream edges in any order,
    across shards and across restarts, without accumulating an in-memory
    adjacency and without producing duplicate edges. `put(citing, [cited...])` is
    kept for back-compat (e.g. S1 increments) and just loops over `add`.
    """

    def __init__(self, path: str = DEFAULT_PATH, map_size: int = DEFAULT_MAP_SIZE,
                 commit_every: int = 2_000_000):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.env = lmdb.open(path, map_size=map_size, subdir=True, max_dbs=1)
        self.db = self.env.open_db(EDGES_DB, dupsort=True)
        self.commit_every = commit_every
        self._txn = self.env.begin(write=True)
        self._n = 0
        # Count of add() calls (NOT unique edges — py-lmdb's dupsort put returns
        # True even for an idempotent duplicate). For the true deduped edge total
        # read CitationStore(path).stat()["entries"].
        self.written = 0

    def add(self, citing: str, cited: str):
        """Append a single forward edge (citing -> cited). Idempotent on repeats
        (dupsort drops a repeat of the same (citing, cited) pair)."""
        if not citing or not cited:
            return
        self._txn.put(citing.encode("utf-8"), cited.encode("utf-8"), db=self.db)
        self.written += 1
        self._n += 1
        if self._n >= self.commit_every:
            self.commit()

    def put(self, citing_arxiv: str, cited_arxiv_ids: List[str]):
        """Back-compat: store one paper's forward edges (each appended via add)."""
        for c in cited_arxiv_ids:
            self.add(citing_arxiv, c)

    def commit(self):
        """Flush the current write transaction and open a fresh one."""
        self._txn.commit()
        self._txn = self.env.begin(write=True)
        self._n = 0

    def close(self):
        try:
            self._txn.commit()
        except Exception:
            pass
        self.env.sync()
        self.env.close()
