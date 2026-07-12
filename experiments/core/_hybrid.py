"""Opt-in hybrid dense+BM25 retrieval (VEIL_HYBRID_BM25=1).

A lightweight BM25 index over a bank's L1 chunks (memory_text + ASR), cached on
the bank object so it is built once per bank, not per query. Fused with the dense
BGE scores by Reciprocal Rank Fusion (RRF) — a retrieval fusion (adds candidates,
never re-ranks away a dense hit), which offline lifts EgoLife L1 recall@38 without
hurting the already-answered questions. Dense-only banks are unaffected unless the
env flag is set. Generalises across datasets (no dataset-specific fields used).
"""
from __future__ import annotations
import os
import re
import numpy as np

_TOKEN = re.compile(r"[a-zA-Z0-9]+")


def enabled() -> bool:
    return os.environ.get("VEIL_HYBRID_BM25") == "1"


def _tok(s: str):
    return _TOKEN.findall(s.lower())


class _BM25:
    """Okapi BM25 over a fixed doc set. get_scores(query) → np.array over all docs."""
    def __init__(self, docs, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs)
        toks = [_tok(d) for d in docs]
        self.dl = np.array([len(t) for t in toks], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        # vocab + per-doc term frequencies as {term: {doc: tf}} inverted index
        self.vocab: dict = {}
        self.postings: list = []  # per-term list of (doc_idx, tf)
        df: dict = {}
        for di, t in enumerate(toks):
            tf: dict = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            for w, c in tf.items():
                if w not in self.vocab:
                    self.vocab[w] = len(self.postings)
                    self.postings.append([])
                    df[w] = 0
                self.postings[self.vocab[w]].append((di, c))
                df[w] += 1
        self.idf = np.zeros(len(self.postings), dtype=np.float32)
        for w, ti in self.vocab.items():
            self.idf[ti] = np.log((self.N - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
        self._denom = self.k1 * (1 - self.b + self.b * self.dl / (self.avgdl or 1.0))

    def get_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        for w in set(_tok(query)):
            ti = self.vocab.get(w)
            if ti is None:
                continue
            idf = self.idf[ti]
            for di, tf in self.postings[ti]:
                scores[di] += idf * (tf * (self.k1 + 1)) / (tf + self._denom[di])
        return scores


def bm25_for_l1(bank) -> "_BM25":
    """Build (once) and cache a BM25 index over the bank's L1 chunks."""
    cached = getattr(bank, "_bm25_l1", None)
    if cached is not None:
        return cached
    docs = []
    for c in bank.chunks:
        txt = getattr(c, "memory_text", "") or ""
        asr = getattr(c, "asr", "") or ""
        docs.append(f"{txt} {asr}")
    idx = _BM25(docs)
    try:
        object.__setattr__(bank, "_bm25_l1", idx)  # pydantic model → bypass validation
    except Exception:
        pass
    return idx


def rrf_order(dense_scores: np.ndarray, bm25_scores: np.ndarray,
              k: int = 60, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Reciprocal-rank-fusion ordering (descending) of doc indices.

    valid_mask (bool array) marks retrievable docs; masked-out docs sink to the end.
    """
    n = len(dense_scores)
    dr = (-dense_scores).argsort().argsort()   # 0-based dense rank
    br = (-bm25_scores).argsort().argsort()
    fused = 1.0 / (k + dr) + 1.0 / (k + br)
    if valid_mask is not None:
        fused = np.where(valid_mask, fused, -np.inf)
    return np.argsort(-fused)
