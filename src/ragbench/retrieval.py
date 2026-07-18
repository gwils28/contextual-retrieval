"""Retrievers over a list of chunk texts: dense, BM25, hybrid (RRF) and cross-encoder reranking.

All retrievers return chunk indices, best first. ``allowed`` optionally restricts the search to a
subset of indices (metadata pre-filter, e.g. the chunks of one company).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ragbench.fusion import rrf_fuse
from ragbench.lexical import BM25, tokenize

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    """Exact cosine search over normalized bi-encoder embeddings (no ANN approximation)."""

    def __init__(self, model, texts: Sequence[str], batch_size: int = 64, embeddings=None):
        self.model = model
        if embeddings is None:
            embeddings = model.encode(
                list(texts),
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        q = self.model.encode(
            BGE_QUERY_PREFIX + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores = self.embeddings @ q.astype(np.float32)
        if allowed is not None:
            masked = np.full_like(scores, -np.inf)
            masked[allowed] = scores[allowed]
            scores = masked
        k = min(k, len(scores) if allowed is None else len(allowed))
        top = np.argpartition(-scores, k - 1)[:k]
        return [int(i) for i in top[np.argsort(-scores[top], kind="stable")]]


class BM25Retriever:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.bm25 = BM25([tokenize(t) for t in texts], k1=k1, b=b)

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[int]:
        """Top-k documents with a strictly positive score (zero-score 'rankings' are noise)."""
        scores = np.asarray(self.bm25.get_scores(tokenize(query)))
        if allowed is not None:
            masked = np.zeros_like(scores)
            masked[allowed] = scores[allowed]
            scores = masked
        positive = np.flatnonzero(scores > 0)
        order = positive[np.argsort(-scores[positive], kind="stable")]
        return [int(i) for i in order[:k]]


def hybrid_search(
    dense: DenseRetriever,
    bm25: BM25Retriever,
    query: str,
    k: int,
    depth: int = 100,
    rrf_k: int = 60,
    allowed: np.ndarray | None = None,
) -> list[int]:
    """RRF fusion of dense top-``depth`` and BM25 top-``depth``."""
    lists = [dense.search(query, depth, allowed), bm25.search(query, depth, allowed)]
    return [int(i) for i, _ in rrf_fuse(lists, k=rrf_k, top_n=k)]


class Reranker:
    """Cross-encoder scoring of (query, chunk) pairs."""

    def __init__(self, model, texts: Sequence[str], batch_size: int = 32):
        self.model = model
        self.texts = texts
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[int], k: int) -> list[int]:
        if not candidates:
            return []
        pairs = [(query, self.texts[i]) for i in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        order = np.argsort(-np.asarray(scores), kind="stable")
        return [int(candidates[i]) for i in order[:k]]
