"""Rank fusion: merge several ranked lists into one.

Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009):

    RRF(d) = sum_{lists L containing d} 1 / (k + rank_L(d))      ranks start at 1

RRF only uses *ranks*, never raw scores. That is why it can fuse BM25 scores (unbounded,
roughly 0..30) with cosine similarities (-1..1) without any normalization. The constant
k (60 in the original paper) dampens the advantage of the very top ranks: with k=60, rank 1
is worth 1/61 and rank 10 is worth 1/70, so a document that is "pretty good everywhere" can
beat one that is "first in one list and absent elsewhere".
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence


def rrf_fuse(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: int = 60,
    top_n: int | None = None,
) -> list[tuple[Hashable, float]]:
    """Fuse ranked lists of document ids (best first) with Reciprocal Rank Fusion.

    Returns (doc_id, rrf_score) pairs sorted by decreasing score. Ties are broken by first
    appearance, so the output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[Hashable, float] = defaultdict(float)
    first_seen: dict[Hashable, int] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            first_seen.setdefault(doc_id, len(first_seen))
    fused = sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))
    return fused[:top_n] if top_n is not None else fused
