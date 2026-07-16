"""Lexical retrieval: a from-scratch BM25 (Okapi) used for teaching and for the benchmark.

The scoring follows ``rank_bm25.BM25Okapi`` exactly (including its IDF floor), so the two can
be cross-checked in tests. Formula, for a query q and a document d:

    score(q, d) = sum_{t in q} IDF(t) * tf(t, d) * (k1 + 1)
                                 / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5))

- ``tf(t, d)``: how many times term t occurs in d.
- ``k1`` controls term-frequency *saturation*: the 10th occurrence of a term adds much less
  than the 1st.
- ``b`` controls *length normalization*: b=1 fully penalizes long documents, b=0 ignores length.
- A term that appears in more than half of the documents gets a negative raw IDF. Like
  rank_bm25, we replace it with ``epsilon * mean(IDF)`` so that very common words never
  *lower* a score. Caveat (also in rank_bm25): on tiny corpora where the mean IDF is itself
  negative, that floor is negative and frequent terms do lower scores. Real corpora with
  thousands of chunks are not affected.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters. Keeps numbers like '2019' or '3.5' parts."""
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        if not corpus:
            raise ValueError("corpus must contain at least one document")
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.doc_freqs: list[Counter[str]] = [Counter(doc) for doc in corpus]
        self.doc_len: list[int] = [len(doc) for doc in corpus]
        self.n_docs = len(corpus)
        self.avgdl = sum(self.doc_len) / self.n_docs
        self.idf = self._compute_idf()

    def _compute_idf(self) -> dict[str, float]:
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        idf = {
            term: math.log((self.n_docs - n + 0.5) / (n + 0.5)) for term, n in df.items()
        }
        floor = self.epsilon * (sum(idf.values()) / len(idf))
        return {term: (v if v >= 0 else floor) for term, v in idf.items()}

    def score(self, query: Sequence[str], doc_index: int) -> float:
        """BM25 score of one document for a tokenized query."""
        freqs = self.doc_freqs[doc_index]
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
        total = 0.0
        for term in query:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            total += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return total

    def get_scores(self, query: Sequence[str]) -> list[float]:
        """Scores of every document, in corpus order."""
        return [self.score(query, i) for i in range(self.n_docs)]

    def top_k(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        """The k best (doc_index, score) pairs, best first. Ties keep corpus order."""
        scores = self.get_scores(query)
        order = sorted(range(self.n_docs), key=lambda i: -scores[i])
        return [(i, scores[i]) for i in order[:k]]
