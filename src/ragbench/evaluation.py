"""Retrieval metrics and paired statistics. Retrieval only: no generated answer is ever looked at.

A question has one or more *gold facts*; each fact maps to a set of acceptable chunk indices.
A fact is "found" in a ranked list if any of its acceptable chunks is in it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest

GoldSets = Sequence[frozenset[int]]


def _found(prefix: Sequence[int], gold: GoldSets) -> list[bool]:
    s = set(prefix)
    return [bool(g & s) for g in gold]


def pass_all(ranked: Sequence[int], gold: GoldSets, k: int) -> float:
    """1.0 if every gold fact is in the top-k."""
    return float(all(_found(ranked[:k], gold)))


def pass_any(ranked: Sequence[int], gold: GoldSets, k: int) -> float:
    """1.0 if at least one gold fact is in the top-k."""
    return float(any(_found(ranked[:k], gold)))


def recall(ranked: Sequence[int], gold: GoldSets, k: int) -> float:
    """Fraction of gold facts in the top-k."""
    found = _found(ranked[:k], gold)
    return sum(found) / len(found)


def mrr(ranked: Sequence[int], gold: GoldSets, k: int = 10) -> float:
    """Reciprocal rank of the first chunk belonging to any gold fact (0 if none in top-k)."""
    relevant = frozenset().union(*gold)
    for rank, idx in enumerate(ranked[:k], start=1):
        if idx in relevant:
            return 1.0 / rank
    return 0.0


def budget_prefix(ranked: Sequence[int], lengths: Sequence[int], budget: int) -> list[int]:
    """Longest prefix of the ranking whose total token length fits in ``budget``."""
    out, used = [], 0
    for idx in ranked:
        if used + lengths[idx] > budget:
            break
        out.append(idx)
        used += lengths[idx]
    return out


def pass_all_budget(
    ranked: Sequence[int], gold: GoldSets, lengths: Sequence[int], budget: int
) -> float:
    return float(all(_found(budget_prefix(ranked, lengths, budget), gold)))


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 10_000, level: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling queries."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def mcnemar_exact(a: Sequence[float], b: Sequence[float]) -> dict:
    """Exact McNemar test on paired binary outcomes (same queries, two systems).

    Only discordant queries carry information: ``b_only`` = B succeeds where A fails.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(min(a_only, b_only), n, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "p_value": float(p)}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
