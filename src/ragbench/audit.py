"""Stratified sample of questions for the hand audit of the gold-to-chunk mapping."""

from __future__ import annotations

import random

from ragbench.finqa import Question


def gold_type(q: Question) -> str:
    kinds = sorted({k.split("_")[0] for k in q.gold_keys})
    return "+".join(kinds)


def audited_sample(questions: list[Question], n: int = 25, seed: int = 0) -> list[Question]:
    """Proportional stratified sample by gold type (table / text / table+text), seed-fixed."""
    rng = random.Random(seed)
    strata: dict[str, list[Question]] = {}
    for q in questions:
        strata.setdefault(gold_type(q), []).append(q)
    total = len(questions)
    quotas = {k: max(1, round(n * len(v) / total)) for k, v in sorted(strata.items())}
    while sum(quotas.values()) > n:
        quotas[max(quotas, key=quotas.get)] -= 1
    while sum(quotas.values()) < n:
        quotas[max(quotas, key=quotas.get)] += 1
    sample = [q for k in sorted(strata) for q in rng.sample(strata[k], quotas[k])]
    return sorted(sample, key=lambda q: q.qid)
