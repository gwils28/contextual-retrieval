"""Error analysis over per-query retrieval results. Retrieval only: no generated answer is read.

``rescues`` is a protocol output (queries where one config succeeds and another fails).
``failure_mode`` is exploratory: it was defined after the results were seen, so it describes
failures but is never used to rank configs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

GoldSets = Sequence[frozenset[int]]

FAILURE_MODES = (
    "hit",
    "partial",
    "other_company",
    "same_company_other_page",
    "same_page_other_chunk",
)


def rescues(
    per_query: pd.DataFrame, winner: str, loser: str, metric: str = "pass_all@10"
) -> pd.DataFrame:
    """Queries where ``winner`` succeeds and ``loser`` fails, per strategy and setting."""
    keys = ["strategy", "setting", "qid"]
    pivot = per_query[per_query.config.isin([winner, loser])].pivot_table(
        index=keys, columns="config", values=metric
    )
    hit = pivot[(pivot[winner] == 1.0) & (pivot[loser] == 0.0)]
    return hit.reset_index()[keys]


def failure_mode(
    ranked: Sequence[int],
    gold: GoldSets,
    chunk_page: Sequence[str],
    page_company: Mapping[str, str],
    q_page: str,
    k: int = 10,
) -> str:
    """Why a query misses at k, judged from the top-1 chunk when no gold fact was found.

    hit: every gold fact in the top-k. partial: some but not all. Otherwise the top-1 chunk
    comes from another company, from another page (year) of the same company, or from the
    right page but not a gold chunk.
    """
    prefix = set(ranked[:k])
    found = [bool(g & prefix) for g in gold]
    if all(found):
        return "hit"
    if any(found):
        return "partial"
    top_page = chunk_page[ranked[0]]
    if page_company[top_page] != page_company[q_page]:
        return "other_company"
    if top_page != q_page:
        return "same_company_other_page"
    return "same_page_other_chunk"
