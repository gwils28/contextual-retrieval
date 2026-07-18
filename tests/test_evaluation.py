import pytest

from ragbench.evaluation import (
    bootstrap_ci,
    budget_prefix,
    holm,
    mcnemar_exact,
    mrr,
    pass_all,
    pass_all_budget,
    pass_any,
    recall,
)

# Two gold facts: fact A is chunk 3; fact B is acceptable as chunk 7 or 8.
GOLD = [frozenset({3}), frozenset({7, 8})]
RANKED = [5, 3, 9, 8, 1]


def test_pass_all_requires_every_fact():
    assert pass_all(RANKED, GOLD, 2) == 0.0  # only fact A found
    assert pass_all(RANKED, GOLD, 4) == 1.0  # chunk 8 satisfies fact B


def test_pass_any_and_recall():
    assert pass_any(RANKED, GOLD, 1) == 0.0
    assert pass_any(RANKED, GOLD, 2) == 1.0
    assert recall(RANKED, GOLD, 2) == 0.5
    assert recall(RANKED, GOLD, 5) == 1.0


def test_mrr_uses_first_relevant_rank():
    assert mrr(RANKED, GOLD) == pytest.approx(1 / 2)
    assert mrr([1, 2], GOLD) == 0.0


def test_budget_prefix_stops_before_overflow():
    lengths = {5: 100, 3: 100, 9: 300, 8: 50, 1: 10}
    assert budget_prefix(RANKED, lengths, 250) == [5, 3]  # 9 would overflow
    assert pass_all_budget(RANKED, GOLD, lengths, 250) == 0.0
    assert pass_all_budget(RANKED, GOLD, lengths, 550) == 1.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    values = [1.0] * 80 + [0.0] * 20
    lo, hi = bootstrap_ci(values, n_resamples=2000)
    assert lo < 0.8 < hi
    assert (lo, hi) == bootstrap_ci(values, n_resamples=2000)
    assert hi - lo == pytest.approx(2 * 1.96 * (0.8 * 0.2 / 100) ** 0.5, abs=0.03)


def test_mcnemar_counts_only_discordant_pairs():
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 0, 1, 1, 1, 1]
    r = mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (1, 3)
    assert r["p_value"] == pytest.approx(0.625)  # binomial(1; 4, 0.5) two-sided
    assert mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]
