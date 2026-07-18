import pandas as pd

from ragbench.analysis import failure_mode, rescues

# Chunks 0-1 on page A (ACME 2015), 2 on page B (ACME 2016), 3 on page C (Globex).
CHUNK_PAGE = ["A", "A", "B", "C"]
COMPANY = {"A": "ACME", "B": "ACME", "C": "Globex"}


def test_failure_mode_hit_and_partial():
    gold = [frozenset({0}), frozenset({1})]
    assert failure_mode([1, 0, 3], gold, CHUNK_PAGE, COMPANY, "A", k=2) == "hit"
    assert failure_mode([3, 0, 1], gold, CHUNK_PAGE, COMPANY, "A", k=2) == "partial"


def test_failure_mode_is_judged_on_top1_when_nothing_found():
    gold = [frozenset({0})]
    assert failure_mode([3, 2], gold, CHUNK_PAGE, COMPANY, "A", k=2) == "other_company"
    assert failure_mode([2, 3], gold, CHUNK_PAGE, COMPANY, "A", k=2) == "same_company_other_page"
    assert failure_mode([1, 3], gold, CHUNK_PAGE, COMPANY, "A", k=2) == "same_page_other_chunk"


def test_rescues_keeps_only_winner_hit_loser_miss():
    rows = [
        ("q1", "bm25", 1.0), ("q1", "dense", 0.0),  # rescue
        ("q2", "bm25", 1.0), ("q2", "dense", 1.0),  # both hit
        ("q3", "bm25", 0.0), ("q3", "dense", 1.0),  # reverse
        ("q4", "hybrid", 1.0), ("q4", "dense", 0.0),  # other config, ignored
    ]
    df = pd.DataFrame(rows, columns=["qid", "config", "pass_all@10"])
    df["strategy"], df["setting"] = "naive", "main"
    out = rescues(df, winner="bm25", loser="dense")
    assert out.qid.tolist() == ["q1"]
