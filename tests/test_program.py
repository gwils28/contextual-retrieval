import json
from pathlib import Path

import pytest

from ragbench import program as P

TEST_JSON = Path(__file__).resolve().parents[1] / "data" / "raw" / "finqa" / "test.json"

TABLE = (
    ("", "2012", "2011"),
    ("net income, as reported", "$ 1,200", "$ 900 ( 3% )"),
    ("dividends", "100", "n/a"),
)


def test_chained_steps_and_references():
    assert P.execute("subtract(5829, 5735), divide(#0, 5735)") == pytest.approx(94 / 5735)


def test_constants_and_percentages():
    assert P.execute("multiply(const_100, 5%)") == pytest.approx(5.0)
    assert P.execute("add(const_m1, const_1000000)") == 999999.0


def test_greater_returns_yes_no():
    assert P.execute("greater(286.61, 198.09)") == "yes"
    assert P.execute("greater(1, 2)") == "no"


def test_quotes_around_arguments_are_ignored():
    assert P.execute("table_sum('net income, as reported', none)", TABLE) == 2100.0
    assert P.execute('add("1", 2)') == 3.0


def test_exp():
    assert P.execute("exp(2, 10)") == 1024.0


def test_table_ops_use_the_row_label_with_commas():
    assert P.execute("table_sum(net income, as reported, none)", TABLE) == 2100.0
    assert P.execute("table_average(net income, as reported, none)", TABLE) == 1050.0
    assert P.execute("table_max(net income, as reported, none)", TABLE) == 1200.0


@pytest.mark.parametrize(
    "program",
    [
        "",
        "modulo(1, 2)",
        "add(1)",
        "divide(1, 0)",
        "add(#3, 1)",
        "add(#const_1336, 1)",  # seen in a Qwen output
        "add(const_1336, 1)",  # not a FinQA constant
        "exp(-8, 0.5)",  # complex result
        "exp(0, -1)",
        "add(apples, 1)",
        "table_sum(missing row, none)",
        "table_sum(dividends, none)",  # "n/a" cell
        "greater(1, 2), add(#0, 1)",  # yes/no is not a number
    ],
)
def test_invalid_programs_raise(program):
    with pytest.raises(P.ProgramError):
        P.execute(program, TABLE)


def test_answers_match_tolerance_and_percent_scale():
    assert P.answers_match(100.4, 100.0)
    assert not P.answers_match(101.0, 100.0)
    assert P.answers_match(0.001, 0.0)  # tolerance floor of 1 * rel_tol
    assert P.answers_match(25.0, 0.25) and P.answers_match(0.25, 25.0)
    assert P.answers_match("yes", "yes") and not P.answers_match(1.0, "yes")
    assert not P.answers_match("no", 0.0)


def test_gold_programs_reproduce_finqa_answers():
    if not TEST_JSON.exists():
        pytest.skip("FinQA test split not downloaded")
    for r in json.loads(TEST_JSON.read_text()):
        table = tuple(tuple(row) for row in r["table"])
        value = P.execute(r["qa"]["program"], table)
        gold = r["qa"]["exe_ans"]
        if isinstance(gold, str):
            assert value == gold
        else:
            assert value == pytest.approx(gold, rel=1e-4, abs=1e-4), r["id"]
