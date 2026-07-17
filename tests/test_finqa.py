from pathlib import Path

from ragbench.finqa import company_name, load_company_names

NAMES = load_company_names(Path(__file__).parents[1] / "configs" / "finqa_tickers.csv")


def test_every_ticker_has_exactly_one_open_ended_name():
    for ticker, rows in NAMES.items():
        assert sum(until is None for until, _ in rows) == 1, ticker


def test_renamed_company_depends_on_year():
    assert company_name(NAMES, "HWM/2015/page_1.pdf") == "Alcoa"
    assert company_name(NAMES, "HWM/2017/page_1.pdf") == "Arconic"


def test_plain_company():
    assert company_name(NAMES, "ETR/2016/page_23.pdf") == "Entergy"
