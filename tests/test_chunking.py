import re

import pytest

from ragbench.chunking import chunk_naive, chunk_table_aware, is_headerless, serialize_row
from ragbench.finqa import Page


class WordTokenizer:
    """Stand-in for a HF fast tokenizer: one token per non-space run, with offsets."""

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


def count_words(text: str) -> int:
    return len(text.split())


PAGE = Page(
    page_id="ACME/2019/page_1.pdf",
    ticker="ACME",
    pre_text=("revenue grew strongly .", "costs were flat .", "see table below ."),
    table=(
        ("", "2019", "2018"),
        ("net revenue", "$ 120", "$ 100"),
        ("operating costs", "$ 80", "$ 80"),
    ),
    post_text=("margins improved .",),
)


def test_serialize_row_attaches_headers():
    assert serialize_row(PAGE.table[0], PAGE.table[1]) == "net revenue | 2019: $ 120 | 2018: $ 100"


def test_table_aware_one_chunk_per_row_and_exact_mapping():
    corpus = chunk_table_aware([PAGE], count_words, window_tokens=8)  # 4 + 4 words fit
    texts = [c.text for c in corpus.chunks]
    assert "net revenue | 2019: $ 120 | 2018: $ 100" in texts
    m = corpus.gold_map[PAGE.page_id]
    assert corpus.chunks[min(m["table_1"])].text.startswith("net revenue")
    assert corpus.chunks[min(m["text_3"])].text == "margins improved ."  # post_text index = 3
    # Sentences are grouped up to the budget, never across the table.
    assert corpus.chunks[min(m["text_0"])].text == "revenue grew strongly . costs were flat ."


def test_table_aware_header_gold_accepts_any_row():
    corpus = chunk_table_aware([PAGE], count_words)
    m = corpus.gold_map[PAGE.page_id]
    assert m["table_0"] == m["table_1"] | m["table_2"]


def test_naive_windows_cut_through_the_table():
    corpus = chunk_naive([PAGE], WordTokenizer(), window_tokens=12)
    kinds = [c.kind for c in corpus.chunks]
    assert "mixed" in kinds  # a window holds both prose and table cells
    # Every unit is mapped to exactly one chunk.
    m = corpus.gold_map[PAGE.page_id]
    assert set(m) == {"text_0", "text_1", "text_2", "text_3", "table_0", "table_1", "table_2"}
    assert all(len(v) == 1 for v in m.values())


def test_naive_gold_goes_to_window_with_largest_overlap():
    corpus = chunk_naive([PAGE], WordTokenizer(), window_tokens=12)
    m = corpus.gold_map[PAGE.page_id]
    row = "net revenue | $ 120 | $ 100"
    chunk = corpus.chunks[min(m["table_1"])].text
    shared = sum(1 for w in row.split() if w in chunk.split())
    assert shared >= len(row.split()) / 2


def test_gold_sets_merge_duplicates():
    corpus = chunk_table_aware([PAGE], count_words, window_tokens=100)
    sets = corpus.gold_sets(PAGE.page_id, ["text_0", "text_1"])  # same sentence group
    assert len(sets) == 1


@pytest.mark.parametrize("window", [3, 7, 50])
def test_naive_chunks_cover_all_tokens(window):
    corpus = chunk_naive([PAGE], WordTokenizer(), window_tokens=window)
    words = sum(len(c.text.split()) for c in corpus.chunks)
    total = len(" ".join(PAGE.pre_text + PAGE.post_text).split()) + sum(
        len(" | ".join(r).split()) for r in PAGE.table
    )
    assert words == total


def test_headerless_detection_keeps_year_headers():
    assert not is_headerless((("", "2019", "2018"), ("sales", "$ 5", "$ 4")))
    assert is_headerless((("cash", "$ 108"), ("other current assets", "28")))
    assert not is_headerless((("( in millions )", "us", "intl"), ("a", "1", "2")))


def test_headerless_table_gets_one_chunk_per_row_including_row_0():
    page = Page("X/2010/page_1.pdf", "X", (), (("cash", "$ 108"), ("other assets", "28")), ())
    corpus = chunk_table_aware([page], count_words)
    m = corpus.gold_map[page.page_id]
    assert corpus.chunks[min(m["table_0"])].text == "cash | $ 108"
    assert corpus.chunks[min(m["table_1"])].text == "other assets | 28"
    assert len(m["table_0"]) == 1
