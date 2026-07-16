import math

import pytest
from rank_bm25 import BM25Okapi

from ragbench.lexical import BM25, tokenize

CORPUS = [
    "Net revenue grew 3% in Q2 2019 driven by the Americas segment.",
    "Operating expenses increased due to higher amortization of intangible assets.",
    "The company repurchased 2.1 million shares under its buyback program.",
    "Revenue from the EMEA segment declined while Americas revenue grew.",
    "Goodwill impairment of $45 million was recorded in the fourth quarter.",
    "The effective tax rate was 21% compared with 24% in the prior year.",
]
TOKENS = [tokenize(d) for d in CORPUS]


def test_tokenize_lowercases_and_splits():
    assert tokenize("Net Revenue, Q2-2019!") == ["net", "revenue", "q2", "2019"]


@pytest.mark.parametrize(
    "query", ["americas revenue", "goodwill impairment", "tax rate prior year", "unknown"]
)
def test_matches_rank_bm25_reference(query):
    ours = BM25(TOKENS).get_scores(tokenize(query))
    ref = BM25Okapi(TOKENS).get_scores(tokenize(query))
    assert ours == pytest.approx(list(ref))


def test_rare_term_has_higher_idf_than_common_term():
    bm25 = BM25(TOKENS)
    assert bm25.idf["goodwill"] > bm25.idf["revenue"]


def test_idf_formula():
    bm25 = BM25(TOKENS)
    n_t = 1  # "goodwill" appears in one document
    assert bm25.idf["goodwill"] == pytest.approx(math.log((6 - n_t + 0.5) / (n_t + 0.5)))


def test_term_frequency_saturates():
    # "fee" must stay rare (< half the documents) so its IDF is positive.
    filler = [[f"w{i}"] * 10 for i in range(7)]
    docs = [["fee"] * 1 + ["x"] * 9, ["fee"] * 5 + ["x"] * 5, ["fee"] * 9 + ["x"], *filler]
    s = BM25(docs).get_scores(["fee"])
    assert s[0] < s[1] < s[2]
    assert (s[2] - s[1]) < (s[1] - s[0])  # diminishing returns


def test_length_normalization_penalizes_long_documents():
    filler = [[f"w{i}"] * 5 for i in range(4)]
    docs = [["fee", "a"], ["fee"] + ["a"] * 20, *filler]
    short, long_, *_ = BM25(docs, b=0.75).get_scores(["fee"])
    assert short > long_
    short0, long0, *_ = BM25(docs, b=0.0).get_scores(["fee"])
    assert short0 == pytest.approx(long0)


def test_top_k_returns_best_first():
    bm25 = BM25(TOKENS)
    top = bm25.top_k(tokenize("goodwill impairment"), k=2)
    assert top[0][0] == 4
    assert top[0][1] > top[1][1]


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25([])


def test_idf_floor_can_be_negative_on_tiny_corpora():
    # Known rank_bm25 quirk we reproduce on purpose: when most terms are frequent, the mean
    # IDF is negative, so the "floor" epsilon * mean(IDF) is negative too.
    docs = [["fee", "a"], ["fee", "a"], ["c"]]  # 2 of 3 terms are in 2/3 of the docs
    ours = BM25(docs).get_scores(["fee"])
    assert ours == pytest.approx(list(BM25Okapi(docs).get_scores(["fee"])))
    assert ours[0] < 0
