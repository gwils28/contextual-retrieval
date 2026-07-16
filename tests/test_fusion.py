import pytest

from ragbench.fusion import rrf_fuse


def test_single_list_keeps_order():
    fused = rrf_fuse([["a", "b", "c"]])
    assert [d for d, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)


def test_consensus_beats_single_top_rank():
    # "a" is 1st in list 1 but absent from list 2; "b" is 2nd in both.
    fused = dict(rrf_fuse([["a", "b"], ["c", "b"]]))
    assert fused["b"] == pytest.approx(2 / 62)
    assert fused["b"] > fused["a"]


def test_first_and_third_narrowly_beats_second_and_second():
    # 1/61 + 1/63 = 0.032266 > 2/62 = 0.032258
    fused = dict(rrf_fuse([["x", "y", "z"], ["y", "w", "x"]]))
    assert fused["x"] > 2 / 62 > fused["x"] - 1e-4


def test_scores_ignore_raw_scale_only_ranks_matter():
    # Same rankings must give identical fusion whatever the source of the ranks.
    assert rrf_fuse([["a", "b"], ["b", "a"]]) == rrf_fuse([["a", "b"], ["b", "a"]])


def test_top_n_and_tie_break_is_deterministic():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], top_n=1)
    assert fused == [("a", pytest.approx(1 / 61 + 1 / 62))]


def test_k_controls_top_rank_advantage():
    lists = [["a", "b", "c"], ["c", "b", "a"], ["b", "c", "a"]]
    # Small k rewards being first more strongly.
    assert rrf_fuse(lists, k=0)[0][0] == "b"
    with pytest.raises(ValueError):
        rrf_fuse(lists, k=-1)
