import math

from ragbench import b4


def step(tool, results, added=()):
    return {"tool": tool, "args": {}, "results": list(results), "added": list(added)}


def test_process_metrics_replays_the_pool():
    trace = [
        step("keyword_search", [1, 2, 7]),       # 7 is new: a useful search
        step("chunk_read", [1]),                 # in the initial top-10: a re-read
        step("chunk_read", [7], added=[7]),      # adds a chunk
        step("keyword_search", [1, 7]),          # everything already in the pool
        step("page_read", [7, 8, 9], added=[8, 9]),
        step("chunk_read", [8]),                 # empty, but not part of the initial top-10
    ]
    m = b4.process_metrics(trace, initial=[1, 2, 3])
    assert m["tool_calls"] == 6
    assert (m["calls_keyword_search"], m["calls_chunk_read"], m["calls_page_read"]) == (2, 3, 1)
    assert m["search_without_new_candidate"] == 1
    assert m["empty_read"] == 2
    assert m["reread_of_initial"] == 1
    assert m["chunks_added"] == 3


def test_unknown_tools_and_ids_are_invalid():
    trace = [step("DONE", []), step("chunk_read", []), step("semantic_search", [5])]
    m = b4.process_metrics(trace, initial=[1])
    assert m["calls_invalid"] == 2 and m["calls_semantic_search"] == 1
    assert m["empty_read"] == 0  # an invalid read is not counted twice


def test_summarize_shares_use_the_relevant_calls():
    q1 = b4.process_metrics([step("chunk_read", [1]), step("chunk_read", [4], [4])], [1])
    q2 = b4.process_metrics([step("keyword_search", [1])], [1])
    s = b4.summarize([q1, q2])
    assert s["questions"] == 2 and s["tool_calls_mean"] == 1.5
    assert s["empty_read_share"] == 0.5
    assert s["reread_of_initial_share"] == 0.5
    assert s["search_without_new_candidate_share"] == 1.0
    assert math.isclose(s["chunks_added_mean"], 0.5)
