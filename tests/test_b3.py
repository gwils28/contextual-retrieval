import pytest

from ragbench import b3, finqa


def q(i, gold):
    return finqa.Question(f"q{i}", f"P/{i}", "P", "?", gold, "", "")


def test_draw_questions_disjoint_and_stratified():
    qs = [q(i, ("table_1",)) for i in range(60)] + [q(i, ("text_0",)) for i in range(60, 90)]
    sets = b3.draw_questions(qs, n=10, seed=0)
    ev, dev = ({x.qid for x in sets[r]} for r in ("evaluation", "development"))
    assert len(ev) == len(dev) == 10
    assert not ev & dev
    assert sum(x.gold_keys == ("table_1",) for x in sets["evaluation"]) == 7


def test_draw_questions_is_seeded():
    qs = [q(i, ("table_1",)) for i in range(50)]
    assert b3.draw_questions(qs, n=5) == b3.draw_questions(qs, n=5)


def test_frozen_question_list_matches_the_code():
    path = b3.RAW / "test.json"
    if not path.exists():
        pytest.skip("FinQA test split not downloaded")
    _, questions = finqa.load(path)
    drawn = {role: [x.qid for x in qs] for role, qs in b3.draw_questions(questions).items()}
    assert drawn == b3.load_questions()


def fixed_row(status="answer", grounded=True, **cost):
    return {"qid": "q", "status": status, "grounded": grounded, "correct": True,
            "llm_calls": 1, "prompt_tokens": 100, "llm_seconds": 1.0, **cost}


def agentic_row(**cost):
    return {"qid": "q", "status": "insufficient_context", "grounded": False, "correct": False,
            "llm_calls": 7, "prompt_tokens": 900, "llm_seconds": 8.0, "tool_calls": 6, **cost}


def test_cascade_threshold_flags_the_least_confident_share():
    scores = [i / 100 for i in range(100)]
    t = b3.cascade_threshold(scores, 0.2)
    assert sum(s < t for s in scores) == 20


@pytest.mark.parametrize(
    ("row", "top1", "expected"),
    [
        (fixed_row(), 0.9, []),
        (fixed_row(status="insufficient_context", grounded=False), 0.9, ["refusal"]),
        (fixed_row(grounded=False), 0.9, ["ungrounded"]),
        (fixed_row(status="invalid", grounded=False), 0.9, ["ungrounded"]),
        (fixed_row(), 0.4, ["low_score"]),
        (fixed_row(grounded=False), 0.4, ["ungrounded", "low_score"]),
        (fixed_row(), 0.5, []),  # strict: a score equal to the threshold does not escalate
    ],
)
def test_escalation_reasons(row, top1, expected):
    assert b3.escalation_reasons(row, top1, threshold=0.5) == expected


def test_cascade_keeps_fixed_when_no_signal_fires():
    row = b3.cascade_row(fixed_row(), agentic_row(), [])
    assert row["correct"] and row["llm_calls"] == 1 and row["tool_calls"] == 0
    assert not row["escalated"]


def test_cascade_takes_the_agent_answer_even_a_refusal_and_pays_both():
    row = b3.cascade_row(fixed_row(), agentic_row(), ["low_score"])
    assert row["status"] == "insufficient_context" and not row["correct"]
    assert (row["llm_calls"], row["prompt_tokens"], row["llm_seconds"]) == (8, 1000, 9.0)
    assert row["escalated"] and row["escalation_reasons"] == ["low_score"]


def test_cascade_drops_a_cost_measured_for_one_stage_only():
    row = b3.cascade_row(fixed_row(), agentic_row(generation_s=8.5), ["refusal"])
    assert "generation_s" not in row


def report_frame(correct, wrong=None, hit=None):
    import pandas as pd

    n = len(correct)
    wrong = wrong if wrong is not None else [not c for c in correct]
    hit = hit if hit is not None else [True] * n
    return pd.DataFrame({"correct": correct, "wrong_answer": wrong, "pass_all_ctx": hit,
                         "refusal": [False] * n}, index=[f"q{i}" for i in range(n)])


def test_diagnosis_puts_each_question_in_one_cell():
    df = report_frame([True, False, True, False], hit=[True, True, False, False])
    assert b3.diagnosis(df) == {"hit_correct": 1, "hit_wrong": 1, "miss_correct": 1,
                                "miss_wrong": 1}


def test_paired_tests_cover_three_pairs_on_two_metrics_with_holm():
    systems = {
        "fixed": report_frame([True] * 10 + [False] * 10),
        "agentic": report_frame([False] * 10 + [True] * 10),
        "cascade": report_frame([True] * 20),
    }
    tests = b3.paired_tests(systems)
    assert len(tests) == 6
    row = tests[(tests.metric == "correct") & (tests.a == "fixed") & (tests.b == "cascade")].iloc[0]
    assert (row.a_only, row.b_only, row.delta) == (0, 10, 0.5)
    assert (tests.p_holm >= tests.p_value).all()


def test_paired_tests_align_questions_by_id():
    fixed = report_frame([True, False])
    shuffled = report_frame([True, False]).iloc[::-1]
    tests = b3.paired_tests({"fixed": fixed, "agentic": shuffled, "cascade": fixed})
    assert (tests.a_only + tests.b_only == 0).all()


def test_small_strata_get_no_ci():
    import pandas as pd

    df = report_frame([True, False] * 15)
    strata = pd.DataFrame({"s": ["big"] * 25 + ["small"] * 5}, index=df.index)
    out = b3.strata_table({"fixed": df}, strata).set_index("level")
    assert out.loc["big", "n"] == 25 and not pd.isna(out.loc["big", "exec_accuracy_lo"])
    assert out.loc["small", "n"] == 5 and pd.isna(out.loc["small", "exec_accuracy_lo"])


def test_layer_economics_gives_no_cost_per_point_without_a_gain():
    import pandas as pd

    configs = ["dense", "hybrid", "hybrid_rerank", "contextual_hybrid_rerank"]
    b2 = pd.DataFrame({"strategy": "naive", "setting": "main", "config": configs,
                       "pass_all@10": [0.7, 0.75, 0.85, 0.95],
                       "latency_p50_ms": [5.0, 10.0, 300.0, 300.0]})
    summary = pd.DataFrame({
        "system": ["fixed", "agentic"], "exec_accuracy": [0.4, 0.3],
        "prompt_tokens_mean": [1000.0, 9000.0], "completion_tokens_mean": [50.0, 250.0],
        "latency_p50_ms": [2000.0, 9000.0], "usd_per_1000_questions": [0.3, 2.8],
    })
    out = b3.layer_economics(b2, summary, n_chunks=7).set_index("layer")
    assert round(out.loc["hybrid", "gain_points"], 6) == 5.0
    assert round(out.loc["hybrid", "latency_ms_per_point"], 6) == 1.0
    assert out.loc["contextual_chunks", "one_time_cost"] == "one LLM call per chunk (7 calls)"
    assert round(out.loc["agentic_loop", "gain_points"], 6) == -10.0
    assert out.loc["agentic_loop", "llm_tokens_added"] == 8200.0
    assert pd.isna(out.loc["agentic_loop", "usd_per_1000_per_point"])


def test_exploratory_compares_both_systems_on_the_same_questions():
    import pandas as pd

    idx = ["q0", "q1", "q2"]
    fixed = pd.DataFrame({"status": ["answer", "answer", "insufficient_context"],
                          "grounded": [True, False, False], "correct": [True, False, False]},
                         index=idx)
    agentic = pd.DataFrame({"stop": ["done", "tool_budget", "done"],
                            "correct": [False, False, True]}, index=idx)
    cascade = pd.DataFrame({"escalated": [False, True, True],
                            "escalation_reasons": [[], ["ungrounded"], ["refusal", "low_score"]]},
                           index=idx)
    out = b3.exploratory({"fixed": fixed, "agentic": agentic, "cascade": cascade})
    out = out.set_index(["group", "level"])
    assert out.loc[("agent_stop", "done"), "n"] == 2
    assert out.loc[("agent_stop", "done"), "agentic_exec_accuracy"] == 0.5
    assert out.loc[("fixed_answer", "ungrounded"), "n"] == 1
    assert out.loc[("cascade", "escalated"), "agentic_exec_accuracy"] == 0.5
    assert out.loc[("escalation_signal", "low_score"), "n"] == 1
