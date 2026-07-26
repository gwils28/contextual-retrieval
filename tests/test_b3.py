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
