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
