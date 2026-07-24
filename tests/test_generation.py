import json

import pytest

from ragbench import generation as G

TEXTS = [
    "Revenue was $ 5,829 million in 2017 and $ 5,735 million in 2016.",
    "net income | 1,200 | 900\ndividends | 100 | 80",
]
TABLE = (("", "2017", "2016"), ("net income", "1,200", "900"), ("dividends", "100", "80"))
TABLES = [(), TABLE]


SUBTRACT_STEP = {"op": "subtract", "arg1": "5829", "arg1_chunk": 1, "arg2": "5735", "arg2_chunk": 1}


def answer(program, operands, status="answer"):
    return G.Answer(status, program, operands)


def test_parse_answer_valid_and_invalid():
    content = json.dumps({"status": "answer", "steps": [SUBTRACT_STEP]})
    a = G.parse_answer(content)
    assert a.status == "answer" and a.operands == [("5829", 1), ("5735", 1)]
    assert a.program == "subtract(5829, 5735)"
    assert G.parse_answer("not json").status == "invalid"
    assert G.parse_answer('{"status": "maybe", "steps": []}').status == "invalid"
    assert G.parse_answer('{"steps": []}').status == "invalid"
    assert G.parse_answer('{"status": "answer", "steps": [{"op": "add"}]}').status == "invalid"


def test_parse_answer_assembles_chained_steps_and_skips_uncited_arguments():
    steps = [
        SUBTRACT_STEP,
        {"op": "divide", "arg1": "#0", "arg1_chunk": 0, "arg2": "5735", "arg2_chunk": 1},
    ]
    a = G.parse_answer(json.dumps({"status": "answer", "steps": steps}))
    assert a.program == "subtract(5829, 5735), divide(#0, 5735)"
    assert a.operands == [("5829", 1), ("5735", 1), ("5735", 1)]
    assert G.is_grounded(a, TEXTS)


def test_number_with_chunk_zero_is_not_grounded():
    step = {"op": "add", "arg1": "5829", "arg1_chunk": 1, "arg2": "77", "arg2_chunk": 0}
    a = G.parse_answer(json.dumps({"status": "answer", "steps": [step]}))
    assert not G.is_grounded(a, TEXTS)


def test_format_chunks_uses_one_based_ids():
    assert G.format_chunks(["a", "b"]) == '<chunk id="1">\na\n</chunk>\n<chunk id="2">\nb\n</chunk>'


def test_numbers_in_ignores_signs_currency_and_commas():
    assert G.numbers_in("$ 5,829 and ( 3.2 ) and 12%") == {5829.0, 3.2, 12.0}


def test_grounded_when_every_literal_is_cited_from_a_chunk_that_contains_it():
    a = answer("subtract(5829, 5735), divide(#0, 5735)", [("5829", 1), ("5735", 1)])
    assert G.is_grounded(a, TEXTS)


def test_constants_need_no_operand():
    assert G.is_grounded(answer("multiply(5829, const_100)", [("5829", 1)]), TEXTS)


def test_not_grounded_when_a_number_is_invented_or_miscited():
    assert not G.is_grounded(answer("subtract(5829, 5700)", [("5829", 1), ("5700", 1)]), TEXTS)
    assert not G.is_grounded(answer("subtract(5829, 5735)", [("5829", 1)]), TEXTS)  # uncited
    assert not G.is_grounded(answer("add(5829, 100)", [("5829", 1), ("100", 1)]), TEXTS)
    assert not G.is_grounded(answer("add(5829, 1)", [("5829", 9)]), TEXTS)  # no chunk 9


def test_table_row_label_must_be_cited():
    assert G.is_grounded(answer("table_sum(net income, none)", [("net income", 2)]), TEXTS)
    assert not G.is_grounded(answer("table_sum(net income, none)", []), TEXTS)


def test_score_correct_answer():
    a = answer("subtract(5829, 5735), divide(#0, 5735)", [("5829", 1), ("5735", 1)])
    s = G.score_answer(a, TEXTS, TABLES, "0.01639")
    assert s.program_valid and s.correct and s.grounded and not s.wrong_answer


def test_score_table_step_runs_on_the_cited_chunk_table():
    a = answer("table_sum(net income, none)", [("net income", 2)])
    s = G.score_answer(a, TEXTS, TABLES, "2100")
    assert s.correct and s.value == 2100.0


def test_score_wrong_and_invalid_answers():
    a = answer("add(5829, 5735)", [("5829", 1), ("5735", 1)])
    wrong = G.score_answer(a, TEXTS, TABLES, "94")
    assert wrong.wrong_answer and wrong.grounded
    broken = G.score_answer(answer("divide(1, 0)", []), TEXTS, TABLES, "1")
    assert not broken.program_valid and broken.error and not broken.wrong_answer


def test_refusal_is_not_a_wrong_answer():
    s = G.score_answer(answer("", [], status="insufficient_context"), TEXTS, TABLES, "1")
    assert s.status == "insufficient_context" and not s.wrong_answer and not s.correct


class FakeChat:
    name = "fake"

    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, messages, tools=None, schema=None):
        self.calls.append((messages, schema))
        return G.ChatResult({"role": "assistant", "content": self.content}, 120, 30, 0.01)


def test_run_question_with_a_fake_llm():
    from ragbench import b3, finqa

    class FakeCorpus:
        texts = TEXTS
        bench = type("B", (), {"count": staticmethod(lambda t: len(t.split()))})()

        def query(self, q):
            return f"Acme: {q.question}"

        def gold(self, q):
            return [frozenset({0})]

        def table(self, i):
            return TABLES[i]

    content = json.dumps({"status": "answer", "steps": [SUBTRACT_STEP]})
    llm = FakeChat(content)
    q = finqa.Question("q1", "P/1", "P", "change in revenue?", ("text_0",), "", "94")
    row = b3.run_question(llm, FakeCorpus(), q, [0, 1])
    assert row["correct"] and row["grounded"] and row["pass_all_ctx"] == 1.0
    assert row["prompt_tokens"] == 120 and row["llm_calls"] == 1
    messages, schema = llm.calls[0]
    assert schema == G.ANSWER_SCHEMA
    user = messages[1]["content"]
    assert '<chunk id="2">\nnet income' in user and "ids 1 to 2" in user and "Acme:" in user


@pytest.mark.parametrize("system,expected", [("oracle_context", [3, 7]), ("fixed", [9, 3])])
def test_context_for(system, expected):
    from ragbench import b3

    corpus = type("C", (), {"gold": lambda self, q: [frozenset({7}), frozenset({3})]})()
    assert b3.context_for(system, corpus, None, [9, 3]) == expected
