from ragbench import agent as A
from ragbench.generation import ChatResult

TEXTS = [f"chunk number {i} of page {i // 3}" for i in range(9)]


def box():
    return A.ToolBox(
        TEXTS,
        page_chunks=lambda i: [j for j in range(9) if j // 3 == i // 3],
        keyword=lambda q, k: [7, 8][:k],
        semantic=lambda q, k: [4][:k],
    )


def count(text):
    return len(text.split())


def call(name, **args):
    return {"function": {"name": name, "arguments": args}}


class ScriptedChat:
    """Replays assistant messages in order; then always answers DONE."""

    name = "scripted"

    def __init__(self, *turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools=None, schema=None):
        self.seen.append((list(messages), tools))
        calls = self.turns.pop(0) if self.turns else []
        msg = {"role": "assistant", "content": "" if calls else "DONE"}
        if calls:
            msg["tool_calls"] = calls
        return ChatResult(msg, 100, 10, 0.01)


def test_done_at_once_keeps_the_initial_chunks():
    llm = ScriptedChat()
    state = A.gather(llm, box(), count, "Acme: revenue?", [0, 1])
    assert state["stop"] == "done" and state["pool"] == [0, 1] and state["tool_calls"] == 0
    assert state["llm_calls"] == 1 and state["prompt_tokens"] == 100
    messages, tools = llm.seen[0]
    assert tools == A.TOOL_SPECS
    user = messages[1]["content"]
    assert '<chunk id="c1">' in user and "Acme: revenue?" in user


def test_search_shows_snippets_and_only_reads_add_evidence():
    llm = ScriptedChat(
        [call("keyword_search", query="net income")], [call("chunk_read", chunk_id="c7")]
    )
    state = A.gather(llm, box(), count, "q", [0])
    assert state["stop"] == "done" and state["pool"] == [0, 7]
    assert state["tool_calls"] == 2
    assert [t["added"] for t in state["trace"]] == [[], [7]]
    result = state["messages"][3]
    assert result["role"] == "tool" and result["content"].startswith("c7: chunk number 7")


def test_page_read_adds_the_missing_chunks_of_the_page_in_order():
    llm = ScriptedChat([call("page_read", chunk_id="c4")])
    state = A.gather(llm, box(), count, "q", [5, 0])
    assert state["pool"] == [5, 0, 3, 4]


def test_reading_a_chunk_already_in_the_evidence_adds_nothing():
    llm = ScriptedChat([call("chunk_read", chunk_id="c0"), call("page_read", chunk_id="c1")])
    state = A.gather(llm, box(), count, "q", [0, 1, 2])
    assert state["pool"] == [0, 1, 2] and [t["added"] for t in state["trace"]] == [[], []]


def test_tool_budget_stops_the_loop():
    turns = [[call("semantic_search", query=f"q{i}")] for i in range(10)]
    state = A.gather(ScriptedChat(*turns), box(), count, "q", [0])
    assert state["stop"] == "tool_budget" and state["tool_calls"] == A.MAX_TOOL_CALLS


def test_tool_budget_applies_inside_a_single_turn():
    many = [call("keyword_search", query=str(i)) for i in range(9)]
    state = A.gather(ScriptedChat(many), box(), count, "q", [0])
    assert state["stop"] == "tool_budget" and state["tool_calls"] == A.MAX_TOOL_CALLS


def test_context_budget_stops_the_loop(monkeypatch):
    monkeypatch.setattr(A, "MAX_CONTEXT_TOKENS", 12)  # each chunk is 6 words
    llm = ScriptedChat([call("chunk_read", chunk_id="c3")], [call("chunk_read", chunk_id="c6")])
    state = A.gather(llm, box(), count, "q", [0])
    assert state["stop"] == "context_budget" and state["pool"] == [0, 3, 6]


def test_bad_chunk_ids_and_unknown_tools_are_reported_to_the_agent():
    llm = ScriptedChat([call("chunk_read", chunk_id="banana"), call("summarize", text="x")])
    state = A.gather(llm, box(), count, "q", [0])
    contents = [m["content"] for m in state["messages"] if m["role"] == "tool"]
    assert contents[0].startswith("unknown chunk id") and contents[1].startswith("unknown tool")
    assert state["pool"] == [0]


def test_string_arguments_are_decoded():
    llm = ScriptedChat([{"function": {"name": "chunk_read", "arguments": '{"chunk_id": "2"}'}}])
    assert A.gather(llm, box(), count, "q", [0])["pool"] == [0, 2]


def test_parse_label():
    assert A.parse_label("c12") == 12 and A.parse_label("[c3]") == 3 and A.parse_label("7") == 7
    assert A.parse_label("chunk") is None
