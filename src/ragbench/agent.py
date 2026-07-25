"""Agentic retrieval loop (A-RAG style), built with LangGraph.

The agent does not answer. It starts from the fixed pipeline's top-10, judges whether the
evidence is complete, and searches for what is missing with four tools. The chunks it reads
form the evidence pool; the same generator as the fixed pipeline then answers from that pool.
So `fixed` and `agentic` differ only by the context they hand to the generator.

    START -> agent --(tool calls)--> tools --(budget left)--> agent
                   \\--(no tool call: DONE)--> END      \\--(budget exhausted)--> END

Budgets (protocol): at most 6 tool calls and 8,000 bge tokens of evidence. An exhausted budget
ends the question as ``insufficient_context``: the generator is never forced to guess.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from ragbench.generation import ChatModel

MAX_TOOL_CALLS = 6
MAX_CONTEXT_TOKENS = 8000
SEARCH_K = 5
SNIPPET_CHARS = 300

# Version of the agent prompt and tool behaviour, chosen on the development questions
# (v2 was tested and is worse: see results/development/README.md).
AGENT_PROMPT_VERSION = 1

AGENT_PROMPT = """You gather evidence to answer a question about a company's financial report. \
You do not answer the question yourself: another model will compute the answer from the chunks \
you collect.

The chunks found by a search engine are shown below, each with its id. Check whether they \
contain EVERY number the question needs (each year, each line item). If something is missing, \
use the tools:
- keyword_search: exact terms (a line item, a year, a company name);
- semantic_search: a description in other words;
- chunk_read: read a chunk in full (search results only show its beginning);
- page_read: read the whole page a chunk comes from (useful when a table is cut in two).
Only chunks shown below or read with chunk_read / page_read are passed on.

You have at most 6 tool calls. When the evidence is complete, or cannot be found, reply with \
the single word DONE and no tool call."""

AGENT_USER = """Question: {question}

Chunks found by the search engine:
{chunks}"""

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": arg_doc}},
                "required": [arg],
            },
        },
    }
    for name, description, arg, arg_doc in [
        ("keyword_search", "Keyword (BM25) search over all report chunks. Returns 5 chunk ids "
         "with the beginning of each chunk.", "query", "exact terms to look for"),
        ("semantic_search", "Semantic (embedding) search over all report chunks. Returns 5 chunk "
         "ids with the beginning of each chunk.", "query", "what the chunk talks about"),
        ("chunk_read", "Read one chunk in full and pass it on as evidence.", "chunk_id",
         "a chunk id such as c1234"),
        ("page_read", "Read every chunk of the page a chunk comes from, in page order, and pass "
         "them on as evidence.", "chunk_id", "a chunk id such as c1234"),
    ]
]


def chunk_label(i: int) -> str:
    return f"c{i}"


def parse_label(label: str) -> int | None:
    label = str(label).strip().strip("[]<>'\"")
    digits = label[1:] if label[:1] in "cC" else label
    return int(digits) if digits.isdigit() else None


@dataclass
class ToolBox:
    """Search and read over the pooled corpus. ``search_*`` return corpus indices, best first."""

    texts: Sequence[str]
    page_chunks: Callable[[int], list[int]]  # chunk index -> all chunk indices of its page
    keyword: Callable[[str, int], list[int]]
    semantic: Callable[[str, int], list[int]]


class AgentState(TypedDict):
    messages: list[dict]
    pool: list[int]  # evidence handed to the generator, in the order it was gathered
    pool_tokens: int
    tool_calls: int
    trace: list[dict]
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    llm_seconds: float
    stop: str  # "" while running, then "done", "tool_budget" or "context_budget"


def format_pool(texts: Sequence[str], ids: Sequence[int]) -> str:
    return "\n".join(f'<chunk id="{chunk_label(i)}">\n{texts[i]}\n</chunk>' for i in ids)


def build_graph(llm: ChatModel, box: ToolBox, count: Callable[[str], int]):
    def agent(state: AgentState) -> dict:
        result = llm.chat(state["messages"], tools=TOOL_SPECS)
        return {
            "messages": state["messages"] + [result.message],
            "llm_calls": state["llm_calls"] + 1,
            "prompt_tokens": state["prompt_tokens"] + result.prompt_tokens,
            "completion_tokens": state["completion_tokens"] + result.completion_tokens,
            "llm_seconds": state["llm_seconds"] + result.seconds,
            "stop": "" if result.message.get("tool_calls") else "done",
        }

    def tools(state: AgentState) -> dict:
        messages, pool, trace = list(state["messages"]), list(state["pool"]), list(state["trace"])
        pool_tokens, calls, stop = state["pool_tokens"], state["tool_calls"], ""
        for call in messages[-1].get("tool_calls", []):
            if calls >= MAX_TOOL_CALLS:
                stop = "tool_budget"
                break
            calls += 1
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            added: list[int] = []
            if name in ("keyword_search", "semantic_search"):
                search = box.keyword if name == "keyword_search" else box.semantic
                hits = search(str(args.get("query", "")), SEARCH_K)
                content = "\n".join(
                    f"{chunk_label(i)}: {box.texts[i][:SNIPPET_CHARS]}..." for i in hits
                ) or "no result"
                result_ids = hits
            elif name in ("chunk_read", "page_read"):
                idx = parse_label(args.get("chunk_id", ""))
                if idx is None or not 0 <= idx < len(box.texts):
                    content, result_ids = f"unknown chunk id {args.get('chunk_id')!r}", []
                else:
                    result_ids = [idx] if name == "chunk_read" else box.page_chunks(idx)
                    added = [i for i in result_ids if i not in pool]
                    content = format_pool(box.texts, result_ids)
            else:
                content, result_ids = f"unknown tool {name!r}", []
            pool += added
            pool_tokens += sum(count(box.texts[i]) for i in added)
            trace.append({"tool": name, "args": args, "results": result_ids, "added": added})
            messages.append({"role": "tool", "content": content})
            if pool_tokens > MAX_CONTEXT_TOKENS:
                stop = "context_budget"
                break
        if not stop and calls >= MAX_TOOL_CALLS:
            stop = "tool_budget"
        return {
            "messages": messages,
            "pool": pool,
            "pool_tokens": pool_tokens,
            "tool_calls": calls,
            "trace": trace,
            "stop": stop,
        }

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", lambda s: END if s["stop"] else "tools")
    graph.add_conditional_edges("tools", lambda s: END if s["stop"] else "agent")
    return graph.compile()


def gather(
    llm: ChatModel,
    box: ToolBox,
    count: Callable[[str], int],
    question: str,
    initial: Sequence[int],
) -> AgentState:
    """Run the loop for one question and return its final state."""
    initial = list(initial)
    state: AgentState = {
        "messages": [
            {"role": "system", "content": AGENT_PROMPT},
            {
                "role": "user",
                "content": AGENT_USER.format(
                    question=question, chunks=format_pool(box.texts, initial)
                ),
            },
        ],
        "pool": initial,
        "pool_tokens": sum(count(box.texts[i]) for i in initial),
        "tool_calls": 0,
        "trace": [],
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "llm_seconds": 0.0,
        "stop": "",
    }
    # Each agent turn either stops or spends at least one tool call: the loop is bounded.
    return build_graph(llm, box, count).invoke(state, {"recursion_limit": 4 * MAX_TOOL_CALLS + 4})
