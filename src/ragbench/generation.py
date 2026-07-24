"""Answer generation: prompt, structured output, grounding check and scoring.

The generator reads numbered context chunks and returns JSON: a FinQA program, the operands it
uses with the chunk each one comes from, and a status (``answer`` or ``insufficient_context``).
``ragbench.program`` executes the program, so the LLM never does arithmetic.

Retrieval and generation are scored separately: ``retrieval`` metrics look only at which chunks
were handed to the generator, ``generation`` metrics only at its output.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ragbench import program as P

STATUSES = ("answer", "insufficient_context")

# Each step carries its operation and two arguments, each with the chunk it was copied from:
# nesting is impossible and every number has a citation. ``ragbench`` assembles the program text.
STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": list(P.OPS)},
        "arg1": {"type": "string"},
        "arg1_chunk": {"type": "integer"},
        "arg2": {"type": "string"},
        "arg2_chunk": {"type": "integer"},
    },
    "required": ["op", "arg1", "arg1_chunk", "arg2", "arg2_chunk"],
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": list(STATUSES)},
        "steps": {"type": "array", "items": STEP_SCHEMA},
    },
    "required": ["status", "steps"],
}

# Bumped at every prompt change made on the development questions (recorded in each result row).
PROMPT_VERSION = 4

SYSTEM_PROMPT = """You answer questions about financial reports using ONLY the context chunks \
you are given. You never compute the answer yourself: you write the steps of a calculation.

Steps:
- Each step is one operation on exactly two arguments: add, subtract, multiply, divide,
  exp (arg1 to the power arg2), greater ("yes" if arg1 > arg2 else "no").
- An argument is one of:
  * a number copied from a chunk EXACTLY as written there, only without $, commas and %
    ("$ 161.9 million" -> 161.9: never convert units), and in argN_chunk the id of the
    <chunk> tag it was copied from. A copied number never has argN_chunk 0;
  * #n, the result of an EARLIER step: #0 is the first step, #1 the second, and so on;
  * a constant: const_1 const_2 const_3 const_4 const_5 const_6 const_7 const_9 const_10
    const_100 const_1000 const_1000000 const_m1 (= -1).
  For #n and constants, argN_chunk is 0.
- A percentage or a growth rate stops at the division: do not multiply by 100.
- Rarely needed: table_sum, table_average, table_max, table_min over all the numbers of one
  table row: arg1 is the row label copied exactly, arg1_chunk its chunk, arg2 is "none" (chunk 0).

Examples:
- "what was the percentage change in revenue from 2016 to 2017?", revenue 5735 (2016) and
  5829 (2017), both in chunk 2 ->
  [{"op": "subtract", "arg1": "5829", "arg1_chunk": 2, "arg2": "5735", "arg2_chunk": 2},
   {"op": "divide", "arg1": "#0", "arg1_chunk": 0, "arg2": "5735", "arg2_chunk": 2}]
- "what is the total of the three payments?", 120 and 75 in chunk 1, 40 in chunk 3 ->
  [{"op": "add", "arg1": "120", "arg1_chunk": 1, "arg2": "75", "arg2_chunk": 1},
   {"op": "add", "arg1": "#0", "arg1_chunk": 0, "arg2": "40", "arg2_chunk": 3}]

Answer with JSON only: {"status": ..., "steps": [...]}.
"status" is "answer", or "insufficient_context" when the chunks do not contain every number you
need (then "steps" is []). Never guess a number."""

USER_TEMPLATE = """Context ({n} chunks, ids 1 to {n}):
{chunks}

Question: {question}"""


# --- LLM interface --------------------------------------------------------------------------


@dataclass
class ChatResult:
    message: dict  # {"role": "assistant", "content": ..., "tool_calls"?: [...]}
    prompt_tokens: int
    completion_tokens: int
    seconds: float


class ChatModel(Protocol):
    name: str

    def chat(
        self, messages: list[dict], tools: list[dict] | None = None, schema: dict | None = None
    ) -> ChatResult: ...


class OllamaChat:
    """Deterministic chat (temperature 0, seed 0) against a local Ollama server."""

    def __init__(
        self,
        model: str = "qwen2.5:7b-instruct",
        host: str = "http://127.0.0.1:11434",
        num_ctx: int = 8192,
        num_predict: int = 512,
        timeout: int = 300,
    ):
        self.name = model
        self.host = host
        self.options = {"temperature": 0, "seed": 0, "num_ctx": num_ctx, "num_predict": num_predict}
        self.timeout = timeout

    def chat(self, messages, tools=None, schema=None) -> ChatResult:
        body = {"model": self.name, "messages": messages, "stream": False, "options": self.options}
        if tools:
            body["tools"] = tools
        if schema:
            body["format"] = schema
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read())
        return ChatResult(
            out["message"],
            out.get("prompt_eval_count", 0),
            out.get("eval_count", 0),
            time.perf_counter() - t0,
        )

    def unload(self) -> None:
        """Free the GPU (same as ``ollama stop``)."""
        body = json.dumps({"model": self.name, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=body)
        urllib.request.urlopen(req, timeout=self.timeout).read()


# --- Prompt and parsing ---------------------------------------------------------------------


def format_chunks(texts: Sequence[str]) -> str:
    """Chunks are tagged with 1-based ids; the model cites these ids."""
    return "\n".join(f'<chunk id="{n}">\n{t}\n</chunk>' for n, t in enumerate(texts, start=1))


def answer_messages(question: str, texts: Sequence[str]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                n=len(texts), chunks=format_chunks(texts), question=question
            ),
        },
    ]


@dataclass
class Answer:
    status: str  # "answer", "insufficient_context" or "invalid" (unparseable output)
    program: str = ""
    operands: list[tuple[str, int]] = field(default_factory=list)  # (value, 1-based chunk label)


def parse_answer(content: str) -> Answer:
    """Assemble the program text from the steps; arguments with a chunk id > 0 are operands."""
    try:
        data = json.loads(content)
        status = data["status"]
        if status not in STATUSES:
            return Answer("invalid")
        steps, operands = [], []
        for step in data.get("steps", []):
            args = [str(step["arg1"]).strip(), str(step["arg2"]).strip()]
            chunks = [int(step["arg1_chunk"]), int(step["arg2_chunk"])]
            steps.append(f"{step['op']}({args[0]}, {args[1]})")
            operands += [(a, c) for a, c in zip(args, chunks, strict=True) if c > 0]
        return Answer(status, ", ".join(steps), operands)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError):
        return Answer("invalid")


# --- Grounding and scoring ------------------------------------------------------------------

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")


def numbers_in(text: str) -> set[float]:
    """Absolute values of the numbers written in a text ("$ 5,829" -> 5829, "( 3.2 )" -> 3.2)."""
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}


def literal_value(arg: str) -> float | None:
    """Absolute value of a number literal of the program, as written ("5%" -> 5, not 0.05)."""
    if arg.startswith(("#", "const_")):
        return None
    num = P.to_number(arg.removesuffix("%"))
    return None if num is None else abs(num)


def program_literals(steps: list[P.Step]) -> tuple[list[float], list[str]]:
    """Number literals and table row labels that must be grounded in the context."""
    numbers, labels = [], []
    for step in steps:
        if step.op in P.TABLE_OPS:
            labels.append(step.args[0])
            continue
        for arg in step.args:
            value = literal_value(arg)
            if value is not None:
                numbers.append(value)
    return numbers, labels


def is_grounded(answer: Answer, texts: Sequence[str]) -> bool:
    """Every number and row label of the program is an operand, and appears in its cited chunk."""
    try:
        numbers, labels = program_literals(P.parse(answer.program))
    except P.ProgramError:
        return False
    cited: list[tuple[str, str]] = []
    for value, label in answer.operands:
        if not 1 <= label <= len(texts):
            return False
        cited.append((value, texts[label - 1]))
    for num in numbers:
        if not any(
            literal_value(v) is not None
            and abs(literal_value(v) - num) < 1e-9
            and num in numbers_in(text)
            for v, text in cited
        ):
            return False
    for row in labels:
        if not any(v.lower() == row.lower() and row.lower() in text.lower() for v, text in cited):
            return False
    return True


@dataclass
class Scored:
    status: str
    program_valid: bool
    value: P.Value | None
    correct: bool
    grounded: bool
    error: str = ""

    @property
    def wrong_answer(self) -> bool:
        """Answered with a valid program, and wrong: the costly error in finance."""
        return self.status == "answer" and self.program_valid and not self.correct


def score_answer(
    answer: Answer,
    texts: Sequence[str],
    tables: Sequence[tuple[tuple[str, ...], ...]],
    gold: str,
) -> Scored:
    """Execute and score one answer. ``tables[i]`` is the table of the page of chunk i (0-based):
    a table step runs on the tables of the chunks cited for its row label."""
    if answer.status != "answer":
        return Scored(answer.status, False, None, False, False)
    rows = tuple(
        row
        for _, label in answer.operands
        if 1 <= label <= len(tables)
        for row in tables[label - 1]
    )
    try:
        value = P.execute(answer.program, rows)
    except P.ProgramError as e:
        return Scored("answer", False, None, False, is_grounded(answer, texts), str(e))
    correct = P.answers_match(value, gold)
    return Scored("answer", True, value, correct, is_grounded(answer, texts))
