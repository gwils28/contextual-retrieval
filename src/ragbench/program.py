"""FinQA reasoning programs: parse, execute, compare with the gold answer.

A program is a list of steps such as ``subtract(5829, 5735), divide(#0, 5735)``. ``#n`` is the
result of step n, ``const_100`` a constant, ``5%`` the number 0.05. Table operations take a row
label and apply to the numeric cells of that row: ``table_average(net income, none)``.

The generator writes the program; this module does the arithmetic, so a wrong number is never
an arithmetic slip of the LLM. Number parsing follows the official FinQA evaluator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BINARY_OPS = ("add", "subtract", "multiply", "divide", "greater", "exp")
TABLE_OPS = ("table_max", "table_min", "table_sum", "table_average")
OPS = BINARY_OPS + TABLE_OPS
# The only constants FinQA defines; any other ``const_*`` is an invented number.
_CONSTANT_NAMES = ("1", "2", "3", "4", "5", "6", "7", "9", "10", "100", "1000", "1000000", "m1")
CONSTANTS = frozenset(f"const_{n}" for n in _CONSTANT_NAMES)

Value = float | str  # str only for the yes/no of ``greater``


class ProgramError(ValueError):
    """The program does not parse or cannot be executed."""


@dataclass(frozen=True)
class Step:
    op: str
    args: tuple[str, ...]


def parse(program: str) -> list[Step]:
    """Split a program into steps. Row labels may contain commas or parentheses."""
    text = program.strip()
    if not text:
        raise ProgramError("empty program")
    steps = []
    for raw in re.split(r"\)\s*,\s*(?=[a-z_]+\()", text):
        raw = raw.strip()
        if raw.endswith(")"):
            raw = raw[:-1]
        op, sep, inner = raw.partition("(")
        op = op.strip()
        if not sep or op not in OPS:
            raise ProgramError(f"unknown step: {raw!r}")
        if op in TABLE_OPS:
            label, sep, rest = inner.rpartition(",")
            if not sep or rest.strip() != "none":
                raise ProgramError(f"table step needs (row label, none): {raw!r}")
            args = (label.strip().strip("'\""),)
        else:
            args = tuple(a.strip().strip("'\"") for a in inner.split(","))
            if len(args) != 2:
                raise ProgramError(f"{op} takes 2 arguments: {raw!r}")
        steps.append(Step(op, args))
    return steps


def to_number(text: str) -> float | None:
    """FinQA rules: commas dropped, ``5%`` -> 0.05, ``const_m1`` -> -1. None if not a number."""
    text = text.replace(",", "").strip()
    if text.startswith("const_"):
        if text not in CONSTANTS:
            return None
        text = "-1" if text == "const_m1" else text.removeprefix("const_")
    scale = 1.0
    if text.endswith("%"):
        text, scale = text[:-1], 0.01
    try:
        return float(text) * scale
    except ValueError:
        return None


def row_values(row: tuple[str, ...]) -> list[float]:
    """Numeric cells of a table row (label excluded): ``$ 1,200 ( 3% )`` -> 1200."""
    values = []
    for cell in row[1:]:
        num = to_number(cell.replace("$", "").split("(")[0])
        if num is None:
            raise ProgramError(f"non-numeric cell {cell!r} in row {row[0]!r}")
        values.append(num)
    return values


def find_row(table: tuple[tuple[str, ...], ...], label: str) -> tuple[str, ...]:
    for row in table:
        if row and row[0].strip() == label:
            return row
    raise ProgramError(f"no table row labelled {label!r}")


def execute(program: str, table: tuple[tuple[str, ...], ...] = ()) -> Value:
    """Run the program and return the result of its last step."""
    results: list[Value] = []

    def operand(arg: str) -> float:
        if arg.startswith("#"):
            idx = int(arg[1:]) if arg[1:].isdigit() else len(results)
            if idx >= len(results) or isinstance(results[idx], str):
                raise ProgramError(f"bad reference {arg}")
            return results[idx]
        num = to_number(arg)
        if num is None:
            raise ProgramError(f"not a number: {arg!r}")
        return num

    for step in parse(program):
        if step.op in TABLE_OPS:
            values = row_values(find_row(table, step.args[0]))
            if not values:
                raise ProgramError(f"empty row {step.args[0]!r}")
            agg = {"table_max": max, "table_min": min, "table_sum": sum}.get(step.op)
            results.append(agg(values) if agg else sum(values) / len(values))
            continue
        a, b = (operand(x) for x in step.args)
        if step.op == "add":
            results.append(a + b)
        elif step.op == "subtract":
            results.append(a - b)
        elif step.op == "multiply":
            results.append(a * b)
        elif step.op == "divide":
            if b == 0:
                raise ProgramError("division by zero")
            results.append(a / b)
        elif step.op == "exp":
            try:
                value = a**b
            except (OverflowError, ZeroDivisionError) as e:
                raise ProgramError(f"exp({a}, {b}): {e}") from e
            if isinstance(value, complex):
                raise ProgramError(f"exp({a}, {b}) is not a real number")
            results.append(value)
        else:  # greater
            results.append("yes" if a > b else "no")
    return results[-1]


def answers_match(pred: Value, gold: Value, rel_tol: float = 0.005) -> bool:
    """Protocol rule: |pred - gold| <= rel_tol * max(|gold|, 1); 25 and 0.25 both accepted."""
    if isinstance(gold, str) and gold.strip().lower() in ("yes", "no"):
        return isinstance(pred, str) and pred.strip().lower() == gold.strip().lower()
    gold_num = gold if isinstance(gold, float | int) else to_number(str(gold))
    if gold_num is None or isinstance(pred, str):
        return False
    tol = rel_tol * max(abs(gold_num), 1.0)
    return any(abs(pred * scale - gold_num) <= tol for scale in (1.0, 100.0, 0.01))
