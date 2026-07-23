"""Block 3 runner (fixed vs agentic RAG). Implements configs/b3_system_protocol.yaml.

python -m ragbench.b3 questions   # draw the evaluation and development questions (seed 0)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ragbench import finqa
from ragbench.audit import audited_sample, gold_type

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "b3_system_protocol.yaml"
QUESTIONS = ROOT / "configs" / "b3_questions.csv"
RAW = ROOT / "data" / "raw" / "finqa"

N_QUESTIONS = 100
SEED = 0


def draw_questions(
    questions: list[finqa.Question], n: int = N_QUESTIONS, seed: int = SEED
) -> dict[str, list[finqa.Question]]:
    """Evaluation set first, then a disjoint development set, both stratified by gold type."""
    evaluation = audited_sample(questions, n, seed)
    taken = {q.qid for q in evaluation}
    development = audited_sample([q for q in questions if q.qid not in taken], n, seed)
    return {"evaluation": evaluation, "development": development}


def write_questions(sets: dict[str, list[finqa.Question]], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qid", "role", "gold_type", "page_id"])
        for role, qs in sets.items():
            for q in qs:
                w.writerow([q.qid, role, gold_type(q), q.page_id])


def load_questions(path: Path = QUESTIONS) -> dict[str, list[str]]:
    sets: dict[str, list[str]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            sets.setdefault(row["role"], []).append(row["qid"])
    return sets


def cmd_questions(_: argparse.Namespace) -> None:
    finqa.download(RAW)
    _, questions = finqa.load(RAW / "test.json")
    sets = draw_questions(questions)
    write_questions(sets, QUESTIONS)
    for role, qs in sets.items():
        print(role, len(qs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragbench.b3")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("questions").set_defaults(func=cmd_questions)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
