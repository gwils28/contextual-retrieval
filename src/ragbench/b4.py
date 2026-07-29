"""Block 4 — process-level metrics of the Block 3 agent. Implements configs/b4_process_metrics.yaml.

python -m ragbench.b4 process    # reads the Block 3 agent traces, no LLM call

Exploratory: the traces had been read before these metrics were defined. They describe how the
agent spent its budget; they never rank systems.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "blocks" / "03-system-design" / "results" / "evaluation" / "agentic.jsonl"
RESULTS = ROOT / "blocks" / "04-reading" / "results"

SEARCHES = ("keyword_search", "semantic_search")
READS = ("chunk_read", "page_read")
TOOLS = SEARCHES + READS


def process_metrics(trace: Sequence[dict], initial: Sequence[int]) -> dict:
    """Counts for one question, replaying the evidence pool step by step."""
    pool, start = set(initial), set(initial)
    m = {f"calls_{t}": 0 for t in (*TOOLS, "invalid")}
    m.update({"tool_calls": len(trace), "empty_read": 0, "reread_of_initial": 0,
              "search_without_new_candidate": 0, "chunks_added": 0})
    for step in trace:
        tool, results, added = step["tool"], step["results"], step["added"]
        if tool not in TOOLS or (tool in READS and not results):
            m["calls_invalid"] += 1
        elif tool in SEARCHES:
            m[f"calls_{tool}"] += 1
            m["search_without_new_candidate"] += set(results) <= pool
        else:
            m[f"calls_{tool}"] += 1
            m["empty_read"] += not added
            m["reread_of_initial"] += tool == "chunk_read" and set(results) <= start
        pool |= set(added)
        m["chunks_added"] += len(added)
    return m


def summarize(per_question: list[dict]) -> dict:
    """Totals over questions, with shares over the calls of the relevant type."""
    total = {k: sum(q[k] for q in per_question) for k in per_question[0] if k != "stop"}
    reads = total["calls_chunk_read"] + total["calls_page_read"]
    searches = total["calls_keyword_search"] + total["calls_semantic_search"]
    chunk_reads = total["calls_chunk_read"]
    return {
        "questions": len(per_question),
        "tool_calls_mean": total["tool_calls"] / len(per_question),
        **{k: v for k, v in total.items() if k.startswith("calls_")},
        "empty_read_share": total["empty_read"] / reads if reads else float("nan"),
        "reread_of_initial_share": (
            total["reread_of_initial"] / chunk_reads if chunk_reads else float("nan")),
        "search_without_new_candidate_share": (
            total["search_without_new_candidate"] / searches if searches else float("nan")),
        "chunks_added_mean": total["chunks_added"] / len(per_question),
    }


def cmd_process(_: argparse.Namespace) -> None:
    import pandas as pd

    per_question = []
    for row in map(json.loads, TRACES.open()):
        initial = row["context"][: len(row["context"]) - row["added_chunks"]]
        per_question.append({"stop": row["stop"], **process_metrics(row["trace"], initial)})
    groups = {"all": per_question}
    for stop in sorted({q["stop"] for q in per_question}):
        groups[f"stop={stop}"] = [q for q in per_question if q["stop"] == stop]
    table = pd.DataFrame([{"group": g, **summarize(qs)} for g, qs in groups.items()])
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "process_metrics.csv"
    table.to_csv(out, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.round(3).to_string(index=False))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragbench.b4")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("process").set_defaults(func=cmd_process)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
