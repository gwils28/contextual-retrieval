"""Block 3 runner (fixed vs agentic RAG). Implements configs/b3_system_protocol.yaml.

python -m ragbench.b3 questions                 # draw the question sets (seed 0)
python -m ragbench.b3 retrieve --role development  # fixed top-10 + reranker scores (GPU)
python -m ragbench.b3 generate --system fixed --role development   # needs Ollama
python -m ragbench.b3 generate --system agentic --role development # needs Ollama
python -m ragbench.b3 threshold                 # freeze the cascade threshold (development)
python -m ragbench.b3 cascade --role development   # fixed + agentic rows, no LLM call
python -m ragbench.b3 summary --role development   # quick table (no CI), for development only
python -m ragbench.b3 report --role evaluation     # protocol tables: CI, paired tests, strata

Retrieval runs first and is saved, then the GPU is freed before the LLM is loaded: on an 8 GB
GPU, Qwen 7B, the embedder and the reranker do not fit together comfortably.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ragbench import evaluation as ev
from ragbench import finqa
from ragbench import generation as G
from ragbench.audit import audited_sample, gold_type

if TYPE_CHECKING:
    import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "b3_system_protocol.yaml"
CASCADE = ROOT / "configs" / "b3_cascade.yaml"
QUESTIONS = ROOT / "configs" / "b3_questions.csv"
RAW = ROOT / "data" / "raw" / "finqa"
RESULTS = ROOT / "blocks" / "03-system-design" / "results"

N_QUESTIONS = 100
SEED = 0
ROLES = ("development", "evaluation")
STRATEGY = "naive"  # Block 2 winner, with its versioned Qwen contexts
TOP_K = 10
GENERATED = ("oracle_context", "fixed", "agentic")
SYSTEMS = GENERATED + ("cascade",)  # the cascade is composed from fixed and agentic rows
# Share of development questions flagged by the reranker signal (protocol: systems.cascade).
CASCADE_SHARE = 0.20
# Summed when a question escalates: the cascade pays for fixed, then for the agent.
COST_KEYS = ("llm_calls", "prompt_tokens", "completion_tokens", "llm_seconds", "generation_s")
# One context size for the agent loop and its final answer, so Ollama never reloads the model
# between them; large enough that a long agent conversation is never silently truncated.
AGENT_NUM_CTX = 16384


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


class Corpus:
    """Block 2 naive chunks with their contexts: the exact texts indexed and sent to the LLM."""

    def __init__(self):
        from ragbench import b2

        self.bench = b2.load_bench()
        self.chunks = self.bench.chunk(STRATEGY)
        cache = b2.load_cache(b2.context_cache_path(STRATEGY, "ollama"))
        self.texts = [
            b2.contextualized_text(cache[c.chunk_id], c.text) for c in self.chunks.chunks
        ]
        self.by_qid = {q.qid: q for q in self.bench.questions}

    def question(self, qid: str) -> finqa.Question:
        return self.by_qid[qid]

    def query(self, q: finqa.Question) -> str:
        return f"{self.bench.company(q.page_id)}: {q.question}"

    def gold(self, q: finqa.Question) -> list[frozenset[int]]:
        return self.chunks.gold_sets(q.page_id, q.gold_keys)

    def table(self, i: int) -> tuple[tuple[str, ...], ...]:
        return self.bench.pages[self.chunks.chunks[i].page_id].table

    def page_chunks(self, i: int) -> list[int]:
        page_id = self.chunks.chunks[i].page_id
        return [j for j, c in enumerate(self.chunks.chunks) if c.page_id == page_id]


def role_dir(role: str) -> Path:
    return RESULTS / role


def cmd_retrieve(args: argparse.Namespace) -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from ragbench import b2
    from ragbench.retrieval import BM25Retriever, DenseRetriever, Reranker, hybrid_search

    torch.manual_seed(0)
    corpus = Corpus()
    dense_model = SentenceTransformer(b2.DENSE_MODEL, device="cuda")
    cross = CrossEncoder(
        b2.RERANK_MODEL, device="cuda", max_length=512, model_kwargs={"torch_dtype": torch.float16}
    )
    texts = corpus.texts
    dense = DenseRetriever(
        dense_model, texts, embeddings=b2._embeddings(dense_model, texts, f"{STRATEGY}_contextual")
    )
    bm25 = BM25Retriever(texts)
    reranker = Reranker(cross, texts)

    def search(query: str) -> list[tuple[int, float]]:
        candidates = hybrid_search(dense, bm25, query, b2.RERANK_TOP_N, 100, b2.RRF_K)
        return reranker.scored(query, candidates, TOP_K)

    qids = load_questions()[args.role]
    for qid in qids[:3]:  # warm-up before timing
        search(corpus.query(corpus.question(qid)))
    rows = []
    for qid in qids:
        t0 = time.perf_counter()
        top = search(corpus.query(corpus.question(qid)))
        ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "qid": qid,
            "top": [i for i, _ in top],
            "scores": [s for _, s in top],
            "latency_ms": ms,
        })
    out = role_dir(args.role) / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {out}")


def context_for(system: str, corpus: Corpus, q: finqa.Question, top: list[int]) -> list[int]:
    if system == "oracle_context":  # gold chunks in page order (naive gold sets are singletons)
        return sorted({min(s) for s in corpus.gold(q)})
    return top


def run_question(llm: G.ChatModel, corpus: Corpus, q: finqa.Question, ctx: list[int]) -> dict:
    texts = [corpus.texts[i] for i in ctx]
    result = llm.chat(G.answer_messages(corpus.query(q), texts), schema=G.ANSWER_SCHEMA)
    content = result.message.get("content", "")
    answer = G.parse_answer(content)
    scored = G.score_answer(answer, texts, [corpus.table(i) for i in ctx], q.answer)
    gold = corpus.gold(q)
    return {
        "qid": q.qid,
        "prompt_version": G.PROMPT_VERSION,
        "context": ctx,
        # retrieval stage: only which chunks the generator saw
        "pass_all_ctx": ev.pass_all(ctx, gold, len(ctx)),
        "pass_any_ctx": ev.pass_any(ctx, gold, len(ctx)),
        "context_tokens": sum(corpus.bench.count(t) for t in texts),
        # generation stage
        "status": scored.status,
        "program": answer.program,
        "operands": answer.operands,
        "program_valid": scored.program_valid,
        "value": scored.value,
        "gold": q.answer,
        "correct": scored.correct,
        "grounded": scored.grounded,
        "wrong_answer": scored.wrong_answer,
        "error": scored.error,
        # cost
        "llm_calls": 1,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "llm_seconds": result.seconds,
        "raw": content,
    }


def agent_toolbox(corpus: Corpus):
    """Agent tools on CPU (BM25, embedder with cached embeddings): the GPU is left to Ollama."""
    from sentence_transformers import SentenceTransformer

    from ragbench import b2
    from ragbench.agent import ToolBox
    from ragbench.retrieval import BM25Retriever, DenseRetriever

    model = SentenceTransformer(b2.DENSE_MODEL, device="cpu")
    texts = corpus.texts
    dense = DenseRetriever(
        model, texts, embeddings=b2._embeddings(model, texts, f"{STRATEGY}_contextual")
    )
    bm25 = BM25Retriever(texts)
    return ToolBox(texts, corpus.page_chunks, bm25.search, dense.search)


def run_agentic(llm: G.ChatModel, box, corpus: Corpus, q: finqa.Question, top: list[int]) -> dict:
    """Agent loop, then the same generator as `fixed` on the evidence pool."""
    from ragbench.agent import AGENT_PROMPT_VERSION, gather

    state = gather(llm, box, corpus.bench.count, corpus.query(q), top)
    pool = state["pool"]
    if state["stop"] == "done":
        row = run_question(llm, corpus, q, pool)
    else:  # budget exhausted: insufficient_context, the generator is not asked to guess
        gold = corpus.gold(q)
        row = {
            "qid": q.qid,
            "prompt_version": G.PROMPT_VERSION,
            "context": pool,
            "pass_all_ctx": ev.pass_all(pool, gold, len(pool)),
            "pass_any_ctx": ev.pass_any(pool, gold, len(pool)),
            "context_tokens": state["pool_tokens"],
            "status": "insufficient_context",
            "program": "",
            "operands": [],
            "program_valid": False,
            "value": None,
            "gold": q.answer,
            "correct": False,
            "grounded": False,
            "wrong_answer": False,
            "error": "",
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_seconds": 0.0,
            "raw": "",
        }
    row.update({
        "agent_prompt_version": AGENT_PROMPT_VERSION,
        "stop": state["stop"],
        "tool_calls": state["tool_calls"],
        "added_chunks": len(pool) - len(top),
        "llm_calls": row["llm_calls"] + state["llm_calls"],
        "prompt_tokens": row["prompt_tokens"] + state["prompt_tokens"],
        "completion_tokens": row["completion_tokens"] + state["completion_tokens"],
        "llm_seconds": row["llm_seconds"] + state["llm_seconds"],
        "trace": state["trace"],
        "agent_final": state["messages"][-1].get("content", ""),
    })
    return row


def cascade_threshold(top1: Sequence[float], share: float = CASCADE_SHARE) -> float:
    """Top-1 reranker score below which a question counts as least confident."""
    import numpy as np

    return float(np.quantile(np.asarray(top1, dtype=float), share))


def escalation_reasons(fixed: dict, top1: float, threshold: float) -> list[str]:
    """The cheap signals of the protocol that fired on the fixed pipeline's row."""
    reasons = []
    if fixed["status"] == "insufficient_context":
        reasons.append("refusal")
    elif not fixed["grounded"]:  # an invalid output has no grounded program either
        reasons.append("ungrounded")
    if top1 < threshold:
        reasons.append("low_score")
    return reasons


def cascade_row(fixed: dict, agentic: dict, reasons: list[str]) -> dict:
    """Fixed's row if no signal fired, else the agent's row (even a refusal) at both costs."""
    if not reasons:
        row = dict(fixed, tool_calls=0)
    else:
        row = dict(agentic)
        for key in COST_KEYS:
            if key in fixed and key in agentic:
                row[key] = fixed[key] + agentic[key]
            else:  # a cost measured for one stage only would be understated
                row.pop(key, None)
    row.update({"escalated": bool(reasons), "escalation_reasons": reasons})
    return row


def read_rows(role: str, system: str) -> dict[str, dict]:
    path = role_dir(role) / f"{system}.jsonl"
    return {r["qid"]: r for r in map(json.loads, path.open())}


def cmd_threshold(args: argparse.Namespace) -> None:
    if CASCADE.exists():
        raise SystemExit(f"{CASCADE} is frozen; delete it by hand to redraw (and say why)")
    saved = json.loads((role_dir("development") / "retrieval.json").read_text())
    threshold = cascade_threshold([r["scores"][0] for r in saved])
    flagged = sum(r["scores"][0] < threshold for r in saved)
    CASCADE.write_text(
        "# Block 3 cascade threshold, written once by `make b3-threshold` from the\n"
        "# development questions before any evaluation run (b3_system_protocol.yaml).\n"
        "id: b3-cascade-threshold\n"
        "source: development questions, top-1 reranker score of the fixed pipeline\n"
        f"rule: escalate if top-1 score < threshold ({CASCADE_SHARE:.0%} quantile, numpy linear)\n"
        f"flagged_development_questions: {flagged} of {len(saved)}\n"
        f"threshold: {threshold!r}\n"
    )
    print(f"wrote {CASCADE}: threshold {threshold:.4f}, flags {flagged}/{len(saved)}")


def load_threshold() -> float:
    import yaml

    return float(yaml.safe_load(CASCADE.read_text())["threshold"])


def cmd_cascade(args: argparse.Namespace) -> None:
    threshold = load_threshold()
    saved = json.loads((role_dir(args.role) / "retrieval.json").read_text())
    top1 = {r["qid"]: r["scores"][0] for r in saved}
    fixed, agentic = read_rows(args.role, "fixed"), read_rows(args.role, "agentic")
    out = role_dir(args.role) / "cascade.jsonl"
    with out.open("w") as f:
        for qid in load_questions()[args.role]:
            reasons = escalation_reasons(fixed[qid], top1[qid], threshold)
            f.write(json.dumps(cascade_row(fixed[qid], agentic[qid], reasons)) + "\n")
    print(f"wrote {out}")


def cmd_generate(args: argparse.Namespace) -> None:
    corpus = Corpus()
    saved = json.loads((role_dir(args.role) / "retrieval.json").read_text())
    retrieval = {r["qid"]: r for r in saved}
    out = role_dir(args.role) / f"{args.system}.jsonl"
    done = set()
    if out.exists():  # resume an interrupted run
        done = {json.loads(line)["qid"] for line in out.open()}
    agentic = args.system == "agentic"
    llm = G.OllamaChat(num_ctx=AGENT_NUM_CTX) if agentic else G.OllamaChat()
    box = agent_toolbox(corpus) if agentic else None
    qids = [qid for qid in load_questions()[args.role] if qid not in done]
    with out.open("a") as f:
        for n, qid in enumerate(qids, start=1):
            q = corpus.question(qid)
            top = retrieval[qid]["top"]
            t0 = time.perf_counter()
            if agentic:
                row = run_agentic(llm, box, corpus, q, top)
            else:
                row = run_question(llm, corpus, q, context_for(args.system, corpus, q, top))
            row["generation_s"] = time.perf_counter() - t0
            if args.system != "oracle_context":
                row["retrieval_ms"] = retrieval[qid]["latency_ms"]
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{n}/{len(qids)} {qid} {row['status']} correct={row['correct']}", flush=True)
    llm.unload()


def cmd_summary(args: argparse.Namespace) -> None:
    import pandas as pd

    rows = []
    for system in SYSTEMS:
        path = role_dir(args.role) / f"{system}.jsonl"
        if not path.exists():
            continue
        df = pd.read_json(path, lines=True)
        rows.append({
            "system": system,
            "n": len(df),
            "pass_all_ctx": df.pass_all_ctx.mean(),
            "exec_accuracy": df.correct.mean(),
            "program_valid": df.program_valid.mean(),
            "grounded": df.grounded.mean(),
            "refusal_rate": (df.status == "insufficient_context").mean(),
            "wrong_answer_rate": df.wrong_answer.mean(),
            "invalid_output": (df.status == "invalid").mean(),
            "ctx_tokens_p50": df.context_tokens.median(),
            "llm_calls_mean": df.llm_calls.mean(),
            "tool_calls_mean": df.tool_calls.mean() if "tool_calls" in df else 0.0,
            "escalation_rate": df.escalated.mean() if "escalated" in df else float("nan"),
            "llm_s_p50": df.llm_seconds.median(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# Report (protocol: metrics, statistics, layer_economics). Rates get a bootstrap CI.
RATES = {
    "pass_all_ctx": "pass_all_ctx",
    "pass_any_ctx": "pass_any_ctx",
    "exec_accuracy": "correct",
    "program_valid": "program_valid",
    "grounded": "grounded",
    "refusal_rate": "refusal",
    "wrong_answer_rate": "wrong_answer",
}
PAIRS = (("fixed", "agentic"), ("fixed", "cascade"), ("agentic", "cascade"))
PAIRED_METRICS = ("correct", "wrong_answer")
MIN_STRATUM = 20  # below: counts only, no CI (protocol: statistics.small_strata)
B2_RESULTS = ROOT / "blocks" / "02-retrieval-bench" / "results"
# Block 2 configs before and after each retrieval layer (naive chunking, main setting).
RETRIEVAL_LAYERS = {
    "hybrid": ("dense", "hybrid", "BM25 index (CPU)"),
    "reranker": ("hybrid", "hybrid_rerank", "none"),
    "contextual_chunks": (
        "hybrid_rerank", "contextual_hybrid_rerank", "one LLM call per chunk ({n} calls)"
    ),
}


def price_table() -> dict:
    import yaml

    protocol = yaml.safe_load(PROTOCOL.read_text())
    return next(a["price_table"] for a in protocol["amendments"] if "price_table" in a)


def load_system(role: str, system: str, prices: dict) -> pd.DataFrame:
    """Per-question rows of one system, with the derived columns of the report."""
    import pandas as pd

    df = pd.read_json(role_dir(role) / f"{system}.jsonl", lines=True).set_index("qid")
    df["refusal"] = df.status == "insufficient_context"
    for col in ("correct", "program_valid", "grounded", "wrong_answer"):
        df[col] = df[col].astype(bool)
    df["pass_all_ctx"] = df.pass_all_ctx.astype(bool)
    df["pass_any_ctx"] = df.pass_any_ctx.astype(bool)
    if "tool_calls" not in df:
        df["tool_calls"] = 0
    # End to end, batch size 1: retrieval (none for the oracle) + generation (agent tools included).
    # Not measured (NaN) in rows written before generation_s was recorded.
    retrieval_ms = df["retrieval_ms"] if "retrieval_ms" in df else 0.0
    generation_s = df["generation_s"] if "generation_s" in df else float("nan")
    df["latency_ms"] = retrieval_ms + generation_s * 1000
    df["usd"] = (
        df.prompt_tokens * prices["usd_per_million_input_tokens"]
        + df.completion_tokens * prices["usd_per_million_output_tokens"]
    ) / 1e6
    return df


def summary_row(df: pd.DataFrame) -> dict:
    row: dict = {"n": len(df)}
    for name, col in RATES.items():
        row[name] = df[col].mean()
        row[f"{name}_lo"], row[f"{name}_hi"] = ev.bootstrap_ci(df[col].to_numpy())
    row.update({
        "context_tokens_p50": df.context_tokens.median(),
        "llm_calls_mean": df.llm_calls.mean(),
        "prompt_tokens_mean": df.prompt_tokens.mean(),
        "completion_tokens_mean": df.completion_tokens.mean(),
        "tool_calls_mean": df.tool_calls.mean(),
        "latency_p50_ms": df.latency_ms.median(),
        "latency_p95_ms": df.latency_ms.quantile(0.95),
        "usd_per_1000_questions": df.usd.mean() * 1000,
        "gpu_llm_seconds_total": df.llm_seconds.sum(),
        "escalation_rate": df.escalated.mean() if "escalated" in df else float("nan"),
    })
    return row


def diagnosis(df: pd.DataFrame) -> dict:
    """Retrieval and generation joined after both are scored: one cell per question."""
    hit, ok = df.pass_all_ctx, df.correct
    return {
        "hit_correct": int((hit & ok).sum()),
        "hit_wrong": int((hit & ~ok).sum()),  # generation failure on a complete context
        "miss_correct": int((~hit & ok).sum()),
        "miss_wrong": int((~hit & ~ok).sum()),  # retrieval failure
    }


def paired_tests(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Exact McNemar per pair and metric, Holm across all of them."""
    import pandas as pd

    rows = []
    for metric in PAIRED_METRICS:
        for a, b in PAIRS:
            x, y = systems[a][metric], systems[b][metric].reindex(systems[a].index)
            r = ev.mcnemar_exact(x.to_numpy(), y.to_numpy())
            rows.append({"metric": metric, "a": a, "b": b, "rate_a": x.mean(), "rate_b": y.mean(),
                         "delta": y.mean() - x.mean(), **r})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = ev.holm(tests.p_value.tolist())
    tests["significant"] = tests.p_holm < 0.05
    return tests


def strata_table(systems: dict[str, pd.DataFrame], strata: pd.DataFrame) -> pd.DataFrame:
    """Per system and stratum level; a level under MIN_STRATUM questions gets no CI."""
    import pandas as pd

    rows = []
    for system, df in systems.items():
        joined = df.join(strata)
        for stratum in strata.columns:
            for level, g in joined.groupby(stratum):
                row = {"system": system, "stratum": stratum, "level": level, "n": len(g),
                       "pass_all_ctx": g.pass_all_ctx.mean()}
                for name in ("exec_accuracy", "wrong_answer_rate", "refusal_rate"):
                    col = RATES[name]
                    row[name] = g[col].mean()
                    lo, hi = ev.bootstrap_ci(g[col].to_numpy()) if len(g) >= MIN_STRATUM else (
                        float("nan"), float("nan"))
                    row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
                rows.append(row)
    return pd.DataFrame(rows)


def layer_economics(b2_summary: pd.DataFrame, summary: pd.DataFrame, n_chunks: int) -> pd.DataFrame:
    """Each layer against the system without it: Block 2 for retrieval, Block 3 for the agent."""
    import pandas as pd

    b2 = b2_summary[(b2_summary.strategy == STRATEGY) & (b2_summary.setting == "main")]
    b2 = b2.set_index("config")
    rows = []
    for layer, (before, after, one_time) in RETRIEVAL_LAYERS.items():
        gain = (b2.loc[after, "pass_all@10"] - b2.loc[before, "pass_all@10"]) * 100
        latency = b2.loc[after, "latency_p50_ms"] - b2.loc[before, "latency_p50_ms"]
        rows.append({
            "layer": layer, "compared": f"{before} -> {after}", "source": "Block 2, 1147 questions",
            "one_time_cost": one_time.format(n=n_chunks),
            "metric": "pass_all@10", "gain_points": gain, "latency_p50_ms_added": latency,
            "llm_tokens_added": 0.0, "usd_per_1000_added": 0.0,
        })
    s = summary.set_index("system")
    gain = (s.loc["agentic", "exec_accuracy"] - s.loc["fixed", "exec_accuracy"]) * 100
    tokens = [s.loc[x, "prompt_tokens_mean"] + s.loc[x, "completion_tokens_mean"]
              for x in ("fixed", "agentic")]
    rows.append({
        "layer": "agentic_loop", "compared": "fixed -> agentic", "source": "Block 3",
        "one_time_cost": "none", "metric": "exec_accuracy", "gain_points": gain,
        "latency_p50_ms_added": s.latency_p50_ms["agentic"] - s.latency_p50_ms["fixed"],
        "llm_tokens_added": tokens[1] - tokens[0],
        "usd_per_1000_added": s.usd_per_1000_questions["agentic"]
        - s.usd_per_1000_questions["fixed"],
    })
    out = pd.DataFrame(rows)
    # Cost per point is only defined for a gain; a loss or a tie gets no ratio.
    positive = out.gain_points > 0
    out["latency_ms_per_point"] = (out.latency_p50_ms_added / out.gain_points).where(positive)
    out["usd_per_1000_per_point"] = (out.usd_per_1000_added / out.gain_points).where(positive)
    return out


def question_strata(role: str, oracle: pd.DataFrame) -> pd.DataFrame:
    """The protocol strata of each question (statistics.strata)."""
    import pandas as pd

    with QUESTIONS.open() as f:
        types = {r["qid"]: r["gold_type"] for r in csv.DictReader(f) if r["role"] == role}
    b2 = pd.read_parquet(B2_RESULTS / "per_query_contextual.parquet")
    b2 = b2[(b2.strategy == STRATEGY) & (b2.setting == "main")
            & (b2.config == "contextual_hybrid_rerank")].set_index("qid")
    return pd.DataFrame({
        # The oracle context is exactly the distinct gold chunks of the question.
        "n_gold_chunks": oracle.context.map(lambda c: "1" if len(c) == 1 else "2+"),
        "gold_type": pd.Series(types),
        "fixed_retrieval": b2["pass_all@10"].reindex(oracle.index).map({1.0: "hit", 0.0: "miss"}),
    }, index=oracle.index)


def exploratory(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fixed and agentic accuracy on the same subsets of questions, to explain the agent and the
    cascade. Exploratory: defined after the evaluation numbers were seen, never used to rank."""
    import pandas as pd

    fixed, agentic, cascade = systems["fixed"], systems["agentic"], systems["cascade"]
    answered = fixed.status == "answer"
    escalated = cascade.escalated.astype(bool)
    stops = sorted(agentic.stop.unique())
    subsets = [(("agent_stop", stop), agentic.stop == stop) for stop in stops]
    subsets += [
        (("fixed_answer", "grounded"), answered & fixed.grounded),
        (("fixed_answer", "ungrounded"), answered & ~fixed.grounded),
        (("cascade", "kept_fixed"), ~escalated),
        (("cascade", "escalated"), escalated),
    ]
    for signal in ("refusal", "ungrounded", "low_score"):  # overlap: a question can fire two
        subsets.append((("escalation_signal", signal),
                        cascade.escalation_reasons.map(lambda r, s=signal: s in r)))
    rows = []
    for (group, level), mask in subsets:
        rows.append({"group": group, "level": level, "n": int(mask.sum()),
                     "fixed_exec_accuracy": fixed.correct[mask].mean(),
                     "agentic_exec_accuracy": agentic.correct[mask].mean()})
    return pd.DataFrame(rows)


def cmd_report(args: argparse.Namespace) -> None:
    import pandas as pd

    prices = price_table()
    systems = {s: load_system(args.role, s, prices) for s in SYSTEMS}
    qids = load_questions()[args.role]
    for system, df in systems.items():
        if sorted(df.index) != sorted(qids):
            raise SystemExit(f"{system}: rows do not match the {args.role} questions")
        systems[system] = df.loc[qids]
    out = role_dir(args.role)

    summary = pd.DataFrame([{"system": s, **summary_row(df)} for s, df in systems.items()])
    summary.to_csv(out / "summary.csv", index=False)
    diag = pd.DataFrame([{"system": s, **diagnosis(df)} for s, df in systems.items()])
    diag.to_csv(out / "diagnosis.csv", index=False)
    tests = paired_tests(systems)
    tests.to_csv(out / "paired_tests.csv", index=False)
    strata = question_strata(args.role, systems["oracle_context"])
    by_stratum = strata_table(systems, strata)
    by_stratum.to_csv(out / "strata.csv", index=False)
    from ragbench import b2

    n_chunks = len(b2.load_cache(b2.context_cache_path(STRATEGY, "ollama")))
    layers = layer_economics(pd.read_csv(B2_RESULTS / "summary.csv"), summary, n_chunks)
    layers.to_csv(out / "layer_economics.csv", index=False)
    explore = exploratory(systems)
    explore.to_csv(out / "exploratory.csv", index=False)

    # Block 2 gives the fixed hit/miss stratum; the B3 re-run of the same pipeline should agree.
    differ = int((strata.fixed_retrieval.eq("hit") != systems["fixed"].pass_all_ctx).sum())
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["system", "pass_all_ctx", "exec_accuracy", "exec_accuracy_lo", "exec_accuracy_hi",
                "program_valid", "grounded", "refusal_rate", "wrong_answer_rate",
                "context_tokens_p50", "llm_calls_mean", "tool_calls_mean", "latency_p50_ms",
                "latency_p95_ms", "usd_per_1000_questions", "escalation_rate"]
        print(summary[cols].round(3).to_string(index=False), end="\n\n")
        print(diag.to_string(index=False), end="\n\n")
        print(tests.round(4).to_string(index=False), end="\n\n")
        print(by_stratum.round(3).to_string(index=False), end="\n\n")
        print(layers.round(3).to_string(index=False), end="\n\n")
        print(explore.round(3).to_string(index=False), end="\n\n")
    print(f"fixed hit/miss: Block 2 and this run disagree on {differ} of {len(qids)} questions")
    print(f"wrote {out}/{{summary,diagnosis,paired_tests,strata,layer_economics,exploratory}}.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragbench.b3")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("questions").set_defaults(func=cmd_questions)
    r = sub.add_parser("retrieve")
    r.add_argument("--role", choices=ROLES, required=True)
    r.set_defaults(func=cmd_retrieve)
    g = sub.add_parser("generate")
    g.add_argument("--system", choices=GENERATED, required=True)
    g.add_argument("--role", choices=ROLES, required=True)
    g.set_defaults(func=cmd_generate)
    sub.add_parser("threshold").set_defaults(func=cmd_threshold)
    c = sub.add_parser("cascade")
    c.add_argument("--role", choices=ROLES, required=True)
    c.set_defaults(func=cmd_cascade)
    s = sub.add_parser("summary")
    s.add_argument("--role", choices=ROLES, required=True)
    s.set_defaults(func=cmd_summary)
    rp = sub.add_parser("report")
    rp.add_argument("--role", choices=ROLES, required=True)
    rp.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
