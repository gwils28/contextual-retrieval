"""Block 3 runner (fixed vs agentic RAG). Implements configs/b3_system_protocol.yaml.

python -m ragbench.b3 questions                 # draw the question sets (seed 0)
python -m ragbench.b3 retrieve --role development  # fixed top-10 + reranker scores (GPU)
python -m ragbench.b3 generate --system fixed --role development   # needs Ollama
python -m ragbench.b3 generate --system agentic --role development # needs Ollama
python -m ragbench.b3 summary --role development   # quick table (no CI), for development only

Retrieval runs first and is saved, then the GPU is freed before the LLM is loaded: on an 8 GB
GPU, Qwen 7B, the embedder and the reranker do not fit together comfortably.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from ragbench import evaluation as ev
from ragbench import finqa
from ragbench import generation as G
from ragbench.audit import audited_sample, gold_type

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "b3_system_protocol.yaml"
QUESTIONS = ROOT / "configs" / "b3_questions.csv"
RAW = ROOT / "data" / "raw" / "finqa"
RESULTS = ROOT / "blocks" / "03-system-design" / "results"

N_QUESTIONS = 100
SEED = 0
ROLES = ("development", "evaluation")
STRATEGY = "naive"  # Block 2 winner, with its versioned Qwen contexts
TOP_K = 10
SYSTEMS = ("oracle_context", "fixed", "agentic")
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
            "llm_s_p50": df.llm_seconds.median(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ragbench.b3")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("questions").set_defaults(func=cmd_questions)
    r = sub.add_parser("retrieve")
    r.add_argument("--role", choices=ROLES, required=True)
    r.set_defaults(func=cmd_retrieve)
    g = sub.add_parser("generate")
    g.add_argument("--system", choices=SYSTEMS, required=True)
    g.add_argument("--role", choices=ROLES, required=True)
    g.set_defaults(func=cmd_generate)
    s = sub.add_parser("summary")
    s.add_argument("--role", choices=ROLES, required=True)
    s.set_defaults(func=cmd_summary)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
