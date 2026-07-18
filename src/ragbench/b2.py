"""Block 2 benchmark runner. Implements configs/b2_retrieval_protocol.yaml.

python -m ragbench.b2 data                 # download FinQA test (sha256-checked)
python -m ragbench.b2 contextualize --strategy table_aware --backend ollama
python -m ragbench.b2 run --variant plain       # bm25, dense, hybrid, hybrid_rerank
python -m ragbench.b2 run --variant contextual  # same retrievers on contextualized chunks
python -m ragbench.b2 report                # tables + paired tests from per-query results
python -m ragbench.b2 analyze               # BM25 rescues of dense misses + failure modes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ragbench import analysis, finqa
from ragbench import evaluation as ev
from ragbench.chunking import ChunkedCorpus, chunk_naive, chunk_table_aware
from ragbench.contextual import contextualize, contextualized_text, load_cache, ollama_generate
from ragbench.retrieval import BM25Retriever, DenseRetriever, Reranker, hybrid_search

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs" / "b2_retrieval_protocol.yaml"
RAW = ROOT / "data" / "raw" / "finqa"
INDEXES = ROOT / "data" / "indexes"
CONTEXTS = ROOT / "data" / "contexts"  # versioned: an LLM run is slow and not bit-reproducible
RESULTS = ROOT / "blocks" / "02-retrieval-bench" / "results"

DENSE_MODEL = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
STRATEGIES = ("naive", "table_aware")
SETTINGS = ("main", "raw_question", "company_filter")
KS = (5, 10, 20)
BUDGETS = (1000, 2000)
DEPTH = 50  # every config returns a ranking of this length (enough for the token budgets)
WINDOW = 200
RRF_K = 60
RERANK_TOP_N = 50
OLLAMA_MODEL = "qwen2.5:7b-instruct"


@dataclass
class Bench:
    pages: dict[str, finqa.Page]
    questions: list[finqa.Question]
    names: dict
    tokenizer: object

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def company(self, page_id: str) -> str:
        return finqa.company_name(self.names, page_id)

    def chunk(self, strategy: str) -> ChunkedCorpus:
        if strategy == "naive":
            return chunk_naive(self.pages.values(), self.tokenizer, WINDOW)
        return chunk_table_aware(self.pages.values(), self.count, WINDOW)


def load_bench() -> Bench:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DENSE_MODEL)
    tok.model_max_length = 10**6  # we only count tokens; silence the >512 warning
    pages, questions = finqa.load(RAW / "test.json")
    names = finqa.load_company_names(ROOT / "configs" / "finqa_tickers.csv")
    return Bench(pages, questions, names, tok)


def context_cache_path(strategy: str, backend: str) -> Path:
    model = OLLAMA_MODEL.replace(":", "-") if backend == "ollama" else backend
    return CONTEXTS / f"{strategy}__{model}.jsonl"


def cmd_data(_args) -> None:
    manifest = finqa.download(RAW, ("test",))
    (ROOT / "data" / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def cmd_contextualize(args) -> None:
    bench = load_bench()
    corpus = bench.chunk(args.strategy)
    if args.backend != "ollama":
        sys.exit("only the ollama backend is implemented; Claude comes with the API key")
    generate = ollama_generate(OLLAMA_MODEL)
    start = time.time()

    def progress(n, total):
        if n % 50 == 0 or n == total:
            rate = n / (time.time() - start)
            print(
                f"{n}/{total} chunks  {rate:.2f}/s  eta {(total - n) / rate / 60:.0f} min",
                flush=True,
            )

    contextualize(
        corpus.chunks,
        bench.pages,
        bench.company,
        generate,
        context_cache_path(args.strategy, args.backend),
        progress,
    )


def _embeddings(model, texts: list[str], tag: str) -> np.ndarray:
    """Embeddings cached on disk, keyed by a hash of the model and the exact texts."""
    digest = hashlib.sha256((DENSE_MODEL + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = INDEXES / f"{tag}__{digest}.npy"
    if path.exists():
        return np.load(path)
    emb = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    INDEXES.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def _query_text(bench: Bench, q: finqa.Question, setting: str) -> str:
    if setting == "raw_question":
        return q.question
    return f"{bench.company(q.page_id)}: {q.question}"


def _configs(dense: DenseRetriever, bm25: BM25Retriever, reranker: Reranker) -> dict:
    """The four retrievers of the protocol, each returning a ranking of length DEPTH."""
    return {
        "bm25": lambda q, a: bm25.search(q, DEPTH, a),
        "dense": lambda q, a: dense.search(q, DEPTH, a),
        "hybrid": lambda q, a: hybrid_search(dense, bm25, q, DEPTH, 100, RRF_K, a),
        "hybrid_rerank": lambda q, a: reranker.rerank(
            q, hybrid_search(dense, bm25, q, RERANK_TOP_N, 100, RRF_K, a), DEPTH
        ),
    }


def cmd_run(args) -> None:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.manual_seed(0)
    bench = load_bench()
    dense_model = SentenceTransformer(DENSE_MODEL, device="cuda")
    cross = CrossEncoder(
        RERANK_MODEL, device="cuda", max_length=512, model_kwargs={"torch_dtype": torch.float16}
    )
    prefix = "contextual_" if args.variant == "contextual" else ""
    rows = []
    for strategy in args.strategies:
        corpus = bench.chunk(strategy)
        texts = [c.text for c in corpus.chunks]
        if args.variant == "contextual":
            cache = load_cache(context_cache_path(strategy, args.backend))
            missing = [c.chunk_id for c in corpus.chunks if c.chunk_id not in cache]
            if missing:
                sys.exit(f"{len(missing)} chunks of {strategy} have no context yet")
            texts = [contextualized_text(cache[c.chunk_id], c.text) for c in corpus.chunks]
        # Budgets count the tokens actually sent to the LLM: the indexed text.
        lengths = [bench.count(t) for t in texts]
        dense = DenseRetriever(
            dense_model,
            texts,
            embeddings=_embeddings(dense_model, texts, f"{strategy}_{args.variant}"),
        )
        bm25 = BM25Retriever(texts)
        reranker = Reranker(cross, texts)
        by_ticker: dict[str, list[int]] = {}
        for i, c in enumerate(corpus.chunks):
            by_ticker.setdefault(c.ticker, []).append(i)
        by_ticker = {t: np.array(v) for t, v in by_ticker.items()}

        configs = _configs(dense, bm25, reranker)
        for fn in configs.values():  # warm-up (CUDA kernels, caches) before timing
            for q in bench.questions[:3]:
                fn(_query_text(bench, q, "main"), None)

        for setting in SETTINGS:
            for q in bench.questions:
                gold = corpus.gold_sets(q.page_id, q.gold_keys)
                text = _query_text(bench, q, setting)
                allowed = by_ticker[q.ticker] if setting == "company_filter" else None
                for name, fn in configs.items():
                    t0 = time.perf_counter()
                    ranked = fn(text, allowed)
                    latency = (time.perf_counter() - t0) * 1000
                    row = {
                        "strategy": strategy,
                        "setting": setting,
                        "config": prefix + name,
                        "qid": q.qid,
                        "n_gold": len(gold),
                        "latency_ms": latency,
                        "mrr@10": ev.mrr(ranked, gold, 10),
                    }
                    for k in KS:
                        row[f"pass_all@{k}"] = ev.pass_all(ranked, gold, k)
                        row[f"pass_any@{k}"] = ev.pass_any(ranked, gold, k)
                        row[f"recall@{k}"] = ev.recall(ranked, gold, k)
                    for b in BUDGETS:
                        row[f"pass_all@{b}tok"] = ev.pass_all_budget(ranked, gold, lengths, b)
                    row["ranked_top10"] = json.dumps(ranked[:10])
                    rows.append(row)
            print(f"{strategy} {setting} done", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"per_query_{args.variant}.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    meta = {
        "protocol_sha256": finqa.sha256_file(PROTOCOL),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "dense_model": DENSE_MODEL,
        "rerank_model": RERANK_MODEL,
        "context_backend": args.backend if args.variant == "contextual" else None,
    }
    (RESULTS / f"run_meta_{args.variant}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {out}")


METRICS = [
    "pass_all@5",
    "pass_all@10",
    "pass_all@20",
    "pass_any@10",
    "recall@10",
    "mrr@10",
    "pass_all@1000tok",
    "pass_all@2000tok",
]
ORDER = [
    "bm25",
    "dense",
    "hybrid",
    "hybrid_rerank",
    "contextual_bm25",
    "contextual_dense",
    "contextual_hybrid",
    "contextual_hybrid_rerank",
]


def cmd_report(_args) -> None:
    frames = [pd.read_parquet(p) for p in sorted(RESULTS.glob("per_query_*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    table = []
    for (strategy, setting, config), g in df.groupby(["strategy", "setting", "config"]):
        row = {"strategy": strategy, "setting": setting, "config": config, "n": len(g)}
        for m in METRICS:
            lo, hi = ev.bootstrap_ci(g[m].to_numpy())
            row[m] = g[m].mean()
            row[f"{m}_lo"], row[f"{m}_hi"] = lo, hi
        row["latency_p50_ms"] = g["latency_ms"].median()
        row["latency_p95_ms"] = g["latency_ms"].quantile(0.95)
        table.append(row)
    summary = pd.DataFrame(table)
    summary["config"] = pd.Categorical(summary["config"], ORDER, ordered=True)
    summary = summary.sort_values(["setting", "strategy", "config"])
    summary.to_csv(RESULTS / "summary.csv", index=False)

    # Paired tests (protocol): adjacent layers per chunking, and naive vs table_aware per config.
    main = df[df.setting == "main"]
    pivot = main.pivot_table(index="qid", columns=["strategy", "config"], values="pass_all@10")
    pairs = []
    chain = ["dense", "hybrid", "hybrid_rerank", "contextual_hybrid_rerank"]
    for s in STRATEGIES:
        for a, b in zip(chain, chain[1:], strict=False):
            if (s, a) in pivot and (s, b) in pivot:
                pairs.append((f"{s}: {a} -> {b}", pivot[(s, a)], pivot[(s, b)]))
    for c in chain:
        if ("naive", c) in pivot and ("table_aware", c) in pivot:
            pairs.append(
                (f"{c}: naive -> table_aware", pivot[("naive", c)], pivot[("table_aware", c)])
            )
    tests = []
    for label, a, b in pairs:
        r = ev.mcnemar_exact(a.to_numpy(), b.to_numpy())
        tests.append({"comparison": label, "delta_pass_all@10": b.mean() - a.mean(), **r})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = ev.holm(tests["p_value"].tolist())
        tests["significant"] = tests["p_holm"] < 0.05
    tests.to_csv(RESULTS / "paired_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        cols = [
            "setting",
            "strategy",
            "config",
            "pass_all@5",
            "pass_all@10",
            "pass_all@20",
            "pass_all@1000tok",
            "pass_all@2000tok",
            "latency_p50_ms",
            "latency_p95_ms",
        ]
        print(summary[cols].round(3).to_string(index=False))
        print(tests.round(4).to_string(index=False))


def cmd_analyze(_args) -> None:
    """Protocol list of BM25 rescues of dense misses, plus exploratory failure-mode counts."""
    df = pd.concat(
        [pd.read_parquet(p) for p in sorted(RESULTS.glob("per_query_*.parquet"))],
        ignore_index=True,
    )
    bench = load_bench()
    by_qid = {q.qid: q for q in bench.questions}

    resc = analysis.rescues(df, winner="bm25", loser="dense")
    resc["company"] = [bench.company(by_qid[q].page_id) for q in resc.qid]
    resc["question"] = [by_qid[q].question for q in resc.qid]
    resc.to_csv(RESULTS / "bm25_rescues.csv", index=False)

    page_company = {pid: bench.company(pid) for pid in bench.pages}
    modes = []
    for strategy, g in df.groupby("strategy"):
        corpus = bench.chunk(strategy)
        chunk_page = [c.page_id for c in corpus.chunks]
        for row in g.itertuples(index=False):
            q = by_qid[row.qid]
            mode = analysis.failure_mode(
                json.loads(row.ranked_top10),
                corpus.gold_sets(q.page_id, q.gold_keys),
                chunk_page,
                page_company,
                q.page_id,
            )
            modes.append((strategy, row.setting, row.config, mode))
    counts = (
        pd.DataFrame(modes, columns=["strategy", "setting", "config", "mode"])
        .value_counts()
        .unstack("mode", fill_value=0)
        .reindex(columns=list(analysis.FAILURE_MODES), fill_value=0)
        .reset_index()
    )
    counts.to_csv(RESULTS / "failure_modes.csv", index=False)

    # Granularity: finer chunks spread a question's gold facts over more chunks.
    df["n_gold_bucket"] = df.n_gold.clip(upper=3).map({1: "1", 2: "2", 3: "3+"})
    by_gold = (
        df.groupby(["strategy", "setting", "config", "n_gold_bucket"])
        .agg(**{"n": ("qid", "size"), **{m: (m, "mean") for m in ("pass_all@10", "pass_any@10")}})
        .reset_index()
    )
    by_gold.to_csv(RESULTS / "pass_by_n_gold.csv", index=False)
    with pd.option_context("display.width", 200):
        print(resc.groupby(["strategy", "setting"]).size().rename("bm25_rescues").to_string())
        print(counts[counts.setting == "main"].to_string(index=False))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="ragbench.b2")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data").set_defaults(fn=cmd_data)
    c = sub.add_parser("contextualize")
    c.add_argument("--strategy", choices=STRATEGIES, required=True)
    c.add_argument("--backend", default="ollama")
    c.set_defaults(fn=cmd_contextualize)
    r = sub.add_parser("run")
    r.add_argument("--variant", choices=["plain", "contextual"], required=True)
    r.add_argument("--backend", default="ollama")
    r.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    r.set_defaults(fn=cmd_run)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    sub.add_parser("analyze").set_defaults(fn=cmd_analyze)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
