# Project charter — Advanced RAG & Agentic RAG

> Outcome of the alignment interview (5 rounds). This document is the contract between
> the human (owner, decision-maker) and the agent (Claude Code). Any change to a decision
> below is recorded in the [Decision log](#decision-log) with its reason.

## 1. Goal

Learn **in depth** — to the point of mastery — how to engineer retrieval quality in RAG
systems, through the four blocks of the study plan (private, kept out of the repo) and the fictional
client case *Armor Gestion*:

1. **Engineer and measure** retrieval quality (dense → hybrid → rerank → contextual).
2. **Know when to escalate** from a fixed retrieve-then-read pipeline to agentic retrieval.
3. **Evaluate retrieval separately from generation.**

Starting point: comfortable with ML/DS, **beginner in RAG / information retrieval**. Every
block therefore starts from fundamentals (embeddings, BM25, IR metrics) before going deep.

## 2. Deliverables

| # | Block | Deliverable |
|---|-------|-------------|
| B1 | Watch | Pedagogical synthesis of 3 sources (Anthropic Contextual Retrieval, A-RAG, dev.to advanced patterns) |
| B2 | Engineering | Retrieval benchmark on FinQA: 4 configs × 2 chunking strategies, Pass@5/@10, recall, latency, CIs |
| B3 | System design | Architecture decision record: fixed vs agentic RAG, split retrieval/generation eval, layer economics |
| B4 | Reading | Pedagogical synthesis of the Agentic RAG survey (Singh et al., v4 2026) — taxonomy as a mental map |
| L2 | Client case | Design dossier (ADRs, SLOs, costs, AMF/GDPR) **+ executable local POC** |

**Every block ships four pedagogical formats:**

- `README.md` — detailed course: concepts, intuition, formulas, diagrams, pitfalls, *deviations from plan*.
- `notebooks/` — executed, versioned with outputs; they call the package, no business logic inside.
- `cheatsheet.md` — one-page revision sheet per notion + FR/EN glossary entries.
- `exercises.md` — control questions and small exercises, with corrected answers.

## 3. Technical decisions

| Topic | Decision |
|-------|----------|
| Corpus | **FinQA** (`ibm-research/finqa`, HF). Golden = annotated supporting facts (`gold_inds`). |
| Search scope | **Pooled corpus**: all chunks from all reports in one index (production-realistic). |
| Golden set | **25 hand-audited queries** (pedagogy) **+ all 1,147 FinQA test questions** (statistics). |
| Statistics | Bootstrap 95 % CIs per metric; paired test between configs on the same queries. |
| Chunking | **Compare 2 strategies**: naive fixed token window (breaks tables) vs table-aware (row + header). |
| Dense model | `BAAI/bge-base-en-v1.5` |
| Lexical | `rank-bm25` (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion, hand-written |
| Reranker | `BAAI/bge-reranker-v2-m3` (fp16, GPU) — top-N=50 → top-k |
| LLM | **Hybrid**: local (Ollama in Docker, ~7-8B quantized) to iterate; Claude API for reference runs, behind one abstract interface |
| Claude budget | **≤ 20 USD** total; Haiku-class for contextualization, prompt caching on. Key provided later. |
| Agentic layer | **LangGraph** directly, A-RAG style tools (`keyword_search`, `semantic_search`, `chunk_read`/`table_read`), bounded iterations + stop condition |
| Language | **English** for code, docs and commits (chat with the agent in French) |
| Python | 3.12, managed by `uv`, versions pinned in `uv.lock` |

**Watch (B1) / reading (B4) depth:** pedagogical synthesis (clear summaries, diagrams), not
a line-by-line audit of every figure.

## 4. Working method

Based on the owner's article *Claude Code for data science*:

- **Loop:** explore → plan → code → verify → commit. Nothing is done without proof.
- **Gate per block:** the agent submits the **plan + eval protocol (YAML)** *before* coding,
  then **results + diff** *before* committing. The owner decides at each gate.
- **Protocol before results:** the eval protocol is frozen in `configs/` before any number is seen.
- **Numbers come from versioned code only:** `make` targets, fixed seeds, pinned versions,
  `sha256` of raw data in a manifest.
- **One Python package** (`src/ragbench`), tested; notebooks explain, they do not implement.
- **One subject per commit**; full diff review before each commit.

## 5. Git & GitHub

- Public repo **`gwils28/contextual-retrieval`**.
- **Sole contributor:** author = committer = `wilson goma <10559068+gwils28@users.noreply.github.com>`;
  no `Co-Authored-By` or tool trailer, ever.
- **Backdated commits** starting `2026-07-15`, spread over days at plausible hours:
  `GIT_AUTHOR_DATE=... GIT_COMMITTER_DATE=... git commit -m "..."`.
- **First push** once the owner has reviewed and validated the scaffold (this charter,
  `CLAUDE.md`, repo layout). Then regular pushes at each validated gate.

## 6. Out of scope (for now)

- Paid embedding/rerank APIs (Voyage, Cohere).
- Full production stack (OpenSearch, Qdrant cluster, dashboards) — designed in L2, POC stays local.
- Reproducing the Anthropic cookbook notebook end-to-end.

## Decision log

| Date | Decision | Reason |
|------|----------|--------|
| 2026-07-15 | Charter adopted | Alignment interview |
| 2026-07-15 | Study plan kept out of the public repo | Owner's choice: private source document |
| 2026-07-17 | Golden set: all 1,147 test questions instead of ~300 | Contextualization cost scales with the corpus, not the queries; more queries = more statistical power for free |
| 2026-07-17 | Queries prefixed with the company name (`"{company}: {question}"`) | FinQA questions rarely name the company and are ambiguous in a pooled index; no year in the prefix because FinQA folder years do not reliably match fiscal years |
| 2026-07-17 | Primary metric `pass_all@k` (all gold facts in top-k) | A numeric answer needs all its operands |
