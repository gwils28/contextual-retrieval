# Contextual Retrieval — Advanced RAG & Agentic RAG, measured

A hands-on, measured study of retrieval quality in RAG systems: dense vs hybrid (BM25 + RRF)
vs cross-encoder reranking vs Contextual Retrieval, on financial filings (FinQA). It also
looks at when to escalate from a fixed pipeline to an agentic retriever.

> **Discipline:** retrieval is evaluated separately from generation. Never judge retrieval
> by looking at the answer.

## Roadmap

| Block | Topic | Status |
|-------|-------|--------|
| [B1 — Watch](blocks/01-watch/) | Contextual Retrieval, A-RAG, advanced retrieval patterns | done |
| [B2 — Retrieval bench](blocks/02-retrieval-bench/) | Pass@k / recall / latency across 4 configs × 2 chunkings | done |
| [B3 — System design](blocks/03-system-design/) | Fixed vs agentic RAG, split evaluation, layer economics | done |
| [B4 — Reading](blocks/04-reading/) | Agentic RAG survey — taxonomy | done |
| [L2 — Armor Gestion](deliverable-2-armor-gestion/) | Client case: table-aware agentic RAG with numeric grounding | planned |

Each block ships a course (`README.md`), executed notebooks, a cheat-sheet and corrected exercises.

## Setup

```bash
uv sync
make test
```

Scope, decisions and working method: [`docs/charter.md`](docs/charter.md).
