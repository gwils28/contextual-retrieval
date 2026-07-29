# Block 4 — Exercises

Try each question before opening the answers at the bottom. E1–E3 are **pen and paper**; E4
uses `ragbench` and [`results/process_metrics.csv`](results/process_metrics.csv).

## Control questions

**Q1.** Name the five RAG paradigms of the survey, and say which one our fixed pipeline belongs to.

**Q2.** A colleague calls "naïve RAG" a pipeline that embeds chunks and takes the top 5. Is it
naïve RAG in the survey's sense? Why does it matter?

**Q3.** What is the difference between a *workflow* and an *agent*? Classify the Block 3 cascade
and the Block 3 loop.

**Q4.** Which two agentic patterns does the survey call less predictable, and why would that
matter for a bank?

**Q5.** Our grounding check evaluates every answer. Is it *reflection*? What would turn it into an
evaluator–optimizer loop?

**Q6.** Why is our Block 3 agent an awkward fit for the survey's "single-agent router" box?

**Q7.** Adaptive-RAG and our cascade both route. What do they route on, and when?

**Q8.** Lesson 10.3 says retrieval remains the primary bottleneck. Give the Block 3 number that
contradicts it here, and explain why the lesson is still useful.

**Q9.** What does process-level evaluation reveal about the Block 3 agent that its accuracy
(0.18) did not?

**Q10.** Give two reasons not to choose an architecture from the survey's Table 3 alone.

## Exercises

**E1 — Name the architecture (pen and paper).** For each design, give the closest survey box and
the workflow pattern(s):
(a) a classifier sends questions about figures to a Text-to-SQL engine and the rest to vector
search; (b) a model drafts an answer, a second model grades its citations, and the draft is
revised until the grade passes or 3 rounds are used; (c) a top-level agent splits "compare the
2023 margins of five competitors" into five sub-questions handled by five retrieval agents, then
merges.

**E2 — Add the missing dimension.** Write the *autonomy* row of Table 3 for our three Block 3
systems: who decides the next step, and what bounds it?

**E3 — Fix the process, on paper.** Using the process metrics (97 % of `chunk_read` re-read the
initial top-10; 77 % of searches surface a new candidate), propose two changes to the tool layer
or the prompt. For each, say what you would measure on the *development* questions to accept it,
and why it must not be tried on the evaluation questions.

**E4 — Process metrics on another run (code).** Apply `ragbench.b4.process_metrics` to the
development traces in `blocks/03-system-design/results/development/history/` (agent prompts v1
and v2). Did v2 ("most questions need no tool") change the share of empty reads? Compare with
the development log's finding (v2 re-read even more).

---

## Answers

**Q1.** Naïve, advanced, modular, graph, agentic. Fixed is **modular RAG**: hybrid BM25 + dense
retrieval, RRF fusion, a reranker, contextual chunks — interchangeable components.

**Q2.** No: for the survey, naïve RAG uses *keyword* retrieval (TF-IDF, BM25); dense retrieval is
already "advanced RAG". It matters because two people can agree on "naïve" while describing
different systems; always define the term.

**Q3.** A workflow follows code paths written in advance; an agent decides its own path at run
time. The cascade is a workflow (routing by a coded rule); the loop is an agent (the model picks
tools and when to stop).

**Q4.** Planning and multi-agent collaboration. A bank needs predictable, auditable behaviour;
less predictable patterns are harder to test, to explain to a regulator, and to bound in cost.

**Q5.** No: reflection means the agent critiques *and revises*. Our check only reports. Sending
its verdict back to the generator ("operand 5735 is not in chunk 2, fix the citation") and
re-generating, within a round limit, would make it an evaluator–optimizer loop.

**Q6.** The survey's router chooses between *sources* (SQL, vector store, web). Ours searches one
corpus and chooses *granularity* (search snippets, a chunk, a whole page) — A-RAG's hierarchical
retrieval interfaces, which the taxonomy does not have a box for.

**Q7.** Adaptive-RAG routes **before** retrieval, on the predicted *complexity* of the query. Our
cascade routes **after** the fixed answer, on *confidence* signals (refusal, grounding, reranker
score).

**Q8.** Fixed had every gold chunk for 94 of 100 questions and got 67 of those wrong (oracle:
0.44). Still useful as an *order*: fix retrieval first (Block 2 did), then measure again — the
bottleneck moves.

**Q9.** *Why* it loses: it re-reads chunks already in its prompt (97 % of `chunk_read`), 81 % of
reads add nothing, and the budget runs out on re-reading — while its searches do find new
candidates. Accuracy only says that it loses.

**Q10.** Its ratings are the authors' judgements, not measurements; and they are generic, while
the right choice depends on your data, questions, model and costs — which only a protocol on your
own questions measures.

**E1.** (a) Single-agent router (or adaptive RAG if the classifier is about complexity);
**routing** workflow. (b) Corrective / evaluator-based; **evaluator–optimizer** loop with a round
budget. (c) Hierarchical (or multi-agent with a coordinator); **orchestrator–workers** with
**parallelization** (sectioning).

**E2.** Fixed: nobody decides — the chain is fixed in code. Agentic: the LLM decides each tool
call and when to stop; bounded by 6 tool calls and 8,000 tokens, an exhausted budget → refusal.
Cascade: a coded rule decides escalation (no autonomy in the router); after escalation, the
agent's bounded autonomy applies.

**E3.** Examples: (1) make `chunk_read` return "already in your evidence" for chunks in the pool
without spending budget, or not offer it for them; (2) list in each agent turn the ids already
held and the new candidates found by the last search. Measure on development questions: share of
empty reads, budget exhaustions, and — the real target — accuracy and refusals against fixed.
Trying it on evaluation questions would fit the design to the test set; their single run is spent.

**E4.** Sketch:

```python
import json
from pathlib import Path
from ragbench import b4

H = Path("blocks/03-system-design/results/development/history")
for f in sorted(H.glob("agentic__*.jsonl")):
    qs = []
    for row in map(json.loads, f.open()):
        initial = row["context"][: len(row["context"]) - row["added_chunks"]]
        qs.append(b4.process_metrics(row["trace"], initial))
    s = b4.summarize(qs)
    print(f.name, round(s["empty_read_share"], 3), round(s["reread_of_initial_share"], 3))
```

Expected: empty reads go from 0.83 (v1) to 0.87 (v2), and `chunk_read` calls from 232 to 309,
while page reads drop from 56 to 14. Same story as the development log (224 of 232 `chunk_read`
useless with v1, 278 of 309 with v2): asking the model to stop earlier did not stop it from
re-reading; it just read pages less. (Exploratory: development data.)
