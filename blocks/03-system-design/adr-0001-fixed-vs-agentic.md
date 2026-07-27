# ADR-0001 — Fixed retrieve-then-read, not an agentic loop, for FinQA question answering

- **Status:** accepted
- **Date:** 2026-07-27
- **Scope:** numerical question answering over financial filings (FinQA test pages), generator
  `qwen2.5:7b-instruct` run locally
- **Evidence:** protocol [`configs/b3_system_protocol.yaml`](../../configs/b3_system_protocol.yaml)
  (frozen before any run), one evaluation run on 100 stratified test questions (`make b3`),
  tables in [`results/evaluation/`](results/evaluation/). Every number below comes from these
  files; none was computed by hand.

## Context

Block 2 built a retrieval pipeline (hybrid search, reranker, contextual chunks) that puts every
gold fact in the top 10 for 94 % of FinQA questions. Block 3 asks whether the model should then
read those 10 chunks once (**fixed**), or drive its own search with tools (**agentic**: keyword
search, semantic search, read a chunk, read a page; budget 6 tool calls), or use the agent only
when a cheap signal says the fixed answer is doubtful (**cascade**). An **oracle** that receives
exactly the gold chunks measures the generator alone.

The two stages are scored separately: retrieval on the chunks handed to the generator
(`pass_all_ctx`), generation on the executed program (`exec_accuracy`, numeric match).

## Decision

**The fixed pipeline is the default architecture. The agentic loop and the cascade are not
adopted with this generator.** The next effort goes to generation, not to retrieval.

## Evidence

**1. The agent finds slightly more evidence and answers worse.** (`summary.csv`)

| system | all gold in context | exec. accuracy [95 % CI] | wrong answer | refusal | LLM calls | latency p50 | USD / 1000 q |
|---|---|---|---|---|---|---|---|
| oracle | 1.00 | 0.44 [0.34, 0.54] | 0.34 | 0.12 | 1.0 | 1.1 s | 0.35 |
| **fixed** | 0.94 | **0.28** [0.19, 0.37] | 0.40 | 0.05 | 1.0 | **3.2 s** | **1.18** |
| agentic | 0.98 | 0.18 [0.11, 0.26] | 0.31 | 0.33 | 6.3 | 8.8 s | 8.95 |
| cascade | 0.98 | 0.21 [0.13, 0.29] | 0.32 | 0.21 | 5.1 | 9.7 s | 7.02 |

USD: tokens counted by Ollama × the price table of protocol amendment A1 (Together AI, Qwen2.5 7B
Instruct Turbo, $0.30 per million input and output tokens).

**2. No gain of the agent or the cascade is shown.** (`paired_tests.csv`, exact McNemar, Holm
over 6 tests) Fixed vs agentic: −10 points of accuracy (17 questions lost, 7 gained), p = 0.064,
Holm 0.38. Fixed vs cascade: −7 points, Holm 0.38. No difference is significant, in either
direction: with 100 questions, a loss is not shown either. The decision does not rest on a
proven loss. It rests on the burden of proof: the agent costs 7.6× more per question and 2.7×
more latency, so it has to show a gain to be adopted, and it does not.

**3. Generation is the bottleneck, not retrieval.** (`diagnosis.csv`) Fixed has every gold
chunk in context for 94 questions and still gets 67 of them wrong. Even the oracle, with the
exact gold chunks, reaches only 0.44. More evidence cannot help a reader that fails on complete
evidence: the agent lifts `pass_all_ctx` from 0.94 to 0.98, and accuracy goes down.

**4. Why the agent loses** (`exploratory.csv`, exploratory: defined after the results were seen)

- 29 questions exhaust the 6 tool calls. The frozen rule turns them into `insufficient_context`
  (never a forced guess), so they score 0, while fixed answers 38 % of the same questions.
- On the 71 questions where the agent stops by itself, it does as well as fixed (0.25 vs 0.24).
  The loop does not hurt when it ends, but it does not help either.

**5. Why the cascade does not rescue it** (`exploratory.csv`, `summary.csv`)

- It escalates 65 % of the questions. The grounding check (every operand must appear in the chunk
  it cites) fires on 50 of the 95 fixed answers, but it does not separate right from wrong:
  fixed is right on 0.31 of grounded answers and 0.28 of ungrounded ones.
- On the escalated questions the agent scores 0.15 where fixed scored 0.26. The cascade swaps
  fixed answers for weaker ones, and pays for both stages.

**6. Retrieval layers are cheap per point, the agent is not.** (`layer_economics.csv`)

| layer | gain | added latency p50 | cost per point |
|---|---|---|---|
| hybrid (dense → hybrid) | +5.2 pts `pass_all@10` | +3 ms | 0.6 ms |
| reranker | +11.3 pts | +345 ms | 30 ms |
| contextual chunks | +10.5 pts | +91 ms (one-time: 1,702 LLM calls) | 8.7 ms |
| agentic loop (fixed → agentic) | −10 pts exec. accuracy | +5.6 s, +25,900 tokens | — (no gain) |

Retrieval gains come from Block 2 (1,147 questions); the agent's from Block 3.

## Consequences

- **Serve fixed:** one LLM call, about 3 s and $1.2 per 1,000 questions at API prices.
- **Work on the reader:** the 16-point gap between fixed (0.28) and the oracle (0.44), and the
  oracle's own ceiling, are where the accuracy is. Candidates: a stronger generator, fewer and
  better-ordered chunks (fixed sends ~2,500 tokens, the oracle ~260), a better program format.
- **Do not use the grounding check as a confidence signal** with this generator; it may still be
  a useful audit flag on its own.
- **Wrong answers stay high** (0.40 for fixed). In finance a wrong number costs more than a
  refusal; any deployment needs a human check or an abstention rule that is tested first.

## Alternatives considered

- **Agentic loop as default:** rejected (points 1, 2, 4).
- **Cascade, fixed first:** rejected (point 5). Its escalation rule is frozen; changing the
  signals after seeing these numbers would fit the rule to them.
- **Forcing an answer when the agent's budget runs out:** not tested. It would have changed a
  frozen protocol rule after seeing the development results.

## Limits

- 100 questions: CIs are about ±9 points; strata under 20 questions are descriptive only.
- One generator, a 7B open model; no stronger reference model was run (owner's decision). The
  conclusion is about this generator, not about agentic RAG in general.
- FinQA mostly needs one or two facts from a single page that the fixed pipeline already finds.
  An agent is meant for open, multi-hop search across documents, which this benchmark rarely asks.
- Prompts were designed on 100 other test questions that share some pages with the evaluation set.

## Revisit when

- the generator changes (a stronger model could stop on its own and use the extra evidence);
- questions need evidence from several documents, or retrieval stops finding the gold facts
  (fixed retrieval misses only 6 of 100 here);
- a confidence signal is shown, on development questions, to separate right from wrong answers.
