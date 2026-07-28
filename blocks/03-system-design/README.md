# Block 3 — System design: fixed vs agentic RAG, measured stage by stage

> **Goal of this block.** Block 2 measured retrieval alone. This block adds the generator and asks
> an architecture question: should the model read the retrieved chunks once (**fixed**), drive its
> own search with tools (**agentic**), or call the agent only when a cheap signal says the fixed
> answer is doubtful (**cascade**)? Retrieval and generation are scored **separately**, and every
> option gets a cost per query. The output is a decision record, not a leaderboard.
>
> **Material.** This course · notebooks: [01 the systems](notebooks/01_systems.ipynb),
> [02 results](notebooks/02_results.ipynb) · [cheat-sheet](cheatsheet.md) ·
> [exercises](exercises.md) · decision record [ADR-0001](adr-0001-fixed-vs-agentic.md) · protocol
> [`configs/b3_system_protocol.yaml`](../../configs/b3_system_protocol.yaml) · results in
> [`results/evaluation/`](results/evaluation/) · development log
> [`results/development/`](results/development/README.md) · reproduce with `make b3`.

> **Results in five lines** (100 test questions, generator `qwen2.5:7b-instruct`, exact match of
> the executed answer):
>
> 1. **Fixed is the cheapest system and has the highest accuracy estimate: 0.28**, one LLM call,
>    3.2 s, $1.2 per 1,000 questions. Agentic: 0.18, 6.3 calls, 8.8 s, $9.0. Cascade: 0.21.
> 2. **No difference is significant after Holm** (fixed vs agentic: −10 points, Holm p = 0.38).
>    The agent is not adopted because it costs 7.6× more and shows no gain, not because a loss is
>    proven.
> 3. **Generation is the bottleneck.** Fixed has every gold chunk in context for 94 questions and
>    gets 67 of them wrong. Even with exactly the gold chunks (oracle), the 7B model reaches 0.44.
> 4. **The agent fails on its stop rule:** 29 questions exhaust the 6 tool calls and end as a
>    refusal; when it stops by itself it ties with fixed (0.25 vs 0.24).
> 5. **The cascade's main signal does not work:** the grounding check fires on half the answers
>    and does not separate right from wrong (0.31 vs 0.28), so 65 % of questions go to the weaker agent.

---

## 1. From retrieval to answers: what changes

In Block 2 a question was a success if its gold chunks were in the top-k. Nobody read them. Now a
**generator** (an LLM) reads the chunks and must produce a number. Two new things can go wrong:

- the right chunks are there, but the model **misreads** them (wrong row, wrong year, wrong
  operation);
- the model **answers** when it should have said "I don't know", or the reverse.

A RAG system has two stages, and an end-to-end accuracy mixes them. A score of 0.28 says nothing
about *where* the 72 failures come from. So the rule of this project holds here too: **retrieval
is judged on the chunks handed to the generator, never on the answer; generation is judged on the
answer.** The two are joined only afterwards, in a diagnosis table (section 6.3).

## 2. Three architectures

```
fixed      question ──► retrieve top-10 ──► generator ──► program ──► execute ──► answer
                         (Block 2 best)       (1 call)

agentic    question ──► retrieve top-10 ──► agent loop ──────────────► generator ──► ...
                                            │ keyword_search  ▲         (on the evidence pool)
                                            │ semantic_search │
                                            │ chunk_read      │  ≤ 6 tool calls
                                            └ page_read ──────┘  then DONE, or budget out

cascade    question ──► fixed ──► signal fires? ── no ──► fixed answer
                                       └── yes ─► agentic answer (pays for both)
```

### 2.1 Fixed: retrieve-then-read

The default RAG design: one retrieval, one LLM call. The retriever is Block 2's winner,
`contextual_hybrid_rerank` (BM25 + dense fused by RRF, reranked by a cross-encoder, over chunks
that carry an LLM-written context), with naive 200-token chunks. Top 10 chunks, about 2,500 tokens.

### 2.2 Agentic: the model drives the search

An **agent** is an LLM in a loop that can call **tools** and decides itself when to stop. Ours
follows the A-RAG pattern: the model is given search and read tools over the same corpus, at
several granularities.

| tool | what it returns |
|---|---|
| `keyword_search(query)` | 5 chunk ids + the start of each chunk (BM25) |
| `semantic_search(query)` | 5 chunk ids + the start of each chunk (dense) |
| `chunk_read(chunk_id)` | the full chunk, added to the evidence |
| `page_read(chunk_id)` | every chunk of that page, in order, added to the evidence (*small-to-big*) |

Three design choices, all frozen in the protocol before any run:

- **It starts from fixed's top-10.** The agent can only *add* evidence, never start from less.
  So any loss against fixed comes from the loop itself, not from a worse start.
- **The agent gathers, it does not answer.** When it replies `DONE`, the *same* generator as fixed
  answers from the evidence pool. Fixed and agentic then differ only by the context they hand over.
- **Budget: 6 tool calls, 8,000 tokens. An exhausted budget means `insufficient_context`**, never a
  forced guess. In finance a guessed number is worse than an honest "I don't know".

The loop is a small graph written with LangGraph (`ragbench.agent`):

```
START ─► agent ──(tool calls)──► tools ──(budget left)──► agent
           └──(no tool call: DONE)──► END     └──(budget exhausted)──► END
```

### 2.3 Cascade: pay for the agent only when needed

An agent costs several LLM calls. A **cascade** runs the cheap system first and escalates only
when a cheap signal says the answer is doubtful. Ours escalates if **any** of three signals fires
on fixed's output:

1. fixed refused (`insufficient_context`);
2. fixed's answer fails the **grounding check** (section 3.3);
3. the reranker's top-1 score is below a threshold, set on the development questions as the score
   that flags the 20 % least confident ones (0.8886, frozen in `configs/b3_cascade.yaml`).

An escalated question gets the agent's answer — even a refusal — and pays for both stages.

### 2.4 Oracle: the generator's ceiling

The **oracle** is not a system you could deploy. It receives exactly the gold chunks of each
question (about 260 tokens), so its score measures the generator alone, with perfect retrieval.
The gap between fixed and the oracle is what better retrieval or better context could still buy.

## 3. The generator: programs, not numbers

### 3.1 The LLM never does arithmetic

LLMs are unreliable at arithmetic, and a 7B model even more so. FinQA ships every answer with a
**program** in a tiny language (DSL), for example:

```
subtract(5829, 5735), divide(#0, 5735)        # percentage change: (5829 − 5735) / 5735
```

`#0` is the result of the first step. So the generator writes the program and `ragbench.program`
executes it. The answer is right if the executed value matches FinQA's within 0.5 %
(`|pred − gold| ≤ 0.005 · max(|gold|, 1)`; 25 and 0.25 are both accepted, since FinQA mixes
percentages and fractions).

**Pitfall: the tolerance is loose on small answers.** Because of `max(|gold|, 1)`, any answer below
1 — most ratios and growth rates — gets an *absolute* tolerance of 0.005. A growth rate of 0.0117
against a gold 0.0120 (2.7 % relative error) is scored correct; notebook 01 shows a real case. The
rule was frozen before the runs and is the same for every system, but absolute accuracies are
optimistic, and a system whose near-misses fall closer to the gold could be favoured (not measured).
A relative tolerance would be the better choice next time.

### 3.2 Structured output: each number carries its citation

The generator returns JSON constrained by a **JSON schema** (Ollama enforces it while decoding):

```json
{"status": "answer",
 "steps": [{"op": "subtract", "arg1": "5829", "arg1_chunk": 2, "arg2": "5735", "arg2_chunk": 2},
           {"op": "divide",   "arg1": "#0",   "arg1_chunk": 0, "arg2": "5735", "arg2_chunk": 2}]}
```

Each step has one operation (an enum of the schema) and exactly two arguments, and each argument
says which chunk it was copied from. This format is the result of four prompt versions on the
development questions (oracle accuracy 0.26 → 0.38 → 0.48 → 0.50): free text let the 7B model
nest operations and invent chunk ids; making the citation part of each argument fixed both.

### 3.3 The grounding check

An answer is **grounded** if every number of its program appears, after normalization ("$ 5,829"
→ 5829), in the chunk it cites. It is a cheap, deterministic check that needs no gold answer, so
it looks like a natural confidence signal for the cascade. Section 7.4 shows why it is not one.

**Pitfall.** A program with no number at all (`divide(#0, #1)`, referencing steps that do not
exist) is vacuously grounded: there is nothing to check. Grounded is not the same as valid.

## 4. Protocol: decisions frozen before the first answer

Everything in `configs/b3_system_protocol.yaml` was frozen on 2026-07-23, before any generation:

- **Two disjoint sets of 100 test questions**, stratified by gold type (table / text /
  table+text). The **development** set is used to design prompts, tools and the threshold. The
  **evaluation** set is run **once**, at the end, with nothing changed after seeing it.
- Systems, budgets, stop rule, escalation signals, metrics, statistics, strata.
- One amendment (A1), the API price table, filled before the evaluation run.

Why 100 questions and not 1,147? Each question costs up to 8 LLM calls for the agent. The owner
accepted wider intervals (about ±9 points on a rate near 0.5) to keep runs short. Consequence: only
large differences can be shown, and a non-significant gap is reported as **"not shown"**, never as
"no difference".

### 4.1 What the development set taught (and why it stopped)

Development is where you are allowed to look and change things. The log is in
[`results/development/README.md`](results/development/README.md). Two lessons:

- **A stopping rule protects you from yourself.** Prompt v4 changed accuracy by 2 points over v3,
  within the noise of 100 questions. Iterating further would have fitted the prompt to those 100.
- **The agent's weakness was visible in development** (29 budget exhaustions, 0.25 vs 0.31 for
  fixed). Changing the frozen stop rule then would have tuned the protocol to the outcome. It was
  kept, and the evaluation confirmed the pattern on new questions.

## 5. Metrics

**Retrieval stage** (only which chunks the generator saw):

| metric | meaning |
|---|---|
| `pass_all_ctx` | every gold chunk is in the context handed to the generator |
| `pass_any_ctx` | at least one is |
| `context_tokens` | size of that context |

**Generation stage** (only the output):

| metric | meaning |
|---|---|
| `exec_accuracy` | the executed program matches the gold answer |
| `program_valid` | the program parses and runs |
| `grounded` | every number appears in the chunk it cites |
| `refusal_rate` | `insufficient_context` |
| `wrong_answer_rate` | answered with a valid program, and wrong — **the costly error in finance** |

A refusal and a wrong answer both score 0 on accuracy, but they are not the same failure: a
refusal sends the question to a human, a wrong number goes into a report.

**Cost:** LLM calls, prompt and completion tokens (counted by Ollama), tool calls, end-to-end
latency p50/p95 at batch size 1, and a **USD equivalent**: tokens × the price of the same model on
an API (Together AI, Qwen2.5 7B Instruct Turbo, $0.30 per million tokens in and out). It lets a
local run be compared with a deployment.

## 6. Statistics on 100 questions

The tools are those of Block 2 (bootstrap intervals, exact McNemar, Holm); what changes is the
sample size.

### 6.1 Intervals are wide

With n = 100 and a rate near 0.3, the 95 % bootstrap interval spans about ±9 points. Fixed's 0.28
is [0.19, 0.37]; agentic's 0.18 is [0.11, 0.26]. The intervals overlap, but that alone decides
nothing: paired tests are more sensitive.

### 6.2 Paired tests and Holm

All systems answer the **same** questions, so they are compared with the exact McNemar test, which
counts only **discordant** questions. Fixed vs agentic: fixed alone is right on 17, agentic alone
on 7. Under "no difference", 17 vs 7 out of 24 gives p = 0.064.

Six tests are run (three pairs × accuracy and wrong answers). **Holm** corrects for running six:
the smallest p-value is multiplied by 6, the next by 5, and so on. 0.064 becomes 0.38. Nothing is
significant.

**How to read a non-significant result.** It does not mean the systems are equal. It means 100
questions cannot tell: a true gap of 10 points needs more data to be shown. The decision then rests
on something else — here, cost (section 8).

### 6.3 The diagnosis table: joining the stages after scoring

Each question falls into one cell:

| | answer right | answer wrong |
|---|---|---|
| **all gold chunks in context** | success | **generation failure** |
| **some gold chunk missing** | lucky (or other valid evidence) | **retrieval failure** |

For fixed: 27 / **67** / 1 / 5. Sixty-seven generation failures against five retrieval failures:
improving retrieval further cannot fix this system.

### 6.4 Strata

Results are also split by number of gold chunks (1 vs 2+), gold type, and whether fixed retrieval
found everything (hit vs miss, known from Block 2). A stratum under 20 questions gets counts only,
no interval. The miss stratum has 6 questions: too few to say whether the agent helps where
retrieval failed — the question an agent is supposed to answer.

## 7. Results

### 7.1 The main table

| system | all gold in context | exec. accuracy [95 % CI] | wrong answer | refusal | LLM calls | latency p50 | USD / 1000 q |
|---|---|---|---|---|---|---|---|
| oracle | 1.00 | 0.44 [0.34, 0.54] | 0.34 | 0.12 | 1.0 | 1.1 s | 0.35 |
| **fixed** | 0.94 | **0.28** [0.19, 0.37] | 0.40 | 0.05 | 1.0 | 3.2 s | 1.18 |
| agentic | 0.98 | 0.18 [0.11, 0.26] | 0.31 | 0.33 | 6.3 | 8.8 s | 8.95 |
| cascade | 0.98 | 0.21 [0.13, 0.29] | 0.32 | 0.21 | 5.1 | 9.7 s | 7.02 |

Source: `results/evaluation/summary.csv` (`make b3-report`).

### 7.2 More evidence, worse answers

The agent lifts `pass_all_ctx` from 0.94 to 0.98: it does find missing chunks. Yet accuracy drops.
Retrieval was not the problem (section 6.3), so more evidence could not help, and the loop adds a
new way to fail. Note also that the agent's lower wrong-answer rate (0.31 vs 0.40) is not a
virtue: it comes from refusing a third of the questions.

### 7.3 Why the agent loses: the stop rule

Split by how the loop ended (`results/evaluation/exploratory.csv`):

| agent stop | questions | agentic accuracy | fixed accuracy, same questions |
|---|---|---|---|
| replied DONE | 71 | 0.25 | 0.24 |
| tool budget exhausted | 29 | 0.00 | 0.38 |

When it stops by itself, the agent ties with fixed. When it does not, it scores 0 by design, on
questions fixed often gets right. The development traces show why: the 7B model keeps calling
`chunk_read` on chunks it already has, instead of replying `DONE`. A weak model does not know when
it knows enough.

*This split is exploratory: it was defined after the results were seen. It explains the outcome; it
never ranks systems.*

### 7.4 Why the cascade does not rescue it: a signal that does not separate

A cascade is only as good as its signal. The signal should fire on wrong answers and stay quiet on
right ones. On the 95 questions fixed answered:

| fixed answer | questions | fixed accuracy |
|---|---|---|
| grounded | 45 | 0.31 |
| ungrounded | 50 | 0.28 |

The check fires on half the answers, and right and wrong answers fail it almost equally (in
development the failures were mostly wrong *citations*, not invented numbers). So the cascade
escalates 65 questions, on which the agent scores 0.15 where fixed scored 0.26. It replaces
answers by weaker ones and pays for both.

**Lesson:** before using a signal to route traffic, check on development data that it actually
separates right from wrong. Here that check was made, the signal failed it, and the frozen rule was
kept anyway — reporting a bad pre-registered design is worth more than a good post-hoc one.

### 7.5 Strata

Descriptive highlights (`results/evaluation/strata.csv`):

- **2+ gold chunks** (20 questions): agentic 0.25 vs fixed 0.20 — the only stratum where the agent
  is ahead, and the one where its search should matter. Intervals [0.10, 0.45] and [0.05, 0.40]:
  nothing is shown.
- **Fixed retrieval miss** (6 questions): too few to conclude.

## 8. Layer economics: cost per point gained

Each layer is compared to the system without it (`results/evaluation/layer_economics.csv`):

| layer | gain | added latency p50 | latency per point | one-time cost |
|---|---|---|---|---|
| hybrid (dense → hybrid) | +5.2 pts `pass_all@10` | +3 ms | 0.6 ms | BM25 index (CPU) |
| reranker | +11.3 pts | +345 ms | 30 ms | none |
| contextual chunks | +10.5 pts | +91 ms | 8.7 ms | 1,702 LLM calls |
| agentic loop | −10 pts accuracy | +5.6 s, +25,900 tokens | — | none |

Retrieval gains come from Block 2 (1,147 questions); the agent's from this block. The retrieval
layers buy points for milliseconds. The agent costs seconds and thousands of tokens per question,
and buys nothing measurable here. **Contextual chunks are the cheapest per point at query time**:
their cost is paid once, at ingestion.

## 9. The decision

See [ADR-0001](adr-0001-fixed-vs-agentic.md). In short: **fixed is the default; the agent and the
cascade are not adopted with this generator; the next effort goes to the reader.** An ADR
(*Architecture Decision Record*) writes down the context, the decision, the evidence, the
alternatives and — most useful later — **when to revisit it**: a stronger generator, questions
spanning several documents, or a confidence signal that is shown to work.

## 10. Deviations from plan and limits

- **No Claude reference run** (owner's decision): every number comes from one 7B open model. The
  conclusion is about this generator, not about agentic RAG in general. A stronger model might
  stop on its own and use the extra evidence.
- **100 questions:** wide intervals, small strata.
- **FinQA favours fixed:** questions need one or two facts from a page that fixed retrieval already
  finds 94 % of the time. Agents are meant for open, multi-hop search, which this benchmark rarely
  asks for.
- **Development and evaluation share some pages**, so prompts may fit those pages slightly.
- **Latency:** measured on a laptop GPU (RTX 5070, 8 GB) at batch size 1; the cost of
  contextualization was counted in LLM calls only (its tokens were not recorded in Block 2).
- **Loose numeric match on answers below 1** (section 3.1): absolute accuracies are optimistic;
  the rule is shared by all systems, but its effect on each was not measured.
- **Exploratory tables** (`exploratory.csv`) were added after the results were seen; they explain,
  they do not rank.

## 11. What to tell a client

- **Start with retrieve-then-read.** Measure it stage by stage before adding an agent.
- **Find the bottleneck first.** If the model fails on complete context, better search will not
  help; work on the reader (model, context size and order, output format).
- **An agent must earn its cost.** Several LLM calls per question, seconds of latency, and a new
  failure mode (not knowing when to stop). Demand a measured gain on your own questions.
- **Test escalation signals before routing on them.** A signal that does not separate right from
  wrong answers sends traffic to the more expensive path for nothing.
- **Track wrong answers separately from refusals.** In finance they do not cost the same.
