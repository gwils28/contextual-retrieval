# Block 4 — Reading: the Agentic RAG survey as a map, checked against our own territory

> **Goal of this block.** Read the reference survey on agentic RAG and turn it into a **mental
> map**: a vocabulary to *name* an architecture instead of saying "we'll add agents". Then do
> what a survey cannot do for us: check its lessons against the measurements of Blocks 2 and 3.
>
> **Source.** A. Singh, A. Ehtesham, S. Kumar, T. Talaei Khoei, A. V. Vasilakos, *Agentic
> Retrieval-Augmented Generation: A Survey on Agentic RAG*, arXiv:2501.09136, **v4, 1 April
> 2026** ([abs](https://arxiv.org/abs/2501.09136)). Section numbers below refer to v4.
>
> **Material.** This course · notebook: [01 taxonomy and process](notebooks/01_taxonomy_and_process.ipynb) ·
> [cheat-sheet](cheatsheet.md) · [exercises](exercises.md) · process metrics protocol
> [`configs/b4_process_metrics.yaml`](../../configs/b4_process_metrics.yaml) · results in
> [`results/`](results/) · reproduce with `make b4`.

> **The block in five lines**
>
> 1. The survey gives a **map**: five RAG paradigms, four agentic patterns, five workflow
>    patterns, and a taxonomy of architectures rated on eleven dimensions (Table 3).
> 2. On that map, our Block 3 agent is a **single-agent, centralized, tool-use** architecture with
>    bounded autonomy; the cascade is a **routing** workflow; the fixed pipeline is **modular RAG**.
> 3. Its lesson "agentic RAG is not always the right default" (10.1) is **confirmed** by Block 3;
>    "retrieval remains the primary bottleneck" (10.3) is **not** — once Block 2 fixed retrieval,
>    generation was.
> 4. Its call for **process-level evaluation** (10.5) pays off at once: 97 % of the agent's
>    `chunk_read` calls re-read a chunk it had already been given in full.
> 5. The survey is **qualitative**: its ratings are judgements, not measurements, and its example
>    answers contain illustrative numbers. Use it to name things, not to choose between them.

---

## 1. Why read a survey after running the experiments

A survey answers "what exists and what is it called?". Experiments answer "what works *here*?".
Reading the survey first would have given us vocabulary; reading it now also lets us **test it**.
Every lesson below is marked *confirmed*, *contradicted* or *not tested* by our own numbers.

How to read a survey efficiently:

- Start with the **taxonomy and the comparison table** (here Table 3, section 9): it is the map.
- Read the **lessons learned** (section 10): it is the authors' opinion, condensed.
- Skim the frameworks and applications: they date fast.
- For every claim, ask: *measured, cited, or asserted?*

## 2. Five RAG paradigms (section 2.3)

| paradigm | what it adds | example in this project |
|---|---|---|
| **Naïve RAG** | retrieve, then read | — |
| **Advanced RAG** | dense retrieval, neural reranking, multi-hop retrieval | dense + cross-encoder reranker (Block 2) |
| **Modular RAG** | hybrid sparse + dense, tools and APIs, interchangeable components | **our fixed pipeline**: BM25 + dense + RRF + reranker + contextual chunks |
| **Graph RAG** | a graph of entities and relations, for multi-hop relational reasoning | not used |
| **Agentic RAG** | autonomous agents that decide how to retrieve, iterate and refine | **our Block 3 agent** |

**Pitfall: "naïve" does not mean the same thing everywhere.** The survey defines naïve RAG as
*keyword* retrieval (TF-IDF, BM25) over static data. Most practitioners, and Block 1, call "naïve"
the *chunk → embed → top-k → generate* pipeline. In the survey's terms, that is already advanced
RAG. Always check which definition a source uses.

The paradigms are presented as an evolution, but they are **layers, not replacements**. Our
agent sits *on top of* a modular pipeline: it starts from its top-10 and calls BM25 and dense
search as tools.

## 3. What an agent is (section 3)

The survey's agent has four components:

| component | role | in our Block 3 agent |
|---|---|---|
| **LLM** with a role and a task | reasoning engine | `qwen2.5:7b-instruct`, prompt "gather evidence, reply DONE" |
| **Memory** (short / long term) | context across steps / across sessions | short-term only: the message history of one question |
| **Planning** (reflection, self-critique) | decomposes the task | none explicit: the model decides one call at a time |
| **Tools** | act beyond text | 4 retrieval tools (two searches, two reads) |

## 4. Four agentic patterns (sections 3.1–3.4)

| pattern | idea | predictability (survey) |
|---|---|---|
| **Reflection** | the agent critiques its own output and revises (Self-Refine, Reflexion, CRITIC) | mature |
| **Planning** | the agent decomposes a task into subtasks it chooses | less predictable |
| **Tool use** | the agent calls functions: search, APIs, code | mature |
| **Multi-agent** | specialized agents with their own memory collaborate | less predictable |

Our agent uses **tool use** only. Our grounding check (Block 3) is a *deterministic* critic: it
evaluates the answer but nothing loops back on its verdict, so it is not reflection.

## 5. Five workflow patterns (section 4)

The survey borrows these from Anthropic's *Building effective agents* (its reference [13]):

| pattern | shape | when |
|---|---|---|
| **Prompt chaining** | step 1 → step 2 → … | the task splits into fixed subtasks |
| **Routing** | classify, then send to a specialized path | inputs of different kinds need different handling |
| **Parallelization** | run in parallel: *sectioning* (subtasks) or *voting* (same task, several outputs) | speed, or confidence by agreement |
| **Orchestrator–workers** | a central model splits the task at run time and delegates | subtasks are not known in advance |
| **Evaluator–optimizer** | generate, evaluate, revise, loop | clear evaluation criteria exist |

**A distinction worth keeping** (made in Anthropic's text, blurred in the survey): a *workflow*
follows code paths written in advance; an *agent* decides its own path at run time. Our cascade is
a **workflow** (routing, the rule is code); our Block 3 loop is an **agent** (the model chooses
the tools and when to stop). Workflows are more predictable and cheaper to evaluate; start there.

## 6. The taxonomy of agentic RAG architectures (section 5)

| architecture | how it works | survey's example |
|---|---|---|
| **Single-agent (router)** | one agent picks among sources (SQL, vector search, web…) and synthesizes | customer support |
| **Multi-agent** | a coordinator dispatches to specialized retrieval agents, often in parallel | multi-domain research assistant |
| **Hierarchical** | agents in tiers: top agents prioritize and delegate, lower agents retrieve | financial analysis |
| **Corrective (CRAG)** | grade retrieved documents; if weak, rewrite the query or search elsewhere | academic research assistant |
| **Adaptive (Adaptive-RAG)** | a small classifier predicts query complexity: no retrieval, one step, or multi-step | customer support |
| **Graph-based** (Agent-G, GeAR) | agents combine graph and text retrieval, with a critic | healthcare, multi-hop QA |
| **Agentic document workflows (ADW)** | stateful, document-centric pipelines with business rules | invoice processing |

### 6.1 The eleven dimensions (Table 3, section 9.1)

The survey rates four architecture families on eleven dimensions. Reproduced from Table 3:

| dimension | single-agent | multi-agent | hierarchical / corrective | graph / document-centric |
|---|---|---|---|---|
| agent cardinality | single | multiple | multiple (tiered) | multiple |
| control structure | centralized | flat | hierarchical / evaluator-based | adaptive / structured |
| knowledge representation | unstructured | unstructured or hybrid | unstructured or hybrid | graph-based or document-centric |
| knowledge dependency | low | moderate | moderate | high, sensitive to graph quality |
| reasoning mode | implicit, LLM-driven | distributed across agents | strategic, supervisory control | relational or symbolic + LLM |
| reasoning depth | low to moderate | moderate to high | high | high, multi-hop |
| retrieval adaptivity | limited | high | high | high |
| scalability | moderate | high | moderate | moderate |
| latency | low | moderate | high | moderate to high |
| engineering complexity | low | moderate | high | high |
| best-fit scenarios | simple QA and routing | multi-domain synthesis | high-stakes, reliability-critical tasks | knowledge-intensive, relational domains |

Two remarks:

- **These are judgements, not measurements.** No experiment in the survey produces a "moderate"
  or a "high". Treat the table as a checklist of questions to ask, not as results.
- **The abstract announces four dimensions** (cardinality, control structure, *autonomy*,
  knowledge representation), but Table 3 has no autonomy row; "retrieval adaptivity" is the
  closest. For a design document, autonomy is worth stating explicitly: *who decides the next
  step, and what bounds it?*

## 7. Our systems on the map

| | fixed (Block 3) | agentic (Block 3) | cascade (Block 3) |
|---|---|---|---|
| paradigm | modular RAG | agentic RAG | workflow over both |
| workflow / agent | workflow (fixed chain) | agent | workflow (routing) |
| cardinality | — (no agent) | single | single, called conditionally |
| control structure | — | centralized | centralized, rule-based |
| autonomy | none | bounded: 6 tool calls, 8,000 tokens, stop = DONE | none in the router; bounded in the agent |
| knowledge representation | unstructured chunks | unstructured chunks, several granularities | unstructured |
| patterns | — | tool use | routing + tool use |
| closest survey box | — | single-agent (router) | adaptive RAG (but see below) |

Where the map does not fit:

- **Our agent routes between granularities, not between sources.** The survey's single-agent
  router picks SQL vs vector search vs web. Ours searches *one* corpus by keyword or meaning, then
  reads a chunk or a whole page: A-RAG's *hierarchical retrieval interfaces* (Block 1). The
  taxonomy has no box for that.
- **Our cascade is not Adaptive-RAG.** Adaptive-RAG routes *before* retrieval, on predicted
  query complexity. Our cascade routes *after* the fixed answer, on confidence signals. Same
  pattern (routing), different signal — and Block 3 showed that the signal is the whole game.

## 8. The survey's lessons, checked against our numbers (section 10)

| lesson | verdict here | evidence |
|---|---|---|
| **10.1** Agentic RAG is not always the right default | **confirmed** | fixed 0.28 vs agentic 0.18 accuracy (gap not significant), at 7.6× the cost and 2.7× the latency (ADR-0001) |
| **10.2** Architecture shapes behaviour | confirmed | same generator and same top-10: the loop changed refusals from 0.05 to 0.33 |
| **10.3** Retrieval quality remains the primary bottleneck | **contradicted here** | after Block 2, fixed had every gold chunk for 94 of 100 questions and still got 67 of them wrong; even with only the gold chunks the reader reaches 0.44 |
| **10.4** Autonomy needs explicit constraints | **confirmed, and not sufficient** | the budget bounded the cost, but 29 questions ended by exhausting it: bounding the loop does not teach the model when to stop |
| **10.5** Evaluate the process, not just the outcome | **confirmed** | see section 9: the traces explain the loss the outcome metrics only showed |
| **10.6** Domain knowledge amplifies agentic benefits | not tested | FinQA is financial, but our agent had no domain rules or structured knowledge |
| **10.7** Responsible deployment needs traceability | applied | every agent step is recorded (`trace`) and every number cites its chunk |

**On 10.3.** The lesson is right *in order*: retrieval first. We did exactly that in Block 2, and
it worked so well that the bottleneck moved. The general lesson is: **measure where the failures
are before choosing what to add.** An agent improves evidence gathering; it cannot fix a reader
that misreads complete evidence.

## 9. Process-level evaluation, applied (lesson 10.5)

The survey asks for "process-level metrics capturing reasoning efficiency [and] tool usage
patterns". We had the traces; `make b4` measures them (protocol frozen before computing,
exploratory because the traces had been read). On the 100 evaluation questions
(`results/process_metrics.csv`):

| | all | stop = DONE | stop = budget exhausted |
|---|---|---|---|
| questions | 100 | 71 | 29 |
| tool calls per question | 4.9 | 4.5 | 6.0 |
| searches (keyword + semantic) | 193 | 124 | 69 |
| … with at least one new candidate | 77 % | 77 % | 78 % |
| reads (chunk + page) | 296 | 191 | 105 |
| … adding nothing to the evidence | **81 %** | 77 % | 87 % |
| `chunk_read` of a chunk already given in full | **97 %** | 97 % | 97 % |
| chunks added per question | 1.1 | 1.2 | 0.7 |

What the process shows and the outcome hid:

- **Search works, reading does not follow.** Most searches surface new candidates, but the agent
  then reads chunks of its initial top-10 — whose full text was in its very first prompt. It
  verifies what it already has instead of fetching what it found.
- **The budget is spent on re-reading.** On the 29 questions that exhaust it, 87 % of reads add
  nothing. The stop rule then turns them into refusals.
- **This points at a fix the outcome metrics could not suggest:** a tool layer that refuses (or
  flags) reads of chunks already in context, or a prompt that lists what is already held. That
  is a design hypothesis for a new development cycle, not a result.

## 10. A critical reading: what the survey is, and is not

**What it is:** a well-organized map of a fragmented field, with a useful vocabulary, a sensible
set of lessons (10.1, 10.4 and 10.5 hold up against our data), and pointers to the original
systems (CRAG, Adaptive-RAG, Agent-G, GeAR).

**What it is not:**

- **Not an empirical comparison.** No architecture is run against another. Table 3's ratings,
  and claims such as "high accuracy" for agentic RAG (Table 1), are asserted, not measured.
- **Some claims contradict its own table and our data.** Section 2.5 says agentic RAG "reduce[s]
  latency through optimized workflows"; Table 3 rates its latency low to high depending on the
  architecture; Block 3 measured 2.7× the latency of fixed.
- **Example answers are illustrations.** The "integrated responses" of the use cases contain
  numbers ("a 20 % reduction in greenhouse gas emissions", "15 % growth over the past quarter")
  that are not results of any system. Never quote them.
- **Finance is thin.** The finance use case is auto-insurance claims (a LlamaCloud demo), and the
  benchmark list has no numerical table-reasoning dataset (FinQA, TAT-QA). Many references are
  blog posts and tutorials.

None of this makes it a bad survey: surveys of a young field are maps, not verdicts. It makes it
the wrong tool for choosing an architecture — that is what Block 3's protocol was for.

## 11. Open challenges (section 12), and where this project meets them

| challenge | the survey's point | in this project |
|---|---|---|
| coordination and emergent behaviour | multi-agent systems are hard to predict | single agent only |
| evaluation beyond output quality | process-aware benchmarks are missing | section 9: a first, small step |
| memory management | persistent memory can drift | not used |
| cost and efficiency | iterative retrieval costs latency and tokens | measured in Block 3: 6.3 LLM calls per question |
| safety, trust, governance | trace responsibility, human oversight | traces, citations, refusal over guessing; L2 adds AMF/GDPR |
| generalization | systems are tuned to one domain | one benchmark, one model: our conclusions are local |

## 12. Deviations from the plan

- The charter planned a "pedagogical synthesis, not a line-by-line audit". The synthesis is
  extended with a **check against our own numbers** and one small **process measurement**
  (`make b4`), because a reading block that ignores the project's evidence would teach less.
- The process metrics are **exploratory**: defined after the Block 3 traces had been read.

## 13. What to tell a client

- **Name the architecture.** "A single tool-using agent with a 6-call budget over a hybrid index"
  is a design; "we'll add agents" is not.
- **Workflow before agent.** Routing, chaining and evaluator loops written as code are cheaper,
  more predictable and easier to test than an autonomous loop.
- **Measure before you climb the taxonomy.** Each step up (multi-agent, hierarchical) adds latency
  and engineering complexity; demand evidence that the current level is the bottleneck.
- **Log the process.** Tool calls, what each one added, why the loop stopped. Without it you will
  see that the agent loses, never why.
