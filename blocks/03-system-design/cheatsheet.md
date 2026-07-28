# Block 3 — Cheat-sheet

One box per notion. Details, numbers and sources: [course](README.md) ·
decision: [ADR-0001](adr-0001-fixed-vs-agentic.md).

---

### Two stages, two scores
- **Retrieval** is judged on the chunks handed to the generator (`pass_all_ctx`), never on the answer.
- **Generation** is judged on the answer (`exec_accuracy`, `wrong_answer_rate`, `refusal_rate`).
- Join them **after** scoring in a 2×2 **diagnosis table**: hit & wrong = generation failure,
  miss & wrong = retrieval failure. Here fixed: 67 generation failures vs 5 retrieval failures.

### Three architectures
| | how | LLM calls |
|---|---|---|
| **fixed** (retrieve-then-read) | top-10 → one generator call | 1 |
| **agentic** | top-10 → agent calls tools (search, read) → DONE → generator | many (6.3 here) |
| **cascade** | fixed; escalate to agentic if a cheap signal fires | 1, or 1 + agent |
| **oracle** (not deployable) | gold chunks → generator: the generator's ceiling | 1 |

### Agent
An LLM in a loop that calls **tools** and decides when to stop. Needs a **budget** (calls,
tokens) and a **stop rule**. Here: 6 tool calls; exhausted budget → `insufficient_context`
(never a forced guess). Started from fixed's top-10, so it could only add evidence.
**Failure mode of a weak model: it does not know when it knows enough** (re-reads, runs out of budget).

### Cascade
Cheap path first, expensive path only when a signal fires. **A cascade is only as good as its
signal**: the signal must separate right from wrong answers. Check that on development data
before routing on it.

### Program-of-thought generation
The LLM writes a **program** (FinQA DSL: `subtract(5829, 5735), divide(#0, 5735)`); code executes
it. The LLM never does arithmetic. `#n` = result of step n.

### Structured output
A **JSON schema** constrains decoding: an enum of operations, exactly two arguments per step,
and a chunk id for each argument. Turned free-text chaos into valid programs (oracle 0.26 → 0.50
over four prompt versions).

### Grounding check
Every number of the program appears in the chunk it cites. Cheap, needs no gold answer.
**Not a confidence signal here**: right on 0.31 of grounded answers, 0.28 of ungrounded ones.
Pitfall: a program with no number (`divide(#0, #1)`) is vacuously grounded.

### Numeric match (frozen rule)
`|pred − gold| ≤ 0.005 · max(|gold|, 1)`; 25 and 0.25 both accepted.
Pitfall: below 1 the tolerance is an **absolute** 0.005 → loose on ratios. Prefer a relative tolerance.

### Refusal ≠ wrong answer
Both score 0 on accuracy. A refusal goes to a human; a wrong number goes into a report.
Report `wrong_answer_rate` separately. A low wrong-answer rate bought by refusing a third of the
questions (the agent) is not a virtue.

### Dev / evaluation split
Design prompts, tools, thresholds on **development** questions; run **evaluation** once, change
nothing after. **Stopping rule**: stop iterating when the gain is within noise (v3 → v4: +2 points
on 100 questions).

### Small samples (n = 100)
95 % CI ≈ ±9 points. Paired exact **McNemar** (discordant questions only) + **Holm** over all
tests. Fixed vs agentic: 17 vs 7 discordant, p = 0.064 → Holm 0.38. **Not significant = "not
shown", never "no difference".** Strata under 20 questions: counts only.

### Burden of proof
When no gap is shown, choose the cheaper option: the expensive one must **earn** its cost.
Agent: 7.6× the USD, 2.7× the latency, no gain shown → not adopted.

### Cost per point
Layer gain ÷ layer cost. Hybrid 0.6 ms/point, reranker 30 ms/point, contextual chunks 8.7 ms/point
(+ one-time ingestion), agent: no gain. USD equivalent = tokens × API price of the same model.

### ADR (Architecture Decision Record)
Context · decision · evidence · alternatives · consequences · limits · **revisit when**.
Cites only numbers from `make` targets.

---

## Glossary (FR / EN)

| English | Français | Meaning |
|---|---|---|
| retrieve-then-read | récupérer puis lire | one retrieval, one LLM call (fixed RAG) |
| agentic RAG | RAG agentique | the LLM drives its own retrieval with tools |
| tool call | appel d'outil | the LLM asks the runtime to run a function |
| agent loop | boucle d'agent | LLM → tools → LLM… until a stop condition |
| budget | budget | max tool calls / tokens per question |
| stop rule | règle d'arrêt | what ends the loop, and what the output is then |
| cascade | cascade | cheap system first, expensive one on a signal |
| escalation signal | signal d'escalade | cheap test that sends a question to the expensive path |
| oracle context | contexte oracle | exactly the gold chunks: generator ceiling |
| generator / reader | générateur / lecteur | the LLM that writes the answer from chunks |
| program of thought | programme de raisonnement | the LLM writes code, code computes |
| DSL | langage dédié | small purpose-built language (FinQA programs) |
| structured output | sortie structurée | output constrained by a JSON schema |
| grounding / grounded | ancrage / ancré | each number found in the chunk it cites |
| refusal | refus / abstention | `insufficient_context` |
| wrong answer | réponse fausse | answered, valid program, wrong value |
| diagnosis table | table de diagnostic | retrieval hit/miss × answer right/wrong |
| small-to-big | du petit au grand | search small chunks, read the whole page |
| burden of proof | charge de la preuve | the costlier option must show a gain |
| ADR | ADR (dossier de décision d'architecture) | written record of an architecture decision |
