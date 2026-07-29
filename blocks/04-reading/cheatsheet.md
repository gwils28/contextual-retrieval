# Block 4 — Cheat-sheet

One box per notion. Details and sources: [course](README.md) · survey: Singh et al.,
arXiv:2501.09136 v4 (2026).

---

### Map vs territory
A survey names what exists (the **map**); experiments say what works here (the **territory**).
For each claim ask: *measured, cited, or asserted?*

### Five RAG paradigms
**Naïve** (retrieve-read; for the survey: keyword retrieval) → **Advanced** (dense, reranking,
multi-hop) → **Modular** (hybrid, tools, swappable parts) → **Graph** (entities + relations) →
**Agentic** (agents decide how to retrieve). Layers, not replacements.
Pitfall: "naïve RAG" = BM25 for the survey, = embed + top-k for most practitioners.

### An agent
**LLM** (role, task) + **memory** (short / long term) + **planning** (decompose, self-critique) +
**tools** (search, APIs, code). Our Block 3 agent: LLM + 4 retrieval tools + short-term memory.

### Four agentic patterns
**Reflection** (critique and revise) · **planning** (decompose) · **tool use** (call functions) ·
**multi-agent** (specialists collaborate). Reflection and tool use are mature; planning and
multi-agent are less predictable.

### Five workflow patterns (from Anthropic, *Building effective agents*)
**Prompt chaining** (fixed steps) · **routing** (classify, then dispatch) · **parallelization**
(sectioning or voting) · **orchestrator–workers** (split at run time) · **evaluator–optimizer**
(generate, grade, revise).

### Workflow vs agent
Workflow = code paths written in advance. Agent = the model chooses its path at run time.
Workflows are cheaper, more predictable, easier to test → **start there**.

### Taxonomy of architectures
| | in one line |
|---|---|
| single-agent (router) | one agent picks the source and synthesizes |
| multi-agent | coordinator + specialized retrieval agents, often parallel |
| hierarchical | tiers: top agents prioritize and delegate |
| corrective (CRAG) | grade retrieved docs; if weak, rewrite query / search elsewhere |
| adaptive (Adaptive-RAG) | classifier on query complexity: no / one / multi-step retrieval |
| graph-based (Agent-G, GeAR) | graph + text retrieval with a critic |
| ADW | stateful document workflows with business rules |

### Table 3's dimensions
cardinality · control structure · knowledge representation · knowledge dependency · reasoning
mode · reasoning depth · retrieval adaptivity · scalability · latency · engineering complexity ·
best fit. **Judgements, not measurements.** Missing row worth adding: **autonomy** (who decides
the next step, what bounds it).

### Our systems on the map
fixed = modular RAG (workflow) · agentic = single agent, centralized, tool use, bounded autonomy,
routes between *granularities* not sources (A-RAG) · cascade = routing workflow, signal *after*
the answer (≠ Adaptive-RAG, which routes *before* retrieval on query complexity).

### Lessons, checked
- 10.1 not the default → **confirmed** (fixed 0.28 vs agentic 0.18, 7.6× cheaper).
- 10.3 retrieval is the bottleneck → **contradicted here** after Block 2: 67 / 94 complete-context failures.
- 10.4 constrain autonomy → **confirmed, not sufficient**: a budget bounds cost, not the stop decision.
- 10.5 evaluate the process → **confirmed**: traces explain the loss.

### Process-level metrics
Measure *how* the agent spends its budget, from its traces: calls by tool, reads that add
nothing (81 %), re-reads of chunks already held (97 % of `chunk_read`), searches with no new
candidate (23 %), chunks added per question (1.1). They suggest fixes outcome metrics cannot.

### Reading a survey critically
No experiments → ratings are opinions. Example answers carry illustrative numbers — never quote
them. Check claims against its own tables (latency: "reduced" in 2.5, "moderate to high" in Table 3).

---

## Glossary (FR / EN)

| English | Français | Meaning |
|---|---|---|
| survey | état de l'art / revue | paper that organizes a field without new experiments |
| taxonomy | taxonomie | classification along named dimensions |
| modular RAG | RAG modulaire | pipeline of swappable components (hybrid, rerank…) |
| graph RAG | RAG sur graphe | retrieval over entities and relations |
| agentic pattern | patron agentique | reflection, planning, tool use, multi-agent |
| workflow pattern | patron de workflow | chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer |
| reflection | réflexion / autocritique | the model critiques and revises its own output |
| planning | planification | decomposing a task into chosen subtasks |
| orchestrator–workers | orchestrateur–exécutants | central model delegates subtasks decided at run time |
| evaluator–optimizer | évaluateur–optimiseur | generate, grade, revise in a loop |
| agent cardinality | cardinalité d'agents | number of agents |
| control structure | structure de contrôle | centralized, flat, hierarchical |
| autonomy | autonomie | who decides the next step, within which bounds |
| corrective RAG (CRAG) | RAG correctif | grades retrieval and corrects it |
| adaptive RAG | RAG adaptatif | picks a retrieval strategy from query complexity |
| process-level evaluation | évaluation du processus | metrics on the trajectory, not only the answer |
| trace | trace | recorded sequence of an agent's actions |
| hierarchical retrieval interfaces | interfaces de récupération hiérarchiques | tools at several granularities (A-RAG) |
