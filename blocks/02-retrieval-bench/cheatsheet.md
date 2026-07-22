# Block 2 — Cheat-sheet

One box per notion. Details, numbers and sources: [course](README.md).

---

### Protocol first
Write every choice down (data, chunking, configs, metrics, tests), **freeze the file, then run**.
A later change is an *amendment*: what, why, and whether results had been seen.
Defining things after seeing results = tuning the story to the outcome.

### FinQA as a retrieval benchmark
- QA dataset: one 10-K **page** per record + supporting facts (`gold_inds`) + a reasoning program.
- Pool the 380 test pages in one index → **near-duplicates** (same company, other years).
- Questions rarely name the company → add it: `"{company}: {question}"`.
- Golden set at the **fact** level, mapped to chunks **per chunking** → both chunkings face the same evidence.

### Metrics with several gold facts
| | 1 if… |
|---|---|
| `pass_all@k` | **every** gold fact in top-k (a subtraction needs both numbers) |
| `pass_any@k` | at least one gold fact in top-k |
| `recall@k` | share of gold facts in top-k (partial credit) |
| `pass_all@Ntok` | `pass_all` on the ranked prefix that fits in N tokens |

**Equal k is unfair across chunkings**: longer chunks send more tokens. Compare at an equal **token budget**.

### Bootstrap confidence interval
Resample the queries with replacement 10,000×, recompute the mean, take the 2.5 % / 97.5 % quantiles.
n = 1,147, rate ≈ 60 % → about ±2.8 points.

### Exact McNemar (paired)
Same queries, two systems. Only **discordant** queries count:
`a_only` (A wins), `b_only` (B wins). Under H0, `b_only` ~ Binomial(`a_only` + `b_only`, 0.5).
Much more sensitive than "do the intervals overlap?".

### Holm correction
m tests: sort p-values, multiply the i-th smallest by (m − i + 1), keep the sequence monotone.
Controls P(any false positive) ≤ α. Always at least as powerful as Bonferroni.
10 tests at α = 0.05 without correction → up to 40 % chance of a false positive.

### Results on FinQA (`pass_all@10`, company in query)
| | naive | table-aware |
|---|---|---|
| dense | 0.674 | 0.516 |
| hybrid | 0.726 | 0.542 |
| + rerank | 0.840 | 0.605 |
| + contextual chunks | **0.944** | **0.811** |

- Every layer adds; **context ≈ reranker** in gain, and costs **0 ms** at query time.
- BM25 > dense on filings (exact line items, years).
- Reranker ≈ 350–450 ms/query (50 candidates); everything else < 15 ms.

### Five ways to miss (exploratory)
`hit` · `partial` (some gold facts) · top-1 **other company** · top-1 same company **other page/year** · top-1 right page **wrong chunk**.
Context nearly removes wrong-company misses (39 → 2). What remains is mostly `partial`.

### Chunking lesson
One chunk **per table row** loses (−13 to −23 pts): the evidence of one question is spread over
more chunks, and `pass_all` needs all of them. Finest ≠ best.
Try instead: one chunk per table, row groups with header, or **small-to-big** (retrieve a row, return its table).

### Entity: query vs filter
- Metadata **filter** on the company: +9.5 pts, free. Extract the entity, filter on it.
- Contextual chunks put the entity *inside* every chunk → the query can match it without a filter.
- The two overlap: most of context's gain here is disambiguation (issuer, year).

### Ops note (8 GB GPU)
Ollama keeps the LLM resident after contextualization → the embedder runs out of memory.
Unload first: `ollama stop <model>` or `keep_alive: 0` on the API.

---

## Glossary (EN → FR)

| English | Français | Definition |
|---|---|---|
| protocol (pre-registration) | protocole (pré-enregistrement) | every evaluation choice written and frozen before the first result |
| amendment | amendement | recorded change to a frozen protocol, with its reason and timing |
| golden set | jeu de référence | questions with their known evidence |
| gold fact | fait de référence | sentence or table row annotated as evidence (`gold_inds`) |
| near-duplicate | quasi-doublon | page almost identical to another (same company, other year) |
| granularity | granularité | size of the retrieval unit (chunk) |
| table-aware chunking | découpage conscient des tableaux | chunking that never splits a row from its header |
| token budget | budget de tokens | maximum number of tokens of retrieved context sent to the LLM |
| `pass_all@k` | — | 1 if every gold fact is in the top-k |
| confidence interval (CI) | intervalle de confiance (IC) | range that contains the true value with a stated probability |
| bootstrap | bootstrap (rééchantillonnage) | estimating variability by resampling the queries with replacement |
| paired test | test apparié | compares two systems on the same queries |
| discordant pair | paire discordante | query solved by one system and not the other |
| McNemar test | test de McNemar | exact binomial test on the discordant pairs |
| multiple comparisons | comparaisons multiples | running many tests inflates the chance of a false positive |
| Holm correction | correction de Holm | step-down adjustment of p-values controlling the family-wise error |
| family-wise error rate (FWER) | taux d'erreur par famille | probability of at least one false positive among all tests |
| reranker | reclasseur | second-stage model that re-scores the first-stage candidates |
| contextual chunk | fragment contextualisé | chunk prefixed with an LLM-written sentence situating it in its document |
| ingestion | ingestion / indexation | offline step that chunks, enriches and indexes documents |
| metadata filter | filtre sur métadonnées | restricting the search to chunks with a given attribute (company, year) |
| failure mode | mode d'échec | category of reason why a query missed |
| exploratory analysis | analyse exploratoire | analysis defined after seeing results: describes, never ranks |
| small-to-big retrieval | recherche small-to-big | retrieve a small unit (row), return its larger parent (table) |
| latency p50 / p95 | latence p50 / p95 | median / 95th-percentile time per query |
