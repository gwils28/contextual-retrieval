# Block 2 — Measuring retrieval: dense vs hybrid vs reranking vs contextual, on FinQA

> **Goal of this block.** Put numbers on every retrieval layer from Block 1, on financial filings,
> with a protocol frozen before the first result, and confidence intervals on every number.
> **Retrieval only**: no answer is generated, no answer is judged.
>
> **Material.** This course · notebooks: [01 data & golden set](notebooks/01_data_and_golden_set.ipynb),
> [02 results](notebooks/02_results.ipynb), [03 error analysis](notebooks/03_error_analysis.ipynb) ·
> [cheat-sheet](cheatsheet.md) · [exercises](exercises.md) · protocol
> [`configs/b2_retrieval_protocol.yaml`](../../configs/b2_retrieval_protocol.yaml) · results in
> [`results/`](results/) · reproduce with `make b2`.

> **Results in five lines** (1,147 questions, `pass_all@10`, company name in the query):
>
> 1. **Every layer pays off, and the gains add up.** Under naive chunking: dense 0.674 → hybrid
>    0.726 → + reranker 0.840 → + contextual chunks **0.944**. All steps are significant after Holm.
> 2. **Contextual chunks are worth as much as the reranker** (+10.5 points on top of it) and cost
>    nothing at query time. They almost eliminate wrong-company hits (39 → 2).
> 3. **Table-aware chunking loses** (0.811 vs 0.944 with every layer), even at an equal token
>    budget. One chunk per row spreads a question's evidence over more chunks, and `pass_all`
>    needs all of them.
> 4. **The reranker is the expensive layer**: ~350–450 ms per query against ~10 ms for hybrid.
> 5. **Dense alone is the weakest option on financial filings**: BM25 beats it under naive chunking,
>    and rescues 148 of its misses.


---

## 1. Why a protocol before the first number

A benchmark has many free choices: which queries, which k, which metric, what counts as a hit,
which hyperparameters. If you make them *after* seeing results, you will unconsciously pick the
ones that tell the story you expected. The cure is boring and effective: **write every choice
down, freeze the file, then run.** Changes made later are listed as *amendments*, each with its
reason and its timing relative to the results.

Our protocol ([`b2_retrieval_protocol.yaml`](../../configs/b2_retrieval_protocol.yaml)) fixes:

- the corpus, the queries, and the sha256 of the raw files;
- the two chunkings and their exact gold mapping;
- the configurations, their hyperparameters (none tuned on test queries), and the reranker depth;
- the metrics, the k values, the statistical tests and the multiple-comparison correction.

Three amendments were made during implementation, all **before any metric was read**:

| | What changed | Why |
|---|---|---|
| A1 | a gold fact maps to a *set* of acceptable chunks | under table-aware chunking, a gold header row is repeated in every row chunk |
| A2 | also report `pass_all` at a fixed **token budget** | naive chunks are 2× longer and merge facts, which favours them at equal k |
| A3 | detect tables whose first row is data | found in the hand audit: a fake header polluted every row of 38 tables |

A3 also meant discarding a first run that had been stopped before any metric was read. That is
what "protocol first" costs in practice, and why it is worth it.

## 2. Turning FinQA into a retrieval benchmark

FinQA (Chen et al., EMNLP 2021) is a *question answering* dataset: 8k questions requiring numerical
reasoning over financial reports, each with annotated **supporting facts** (`gold_inds`) and a
**reasoning program** (`subtract(5829, 5735)`). It was not built for retrieval, and that shows in three ways.

### 2.1 Pages, not reports

Each record embeds **one page** of a 10-K: sentences before the table, the table, sentences after.
Our corpus is the 380 unique pages of the test split (100 companies), pooled in one index. It is
smaller than a real document store but has the property that matters: **near-duplicates**. 71 of
the 100 companies have pages from several years, with the same tables and the same wording.

### 2.2 Questions written for a page

Only **7.6 %** of questions name their company. *"What is the percentage change in rental expense
from 2015 to 2016?"* is a fine question when you are looking at the page, and a hopeless one in a
pooled index: dozens of pages answer it. Measuring whether a retriever finds *the* page would
measure the question's ambiguity.

The protocol restores what an analyst always knows, **which issuer**, by prefixing the company name:
`"Global Payments: what percentage of lease payments will be paid out in the first year?"`. Not
the year: the FinQA folder year is not reliably the report's fiscal year. Two sidebar settings
bound the effect:

- **raw question**: no prefix. It measures the ambiguity ceiling.
- **company filter**: restrict the search to the company's chunks (a metadata pre-filter). This is
  the production alternative to putting the entity in the query.

### 2.3 The golden chunk depends on the chunking

A gold fact is a sentence or a table row. Which **chunk** holds it depends on how the page was cut.
So the golden set is defined at the **fact level** and mapped to chunks per strategy (§3). That
way, both chunkings are evaluated against the same evidence.

## 3. Chunking: where the table problem lives

| | naive | table-aware |
|---|---|---|
| Rule | page linearized, fixed 200-token windows | sentences grouped up to 200 tokens, never across the table; one chunk per table row |
| Table row | `2015 net revenue \| $ 5829` (headers maybe in another chunk) | `2015 net revenue \| amount ( in millions ): $ 5829` |
| Chunks | 1,702 | 4,024 |
| Tokens / chunk | 178 mean | 80 mean, 43 median |
| Gold mapping | chunk holding the largest share of the fact | exact |

Measured on the gold evidence with naive chunking:

- **9.9 %** of gold facts are **cut across two chunks**;
- **27 %** of gold table rows land in a chunk **without their header row**. A cell then loses its
  unit ("in millions") and its column (which year).

That is the Armor Gestion failure ("the naive pilot hallucinated figures") reduced to two numbers.

### The equal-k trap

Naive chunks are twice as long and often hold several gold facts at once (1.22 gold sets per
question vs 1.59). At **equal k**, naive therefore *looks* better on `pass_all@k`: it sends twice
as many tokens and needs fewer hits. Comparing chunkings at equal k is comparing two different
prompt sizes. The fair comparison is at an equal **token budget**: take chunks in rank order until
1,000 (or 2,000) tokens are used, and check whether all gold facts made it. That is also the real
constraint: the LLM's context window, and the cost per token.

## 4. The configurations

| Config | Stage 1 | Stage 2 |
|---|---|---|
| `bm25` | BM25 (k1 = 1.5, b = 0.75), positive scores only | — |
| `dense` | cosine over `bge-base-en-v1.5`, exact search, BGE query prefix | — |
| `hybrid` | RRF (k = 60) of dense top-100 and BM25 top-100 | — |
| `hybrid_rerank` | hybrid top-50 | `bge-reranker-v2-m3` (fp16) scores the 50 (query, chunk) pairs |
| `contextual_*` | same retrievers on **contextualized chunks**: LLM-written context + chunk, for both indexes | |

Every configuration returns a ranking of 50 chunks, which is enough for the token-budget metric.

### Cross-encoder reranking, concretely

The bi-encoder compares two vectors computed separately. The cross-encoder reads
`[CLS] query [SEP] chunk [SEP]` in one pass and outputs one relevance score. Every query token
attends to every chunk token, so it can check that "2015" in the query matches the *2015 column*
in the row, not merely that both texts mention years. Nothing can be precomputed, so the cost is
**one forward pass per candidate per query**. That is why it only sees the top-50 of a cheap
first stage.

## 5. Metrics

A question has one or more gold facts (the operands of its computation). For a ranked list:

| Metric | Definition | Why |
|---|---|---|
| **`pass_all@k`** (primary) | 1 if **every** gold fact is in the top-k | a subtraction needs both numbers; one is not enough |
| `pass_any@k` | 1 if at least one gold fact is in the top-k | comparable to the cookbook's Pass@k |
| `recall@k` | fraction of gold facts in the top-k | partial credit |
| `mrr@10` | 1 / rank of the first gold chunk | does the evidence come *first*? |
| `pass_all@{1000,2000}tok` | `pass_all` on the longest prefix within the token budget | fair across chunkings (§3) |
| latency p50 / p95 | wall-clock per query, retrieval only, warm models, batch 1 | the price of each layer |

## 6. Statistics: how sure are we?

### 6.1 Confidence intervals by bootstrap

The 1,147 queries are a *sample* of the questions analysts could ask. Re-running on another sample
would give other numbers. The **percentile bootstrap** estimates that variability: resample the
queries with replacement 10,000 times, recompute the mean each time, and take the 2.5 % and 97.5 %
quantiles. With n = 1,147 and a rate around 60 %, the 95 % interval is about ±2.8 points.

### 6.2 Paired tests: compare systems on the same queries

Two configurations are evaluated on the **same** queries. Most queries are easy for both or hard
for both; they say nothing about which system is better. Only the **discordant** queries matter:

| | B succeeds | B fails |
|---|---|---|
| **A succeeds** | (no information) | `a_only` |
| **A fails** | `b_only` | (no information) |

Under "no difference", a discordant query is equally likely to favour A or B, so `b_only` follows a
Binomial(`a_only` + `b_only`, 0.5). That is the **exact McNemar test**. It is far more sensitive
than checking whether two confidence intervals overlap, because it removes the shared query
difficulty.

### 6.3 Multiple comparisons: Holm

We run 10 paired tests: 3 along the layer chain per chunking (dense → hybrid → hybrid_rerank →
contextual_hybrid_rerank), and naive vs table-aware for 4 configs. At α = 0.05, testing 10 true null
hypotheses would give at least one false positive with probability up to 1 − 0.95¹⁰ ≈ 40 %.
**Holm's correction** sorts the p-values and multiplies the smallest by 10, the next by 9, and so on (keeping the sequence monotone). It
controls the probability of *any* false positive at 5 %, and is uniformly more powerful than plain
Bonferroni.

## 7. Results

All numbers below are read from [`results/summary.csv`](results/summary.csv) and
[`results/paired_tests.csv`](results/paired_tests.csv) (`make b2-report`). Main setting: company
name prefixed to the question. Brackets: 95 % bootstrap interval. Notebook:
[02 results](notebooks/02_results.ipynb).

| Config | naive `pass_all@10` | table-aware `pass_all@10` | p50 latency (naive) |
|---|---|---|---|
| `bm25` | 0.735 [0.709, 0.760] | 0.480 [0.452, 0.509] | 4 ms |
| `dense` | 0.674 [0.646, 0.701] | 0.516 [0.486, 0.545] | 6 ms |
| `hybrid` | 0.726 [0.700, 0.752] | 0.542 [0.514, 0.571] | 9 ms |
| `hybrid_rerank` | 0.840 [0.818, 0.860] | 0.605 [0.576, 0.633] | 355 ms |
| `contextual_bm25` | 0.874 [0.854, 0.893] | 0.734 [0.708, 0.759] | 4 ms |
| `contextual_dense` | 0.853 [0.831, 0.873] | 0.783 [0.759, 0.806] | 6 ms |
| `contextual_hybrid` | 0.892 [0.874, 0.909] | 0.790 [0.765, 0.813] | 9 ms |
| `contextual_hybrid_rerank` | **0.944** [0.930, 0.957] | **0.811** [0.788, 0.833] | 446 ms |

### 7.1 What each layer buys (paired tests)

| Comparison | Δ `pass_all@10` | `a_only` | `b_only` | Holm p |
|---|---|---|---|---|
| naive: dense → hybrid | +5.2 pts | 36 | 96 | < 10⁻⁴ |
| naive: hybrid → hybrid_rerank | +11.3 pts | 11 | 141 | < 10⁻⁴ |
| naive: hybrid_rerank → contextual_hybrid_rerank | +10.5 pts | 11 | 131 | < 10⁻⁴ |
| table-aware: dense → hybrid | +2.6 pts | 34 | 64 | 0.003 |
| table-aware: hybrid → hybrid_rerank | +6.3 pts | 28 | 100 | < 10⁻⁴ |
| table-aware: hybrid_rerank → contextual_hybrid_rerank | +20.6 pts | 23 | 259 | < 10⁻⁴ |

Read the discordant columns: going from hybrid to hybrid + reranker under naive chunking, 141
questions are newly solved and 11 are lost. A layer is never free of regressions, but here the
balance is lopsided.

Three lessons:

- **Hybrid helps less than the reranker.** RRF can only reorder what BM25 and dense already found.
  The reranker reads query and chunk together and fixes the order *inside* the candidate list.
- **Contextual chunks help every retriever, dense most of all**: +17.9 points for dense under naive,
  +26.7 under table-aware. A context sentence ("This chunk provides receivables from the money pool
  for Entergy Gulf States Louisiana as of December 31 for 2011–2008") gives the embedding the
  company, the year and the table it was missing.
- **Only the chain steps above were tested.** For example, `contextual_hybrid` →
  `contextual_hybrid_rerank` under table-aware (+2.1 points) was not in the protocol, so we do not
  claim it as a gain.

## 8. Chunking: the table-aware surprise

We expected table-aware chunking to win: it never separates a row from its header. It loses, with
every configuration, and the paired tests are all significant (−13 to −23 points).

### 8.1 Not an artefact of equal k

§3 warned that equal k favours naive (longer chunks, more tokens sent). At an equal **token budget**
the gap shrinks, but stays:

| `pass_all` @ 2,000 tokens | naive | table-aware |
|---|---|---|
| `hybrid_rerank` | 0.840 | 0.657 |
| `contextual_hybrid_rerank` | 0.926 | 0.848 |

### 8.2 Granularity: the evidence is spread out

Breakdown by number of gold facts per question, `contextual_hybrid_rerank`
([`results/pass_by_n_gold.csv`](results/pass_by_n_gold.csv)):

| gold chunks per question | naive: n | naive `pass_all@10` | table-aware: n | table-aware `pass_all@10` |
|---|---|---|---|---|
| 1 | 912 | 0.981 | 607 | 0.952 |
| 2 | 224 | 0.808 | 427 | 0.719 |
| 3+ | 11 | 0.636 | 113 | 0.398 |

Two effects add up. Table-aware turns many single-chunk questions into multi-chunk ones (540
questions with 2+ gold chunks, against 235), and every extra gold chunk is one more chance to miss.
`pass_any@10` stays high for table-aware (0.958): it reaches the right page, then misses one of
the rows. Before contextualization, an isolated row was also hard to find at all (0.865 even with a
single gold fact); context closes most of that part of the gap.

**Practical consequence.** "Table-aware" is not a single design choice. One chunk *per row* is too
fine for questions that combine rows. Alternatives to test (not done here): one chunk per table,
or per group of rows with the header, or retrieving a row and expanding to its whole table
(small-to-big retrieval).

## 9. Query form: entity in the query, or entity as a filter

`pass_all@10` by setting:

| | raw question | company in query (main) | company filter |
|---|---|---|---|
| naive `hybrid_rerank` | 0.813 | 0.840 | 0.935 |
| naive `contextual_hybrid_rerank` | 0.873 | 0.944 | 0.970 |
| table-aware `hybrid_rerank` | 0.616 | 0.605 | 0.761 |
| table-aware `contextual_hybrid_rerank` | 0.717 | 0.811 | 0.851 |

- **Filtering on metadata is the strongest single lever** when the entity is known: +9.5 points
  for plain naive retrieval, at no cost. In production, extract the company (and ideally the
  fiscal year) from the question and filter on it.
- **Contextual chunks + company in the query (0.944) beats the filter without context (0.935).**
  Context puts the company name inside every chunk, so the query can match it.
- Context and filter overlap: with the filter, contextualization adds only 3.5 points under naive.
  Much of its value here is disambiguation between issuers and years.

## 10. Error analysis

Notebook: [03 error analysis](notebooks/03_error_analysis.ipynb). Files written by
`make b2-analysis`.

### 10.1 Five ways to miss

For each miss at k = 10, we look at what went wrong. This taxonomy is **exploratory**: it was
defined after the results were seen, so it describes failures but never ranks configurations.

| Mode (counts out of 1,147) | naive `hybrid_rerank` | naive + context | table-aware `hybrid_rerank` | table-aware + context |
|---|---|---|---|---|
| hit | 963 | 1,083 | 694 | 930 |
| partial (some gold facts, not all) | 94 | 46 | 253 | 169 |
| top-1 from **another company** | 39 | 2 | 77 | 3 |
| top-1 same company, **other page** (year) | 25 | 9 | 47 | 18 |
| top-1 right page, wrong chunk | 26 | 7 | 76 | 27 |

Contextual chunks remove almost all wrong-company and most wrong-year misses. What remains is
`partial`: the retriever found the right place but not every number the question needs.

### 10.2 Dense misses that BM25 rescues

[`results/bm25_rescues.csv`](results/bm25_rescues.csv) lists the questions where BM25 finds every
gold fact in the top 10 and dense does not: **148** under naive chunking (main setting), 66 under
table-aware. A typical case, Adobe:

> *"The combined amount of accrued interest and penalties related to tax positions … was what
> percent of the total ending balance as of November 28 2008?"*

The gold chunk is the FIN 48 table of unrecognized tax benefits. BM25 ranks it 2nd: the question
shares rare tokens with it ("tax positions", "ending balance", "2008"). Dense ranks first a
restructuring-charges table from the same notes: same company, same register of language, wrong
table. Dense captures the *topic*; it blurs the exact line items. Financial questions are full of
them, which is why BM25 alone beats dense under naive chunking (0.735 vs 0.674).

## 11. Costs and limits

**Costs** (this machine: RTX 5070 Laptop, 8 GB):

- **Contextualization**: 5,726 chunks with `qwen2.5:7b-instruct` on the local GPU, about 0.9–1.1
  chunk/s, so about 1.5 GPU-hours. It runs **once per chunk at ingestion**, adds nothing to query
  latency, and is cached. The generated contexts are versioned in
  [`data/contexts/`](../../data/contexts/): `make b2` reuses them instead of re-running the LLM,
  whose output is not bit-reproducible across hardware and versions.
- **Reranker**: one cross-encoder pass over 50 candidates per query, 355 ms median (naive). Context
  makes chunks longer, so the reranker gets slower: 446 ms.
- Dense, BM25 and RRF stay under 15 ms.

**Limits** (read the numbers with these in mind):

- **Small corpus.** 380 pages pooled in one index. A real document store has far more distractors;
  absolute scores would be lower, and the value of filters and context likely higher.
- **Local contextualizer.** Contexts come from a 7B open model. The protocol plans a reference run
  with a Claude Haiku-class model (prompt caching on); it is not done yet.
- **Retrieval only.** A retrieved gold chunk does not guarantee a correct answer. Generation is
  measured in a later block.
- **One table-aware design.** "One chunk per row" lost; other table-aware designs were not tested.
- **Fixed hyperparameters.** Nothing was tuned on test queries (window, k of RRF, reranker depth),
  by design. A tuned system could do better, and would need a separate dev set to stay honest.

## 12. What to tell a client

1. **Do not ship dense-only retrieval on financial documents.** Add BM25 (hybrid): it is nearly
   free and catches exact line items.
2. **Put the entity in the search**: a metadata filter on the company (and year) if the question
   names it, contextual chunks otherwise. Most misses on multi-company corpora are wrong-company
   or wrong-year hits.
3. **Contextualize at ingestion** if the budget allows a one-off LLM pass over the corpus: +10 to
   +20 points here, zero latency cost.
4. **Add a reranker** when ~400 ms per query is acceptable; it is the largest single gain without
   context.
5. **Test chunking on your own questions** before believing a "table-aware" label: the finest
   chunking is not the best one when questions combine several figures.

