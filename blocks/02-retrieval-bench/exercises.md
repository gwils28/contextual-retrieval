# Block 2 — Exercises

Try each question before opening the answers at the bottom. E1–E3 are **pen and paper**; E4–E5
use `ragbench` and the files in [`results/`](results/).

## Control questions

**Q1.** Why is the protocol frozen *before* the first result? Give one concrete choice in this block
that could have been tuned to the outcome otherwise.

**Q2.** FinQA was built for question answering. Name two properties that make it awkward as a
retrieval benchmark, and how the protocol handles each.

**Q3.** Why is the golden set defined at the *fact* level and not at the chunk level?

**Q4.** A question needs two numbers. Why is `pass_all@k` the primary metric rather than
`pass_any@k`?

**Q5.** Naive chunking looks better than table-aware at equal k. Why is that comparison unfair,
and what does the protocol use instead?

**Q6.** Two configs have 95 % intervals [0.818, 0.860] and [0.846, 0.884]. They overlap. Can the
difference still be significant? Why?

**Q7.** In a McNemar test, why do the queries that both systems solve carry no information?

**Q8.** Contextual chunks and the reranker give similar gains. Why are their costs so different?

**Q9.** Why does contextualization help *dense* more than BM25?

**Q10.** The company filter gives +9.5 points for free. When can you *not* use it in production?

**Q11.** Table-aware `pass_any@10` is 0.958 but `pass_all@10` is 0.811. What does that gap tell
you, and what would you try next?

**Q12.** Why is the failure-mode taxonomy labelled "exploratory", and what must you *not* do with it?

## Exercises

### E1 — McNemar by hand

Going from `hybrid` to `hybrid_rerank` on 20 queries, you observe `a_only` = 1 and `b_only` = 7.

1. Under H0, what is the distribution of `b_only`?
2. Compute the two-sided exact p-value.
3. Is the gain significant at α = 0.05?

### E2 — Holm by hand

Four paired tests give p = 0.001, 0.030, 0.012, 0.040. Apply Holm at α = 0.05. Which are
significant? Compare with Bonferroni.

### E3 — Token budget

A ranking returns chunks of lengths 180, 175, 190, 160, 170 tokens; the gold facts are in chunks 2
and 5 (1-indexed). Budget: 700 tokens.

1. Which chunks fit in the budget?
2. What is `pass_all@700tok`? `pass_all@5`?
3. Same question with table-aware chunks of lengths 40, 45, 50, … (+5 each), gold in chunks 3 and
   12. Which chunking wins at equal budget? At equal k = 5?

### E4 — Read the results

Using [`results/summary.csv`](results/summary.csv):

1. Which config has the best `pass_all@10` per latency? Is it the one you would ship?
2. Find the `company_filter` setting for `hybrid_rerank` (naive). Compare with
   `contextual_hybrid_rerank` in the main setting. What do you conclude?

### E5 — Inspect a rescue

Load [`results/bm25_rescues.csv`](results/bm25_rescues.csv) and pick a naive / main question. Using
`ragbench.b2.load_bench()` and `per_query_plain.parquet`, print the gold chunk and dense's top-1.
Which tokens of the question appear in the gold chunk but not in dense's top-1?

---

## Answers

**A1.** Otherwise metric choices drift toward the story you expected. Examples: the k values, the
choice of `pass_all` vs `pass_any`, the window size, the reranker depth, which comparisons are
tested. Amendments A1–A3 were all made before any metric was read, and recorded.

**A2.** (a) Pages, not reports: we pool the 380 test pages, which also creates near-duplicates
(same company, several years). (b) Questions rarely name the company (7.6 %): the main setting
prefixes the company name; `raw_question` and `company_filter` bound the effect. (Also: no
retrieval labels at chunk level → map `gold_inds` to chunks.)

**A3.** Which chunk holds a fact depends on the chunking. Defining gold at the fact level and
mapping it per strategy evaluates both chunkings against the same evidence, and survives any
re-chunking.

**A4.** With only one of the two numbers, the computation fails. `pass_any` would count that as a
success and overstate retrieval quality.

**A5.** Naive chunks are ~2× longer and often contain several gold facts, so at equal k they send
more tokens and need fewer hits. The protocol adds `pass_all` at a fixed **token budget** (1,000
and 2,000 tokens), which is also the real constraint (context window, cost).

**A6.** Yes. Overlapping intervals do not imply "no difference": the intervals ignore that both
systems are run on the *same* queries. A paired test removes the shared query difficulty and is
much more sensitive.

**A7.** They are solved by both (or failed by both) whichever system is better: they cannot tell
the systems apart. Only discordant queries carry evidence about the difference.

**A8.** Context is computed **once per chunk at ingestion** (≈ 1.5 GPU-hours here, then cached);
query time is unchanged. The reranker runs a cross-encoder on 50 (query, chunk) pairs **for every
query** (≈ 350–450 ms).

**A9.** A table row like `2015 | revenue: 1,234` has no company, table or topic words: its embedding
is close to thousands of similar rows. Context adds exactly those words. BM25 already matched the
exact tokens it had; context helps it too (+13.9 points under naive), but dense gains more (+17.9).

**A10.** When the question does not name the entity ("which of our clients…", comparisons across
companies), when the entity extraction is wrong, or when metadata is missing or unreliable (here,
the FinQA folder year is not the fiscal year).

**A11.** The right page is almost always reached, but not every gold row makes the top 10. The
problem is granularity, not finding the place. Next: coarser table chunks (one per table, row
groups with header) or small-to-big retrieval (retrieve a row, return its whole table).

**A12.** It was defined after the results were seen. It is fine to *describe* failures with it,
not to rank configurations or claim a gain from it. A ranking must use a pre-registered metric.

**E1.**
1. `b_only` ~ Binomial(8, 0.5).
2. P(X ≥ 7) = (C(8,7) + C(8,8)) / 2⁸ = 9 / 256 ≈ 0.035; two-sided p = 2 × 0.035 ≈ 0.070.
3. No: p ≈ 0.07 > 0.05. Seven wins to one look convincing, but 8 discordant queries are too few.
   Check: `from ragbench.evaluation import mcnemar_exact`.

**E2.** Sorted: 0.001, 0.012, 0.030, 0.040. Holm: 0.001 × 4 = 0.004; 0.012 × 3 = 0.036;
0.030 × 2 = 0.060; 0.040 × 1 = 0.040 → monotone: 0.060. Significant: the first two (0.001,
0.012). Bonferroni (× 4): 0.004, 0.048, 0.120, 0.160 → same two here, but Holm's adjusted values are
never larger. Check: `from ragbench.evaluation import holm`.

**E3.**
1. 180 + 175 + 190 = 545; + 160 = 705 > 700 → chunks 1–3 fit.
2. `pass_all@700tok` = 0 (chunk 5 is out). `pass_all@5` = 1.
3. Lengths 40, 45, …, 95 for chunks 1–12 sum to 810 > 700; chunks 1–11 sum to 715 > 700; chunks
   1–10 sum to 625 → chunks 1–10 fit, chunk 12 does not: `pass_all@700tok` = 0. At k = 5 it is 0
   too. Both fail at 700 tokens; at k = 5, naive "wins" only because each of its chunks is 3–4×
   longer. That is the equal-k trap.

**E4.**
1. `contextual_hybrid` (0.892 at 9 ms) under naive. Whether to ship it or add the reranker
   (+5.2 points, ~440 ms) depends on the latency budget; note that this specific gain was not
   among the pre-registered tests.
2. `company_filter` / `hybrid_rerank` / naive: 0.935. Main / `contextual_hybrid_rerank`: 0.944.
   Context in the chunks reaches the filter's level without needing to extract the entity; with
   both, 0.970.

**E5.** Open-ended. On the Adobe example of the course, the gold chunk contains "tax positions",
"ending balance", "2008" and "fin 48"; dense's top-1 (a restructuring table) does not.
