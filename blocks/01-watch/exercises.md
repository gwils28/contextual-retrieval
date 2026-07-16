# Block 1 — Exercises

Try each question before opening the answers at the bottom. Exercises E1–E3 are meant to be done
**with pen and paper**; you can then check yourself with `ragbench` (snippets in the answers).

## Control questions

**Q1.** A client's RAG assistant gives a wrong figure. Name the two possible causes, and for each,
one thing you would change.

**Q2.** Why does this project evaluate retrieval *without* looking at the generated answer?

**Q3.** With one golden chunk per question, how do Pass@k and recall@k relate? When do they differ?

**Q4.** Anthropic reports a 49 % reduction in retrieval failures. Starting from a 5.7 % failure
rate, what is the new failure rate? How many *absolute* points is that?

**Q5.** Why are dense retrievers fast at query time, and what does that same design prevent them
from doing well?

**Q6.** In BM25, what do $k_1$ and $b$ control? What happens with $b = 0$?

**Q7.** The query is `share buyback`; the right document says "repurchased 2.1 million shares".
Which retriever finds it, which one does not, and why?

**Q8.** What exactly is prepended to a chunk in Contextual Retrieval, and to which indexes is it
applied?

**Q9.** Why is Contextual Retrieval affordable only with prompt caching?

**Q10.** Why does RRF use ranks rather than scores?

**Q11.** Why rerank only the top-N candidates and not the whole corpus? What is the risk if N is
too small?

**Q12.** In A-RAG, what is "progressive disclosure" and what does the paper show happens without it?

## Exercises

### E1 — BM25 by hand

Corpus (already tokenized, $N = 4$, $k_1 = 1.5$, $b = 0.75$):

| doc | tokens | length |
|---|---|---|
| d0 | goodwill impairment recorded | 3 |
| d1 | revenue grew revenue grew strongly this year | 7 |
| d2 | goodwill unchanged | 2 |
| d3 | tax rate fell | 3 |

1. Compute avgdl.
2. Compute IDF(`goodwill`), IDF(`impairment`), IDF(`revenue`).
3. Compute BM25(`goodwill impairment`, d0).
4. Compute BM25(`revenue`, d1). Note that `revenue` appears twice.
5. What is surprising about `goodwill`'s contribution, and why?

### E2 — RRF by hand

Dense ranking: `A, B, C, D`. BM25 ranking: `C, A, E, B`.

1. Compute the fused RRF ranking with $k = 60$.
2. Same with $k = 0$. Does the order change?
3. `E` appears in only one list, at rank 3; `D` in only one list, at rank 4. Explain their order.

### E3 — How many queries do you need?

A configuration reaches Pass@5 = 80 % on a golden set of $n$ queries.

1. Using the normal approximation, give the half-width of the 95 % confidence interval for
   $n = 25$ and $n = 300$.
2. A new configuration scores 83 %. With which $n$ can you claim it is better from these intervals
   alone? What better test does the charter plan, and why is it more powerful?

---

## Answers

<details>
<summary>Q1–Q12</summary>

**Q1.** *Retrieval failure*: the chunk with the right figure was not retrieved. Fix chunking (keep
tables intact), the representation (contextual chunks), or the ranking (hybrid, reranker).
*Generation failure*: the chunk was there, but the model misread it or made up a figure. Fix the
prompt, the model, or add a post-generation grounding check that rejects unsourced numbers.

**Q2.** Because a wrong answer mixes two causes. Measuring retrieval alone (was the golden chunk
in the top-k?) tells you which stage to fix. Judging retrieval by answers leads to "improving the
prompt" while the right chunk never reached the model.

**Q3.** They are equal. They differ when a question has several relevant chunks: Pass@k counts a
success as soon as *one* is found, recall@k measures the *fraction* found.

**Q4.** 5.7 × (1 − 0.49) ≈ 2.9 %. That is −2.8 absolute points.

**Q5.** Document vectors are computed once at ingestion, so a query costs one embedding plus a
matrix–vector product. But query and document never interact: the whole passage must be squeezed
into one vector, so fine distinctions (rare codes, exact numbers) get blurred.

**Q6.** $k_1$ controls term-frequency saturation (how fast repeated occurrences stop adding score).
$b$ controls length normalization. With $b = 0$, document length is ignored: a long and a short
document with the same tf get the same score.

**Q7.** Dense finds it (it matches meaning: buyback ≈ repurchase). BM25 does not: "buyback" never
appears, and without stemming "share" ≠ "shares", so every document scores 0.

**Q8.** A short LLM-written context (≈ 50–100 tokens) situating the chunk in its document (company,
period, section…). It is applied to **both** the embedding index and the BM25 index.

**Q9.** The prompt includes the **whole document** for every chunk. Without caching, a document of
8k tokens split into 10 chunks would be paid 10 times. With caching, the document prefix is
processed once and reused; only the chunk part changes. S1 estimates ≈ $1.02 per million document
tokens.

**Q10.** BM25 scores and cosine similarities live on incompatible, query-dependent scales. Ranks
are comparable across lists without any normalization.

**Q11.** A cross-encoder needs one forward pass per (query, candidate) pair, with nothing
precomputed. Scoring a whole corpus per query is too slow and too costly. If N is too small, the
golden chunk may not be among the candidates, and the reranker cannot recover it: **stage 1
bounds the recall of the whole system**.

**Q12.** Search tools return short **snippets**; the model reads full chunks only on demand. The
variant without it (A-RAG Naive, one embedding tool returning full chunks) consumed roughly 4–10×
more retrieved tokens than naive RAG.
</details>

<details>
<summary>E1 — BM25</summary>

1. avgdl = (3 + 7 + 2 + 3) / 4 = **3.75**.
2. - `goodwill` is in 2 docs: IDF = ln((4 − 2 + 0.5)/(2 + 0.5)) = ln(1) = **0**.
   - `impairment` is in 1 doc: IDF = ln(3.5/1.5) = **0.8473**.
   - `revenue` is in 1 doc: **0.8473**.
3. Only `impairment` contributes (goodwill's IDF is 0). For d0: tf = 1, |d| = 3.
   Denominator: 1 + 1.5 × (1 − 0.75 + 0.75 × 3/3.75) = 1 + 1.5 × 0.85 = 2.275.
   Score = 0.8473 × 1 × 2.5 / 2.275 = **0.9311**.
4. d1: tf = 2, |d| = 7. Denominator: 2 + 1.5 × (0.25 + 0.75 × 7/3.75) = 2 + 1.5 × 1.65 = 4.475.
   Score = 0.8473 × 2 × 2.5 / 4.475 = **0.9467**. Two occurrences give barely more than one would
   in a shorter document: saturation **and** length normalization at work.
5. `goodwill` appears in exactly half of the documents, so its IDF is **exactly zero**: it is
   treated as non-informative and contributes nothing, even though it is in the query and in d0.
   On a real corpus of thousands of chunks it would be rare and highly informative. IDF depends on
   the corpus, not on the word.

Check:

```python
from ragbench.lexical import BM25, tokenize
docs = ["goodwill impairment recorded", "revenue grew revenue grew strongly this year",
        "goodwill unchanged", "tax rate fell"]
bm25 = BM25([tokenize(d) for d in docs])
print(bm25.get_scores(tokenize("goodwill impairment")))  # [0.9311, 0, 0, 0]
print(bm25.get_scores(["revenue"]))                       # [0, 0.9467, 0, 0]
```
</details>

<details>
<summary>E2 — RRF</summary>

1. $k = 60$:
   - A: 1/61 + 1/62 = 0.03252
   - C: 1/63 + 1/61 = 0.03227
   - B: 1/62 + 1/64 = 0.03175
   - E: 1/63 = 0.01587
   - D: 1/64 = 0.01563

   Order: **A, C, B, E, D**.
2. $k = 0$: A = 1 + 1/2 = 1.5, C = 1/3 + 1 = 1.333, B = 1/2 + 1/4 = 0.75, E = 1/3 = 0.333,
   D = 1/4 = 0.25. **Same order** here, but the gaps are far larger: with $k = 0$, a single first
   place weighs as much as a second place and a third place combined. Rankings *can* flip between
   k values when one document is first in one list and absent from the other.
3. Each appears in a single list, so it gets a single contribution. E's rank (3) is better than
   D's (4).

```python
from ragbench.fusion import rrf_fuse
rrf_fuse([["A", "B", "C", "D"], ["C", "A", "E", "B"]], k=60)
```
</details>

<details>
<summary>E3 — Sample size</summary>

1. Standard error = √(p(1 − p)/n).
   - n = 25: √(0.8 × 0.2 / 25) = 0.080 → half-width 1.96 × 0.080 ≈ **±15.7 points**.
   - n = 300: √(0.16/300) = 0.0231 → **±4.5 points**.
2. With n = 25, 80 % [64, 96] and 83 % [68, 98] overlap almost entirely: no claim possible. Even
   with n = 300 ([75.5, 84.5] vs [78.7, 87.3]), the intervals overlap. Comparing two independent
   intervals is very conservative. The charter plans a **paired test** (e.g. McNemar, or a paired
   bootstrap) on the *same* queries. Both configurations face identical difficulty, so only the
   queries where they *disagree* count, which removes most of the noise. That is why the golden set
   has ~300 auto-derived queries on top of the 25 hand-audited ones.
</details>
