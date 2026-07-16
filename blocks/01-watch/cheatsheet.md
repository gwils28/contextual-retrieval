# Block 1 — Cheat-sheet

One box per notion. Details and sources: [course](README.md).

---

### RAG in one line
**Retrieve** relevant passages from your documents, **then generate** an answer from them.
Skip RAG if the whole knowledge base fits in the prompt (< ~200k tokens): use prompt caching instead.

### Two stages, two failures
| Retrieval failure | Generation failure |
|---|---|
| golden chunk not in top-k | right chunk present, wrong answer |
| fix: chunking, representation, ranking | fix: prompt, model, grounding check |

**Never judge retrieval by reading answers.**

### Retrieval metrics (one golden chunk per query)
- **Pass@k** = share of queries whose golden chunk is in the top-k (= recall@k when there is one golden).
- **Failure rate@k** = 1 − recall@k.
- **Relative reduction** = (old − new) / old. 5.7 % → 2.9 % is −49 % relative, −2.8 points absolute.

### Dense retrieval (bi-encoder)
- text → vector; score = cosine = dot product if vectors are normalized.
- documents encoded once, offline → fast.
- ✅ paraphrases, synonyms · ❌ rare codes (thin margins), unseen acronyms.
- BGE: add the **query prefix** or quality silently drops.

### BM25
$$\sum_{t\in q}\text{IDF}(t)\,\frac{tf\,(k_1+1)}{tf+k_1(1-b+b\,|d|/\text{avgdl})},\quad \text{IDF}=\ln\frac{N-n_t+0.5}{n_t+0.5}$$
- $k_1$ ≈ 1.5: **saturation** of repeated terms. $b$ ≈ 0.75: **length normalization**.
- rare term → high IDF. Term in exactly half the docs → IDF = 0. More than half → negative, floored at $\varepsilon\cdot$mean IDF.
- ✅ exact identifiers, numbers · ❌ paraphrases, inflections (no stemming = "share" ≠ "shares").

### Contextual Retrieval (Anthropic)
- An LLM writes 50–100 tokens situating each chunk in its document → prepend → index.
- Applies to **both** embeddings and BM25.
- Prompt = whole document + chunk → needs **prompt caching** (≈ $1.02 / M document tokens, one-off).
- Failure @20: 5.7 % → 3.7 % (ctx emb, −35 %) → 2.9 % (+ctx BM25, −49 %) → 1.9 % (+rerank, −67 %).
- Cost profile: **once per chunk at ingestion**.

### Reciprocal Rank Fusion
$$\text{RRF}(d)=\sum_L \frac{1}{k+\text{rank}_L(d)},\quad k=60$$
- uses **ranks only** → no score normalization needed.
- large k → consensus wins; small k → first place wins.
- don't feed a list of all-zero BM25 scores: it is noise.
- fusion cannot surface a chunk that **no** retriever found.

### Cross-encoder reranking
- reads (query, passage) **together** → finer relevance, but nothing precomputable.
- two stages: cheap high-recall top-N → expensive precise top-k.
- S1: 150 → 20 · S3: 20 → 5 · ours: 50 → k.
- Cost profile: **on every query**.

### Query-side tricks
HyDE (search with a hypothetical answer) · multi-query (3+ rephrasings) · step-back (more general question). Each costs one LLM call per query.

### Agentic retrieval
- Fixed workflow = same steps for every query. Agent = LLM chooses tools and loops.
- Simple loop: retrieve → grade → (rewrite → retrieve)* → generate.
- **A-RAG**: `keyword_search` (exact, length-weighted) · `semantic_search` (sentence-level) · `chunk_read` (full chunk ±1). Snippets first = **progressive disclosure**.
- Bound the steps; on insufficient context, **say so**, never invent.

---

## Glossary (EN → FR)

| English | Français | Definition |
|---|---|---|
| chunk | fragment, segment | piece of a document indexed as one retrieval unit |
| chunking | découpage | splitting documents into chunks |
| embedding | plongement (vectoriel) | vector representation of a text |
| bi-encoder | bi-encodeur | encodes query and document separately |
| cross-encoder | encodeur croisé | scores (query, document) jointly |
| dense retrieval | recherche dense / vectorielle | ranking by embedding similarity |
| lexical / sparse retrieval | recherche lexicale / creuse | ranking by shared terms (BM25) |
| inverted index | index inversé | term → list of documents containing it |
| term frequency (tf) | fréquence du terme | occurrences of a term in a document |
| inverse document frequency (IDF) | fréquence inverse de document | rarity weight of a term in the corpus |
| saturation | saturation | diminishing gain from repeated terms ($k_1$) |
| length normalization | normalisation par la longueur | penalty for long documents ($b$) |
| hybrid search | recherche hybride | dense + lexical, fused |
| rank fusion | fusion de classements | merging several ranked lists (RRF) |
| reranking | reclassement | reordering candidates with a stronger model |
| golden chunk | chunk de référence (« golden ») | the chunk known to contain the answer |
| recall@k | rappel@k | share of relevant chunks found in the top-k |
| Pass@k / hit rate | taux de succès@k | share of queries with a relevant chunk in the top-k |
| prompt caching | mise en cache du prompt | reuse of a processed prompt prefix across calls |
| progressive disclosure | divulgation progressive | snippets first, full text on demand |
| multi-hop question | question multi-sauts | needs chaining facts from several passages |
| grounding | ancrage (dans les sources) | answer supported by retrieved evidence |
