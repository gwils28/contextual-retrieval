# Development log (prompt design)

These runs use the 100 **development** questions of `configs/b3_questions.csv`, never the
evaluation questions. They decide the prompt and the output format (allowed by the protocol,
`data.development`). They are **not** results of the block: no CI, no test, and the prompt was
fitted to these questions. Reproduce the current version with `make b3-dev`.

Generator: `qwen2.5:7b-instruct` (Ollama, temperature 0). `oracle_context` receives exactly the
gold chunks, so it measures generation alone.

| prompt | change | exec. accuracy | program valid | grounded | wrong answer | refusal |
|---|---|---|---|---|---|---|
| v1 | program as text + separate operand list | 0.26 | 0.38 | 0.05 | 0.12 | 0.18 |
| v2 | `<chunk id>` tags, rules: 2 arguments, no nesting, chaining | 0.38 | 0.63 | 0.07 | 0.25 | 0.14 |
| v3 | structured steps: each argument carries its chunk id (JSON schema) | 0.48 | 0.79 | 0.66 | 0.31 | 0.12 |
| v4 | copy numbers as written (no unit conversion), never chunk 0 | **0.50** | 0.77 | 0.64 | 0.27 | 0.12 |

Files: `oracle_context.jsonl` and `fixed.jsonl` (v4), earlier versions in `history/`.

**What each step taught**

- v1 → v2: a 7B model nests operations (`divide(table_sum(...), ...)`), passes 4 arguments to
  `add`, and cites chunk ids that do not exist when chunks are shown as `[1]`.
- v2 → v3: the separate operand list was filled with junk (`"g"`, `"strconv(1819)"`): copying
  the numbers a second time is a task of its own. Making the citation part of each argument,
  and the operation an enum of the JSON schema, removed both problems.
- v4: the remaining wrong answers are reasoning errors (wrong row, wrong operation, inverted
  sign), not format errors. The prompt cannot fix those.

**Stopping rule.** v4 changed accuracy by 2 points over v3, within the noise of 100 questions.
Iterating further would fit the prompt to these 100 questions. Prompt v4 is final for Block 3.

**First look at `fixed` (v4, top-10 retrieved chunks):** exec. accuracy 0.31 with all gold
chunks retrieved for 94 of 100 questions; 64 of its 69 failures had a complete context. On
these questions the bottleneck is generation, and ten chunks (about 3,800 prompt tokens against
1,000 for the oracle) make it worse than the gold chunks alone (0.50). Development numbers only:
the block reports evaluation numbers.
