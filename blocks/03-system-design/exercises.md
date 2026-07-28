# Block 3 — Exercises

Try each question before opening the answers at the bottom. E1–E3 are **pen and paper**; E4–E5
use `ragbench` and the files in [`results/evaluation/`](results/evaluation/).

## Control questions

**Q1.** Why is retrieval scored on the chunks handed to the generator, and never on the answer?
What would you lose by reporting only end-to-end accuracy?

**Q2.** The agent starts from fixed's top-10 and can only add evidence. Why is that design
choice important for interpreting a loss of the agent against fixed?

**Q3.** Why does the generator write a program instead of the final number?

**Q4.** The agent's wrong-answer rate (0.31) is lower than fixed's (0.40). Is the agent the
safer system? Why or why not?

**Q5.** Fixed vs agentic: −10 points, Holm p = 0.38. Write the correct one-sentence conclusion,
and one incorrect conclusion people often write.

**Q6.** If no gap is significant, on what ground can the ADR still choose fixed?

**Q7.** What property must an escalation signal have for a cascade to help? Which number in this
block shows that the grounding check lacks it?

**Q8.** In development, the agent already lost on its stop rule. Why was the rule not changed
before the evaluation run?

**Q9.** Why is the oracle useful even though it cannot be deployed?

**Q10.** Give two reasons why FinQA may favour the fixed pipeline over an agent.

## Exercises

**E1 — Diagnosis table (pen and paper).** A system answers 100 questions. All gold chunks are in
context for 90 of them; it answers 40 of those 90 correctly and 2 of the 10 others correctly.
Fill the 2×2 diagnosis table. Where should the next engineering effort go?

**E2 — McNemar by hand.** Two systems on the same 100 questions: both right on 20, only A right
on 9, only B right on 2, both wrong on the rest. (a) Which questions carry evidence? (b) Under
"no difference", what distribution does the number of B-only wins follow? (c) Compute the
two-sided exact p-value, then its Holm-adjusted value if this is the smallest of 6 p-values.

**E3 — Cascade arithmetic.** Fixed costs 1 LLM call per question; the agent costs 6 more. A
cascade escalates a share *e* of the questions. (a) Mean LLM calls as a function of *e*. (b) With
*e* = 0.65, compare with this block's measured 5.1. (c) At what *e* does the cascade cost half of
always running the agent (7 calls: fixed is not run first)?

**E4 — Is the signal any good? (code).** Using `results/evaluation/fixed.jsonl`, compute fixed's
accuracy on answered questions split by `grounded`. Then do the same with a signal of your own:
`context_tokens` above vs below its median. Does either separate right from wrong answers?
(Exploratory: this is analysis, not a new result of the block.)

**E5 — The tolerance pitfall (code).** With `ragbench.program.answers_match`, find a predicted
value that is 3 % off a gold answer of 0.012 and still matches, and one that is 3 % off a gold of
120 and does not. Explain the difference from the rule.

---

## Answers

**Q1.** Mixing stages hides where failures come from. A 0.28 accuracy could mean retrieval misses
or a reader that misreads good context; the fix is opposite in each case. Here the split shows 67
generation failures against 5 retrieval failures for fixed: better search would not help.

**Q2.** Since the agent never starts from less than fixed, a loss cannot be blamed on a worse
start. It must come from the loop itself: the stop rule, the extra context, or both.

**Q3.** LLMs, especially small ones, are unreliable at arithmetic. A program is executed
exactly by code; the LLM only has to choose the numbers and the operations. It also makes every
number checkable (citation per argument).

**Q4.** No. It refuses 33 % of the questions; fewer wrong answers come from answering less, not
from answering better. Its accuracy is lower (0.18 vs 0.28). Safety needs low wrong answers *at a
useful answer rate*, and an abstention rule tested on development data.

**Q5.** Correct: "A difference between fixed and agentic is not shown on 100 questions (Holm p =
0.38)." Incorrect: "Fixed and agentic perform the same" or "fixed is significantly better".

**Q6.** Burden of proof: the agent costs 7.6× more per question and 2.7× more latency. A more
expensive option has to show a gain to be adopted; it does not.

**Q7.** It must separate right from wrong answers: fire mostly on wrong ones. Fixed is right on
0.31 of grounded answers and 0.28 of ungrounded ones — the check barely separates them, and fires
on half the answers.

**Q8.** Changing a frozen rule after seeing results tunes the protocol to the outcome; the
evaluation would then no longer test a design fixed in advance. The problem is reported instead,
and the ADR lists it as a reason to revisit.

**Q9.** It measures the generator alone, with perfect retrieval. It tells you the ceiling of the
reader (0.44 here) and the gap better context could still buy (0.28 → 0.44).

**Q10.** Questions need one or two facts from a single page, and fixed retrieval already finds all
of them for 94 % of questions — little is left for an agent to find. Agents are designed for open,
multi-hop, multi-document search.

**E1.**

| | right | wrong |
|---|---|---|
| all gold in context (90) | 40 | 50 |
| gold missing (10) | 2 | 8 |

50 generation failures vs 8 retrieval failures: work on the reader (model, context, format).

**E2.** (a) Only the 11 discordant questions (9 A-only, 2 B-only). (b) Binomial(11, 0.5).
(c) P(X ≤ 2) = (1 + 11 + 55) / 2048 = 67 / 2048 ≈ 0.0327; two-sided p ≈ 0.065. Holm, smallest of
6: 6 × 0.065 ≈ 0.39 — not significant. (This is the fixed vs cascade comparison of the block.)

**E3.** (a) 1 + 6*e*. (b) 1 + 6 × 0.65 = 4.9, close to the measured 5.1 (the agent's real mean,
6.3 calls including its final answer, differs a little from 6). (c) 1 + 6*e* = 3.5 → *e* ≈ 0.42.

**E4.** Sketch:

```python
import pandas as pd
df = pd.read_json("results/evaluation/fixed.jsonl", lines=True)
ans = df[df.status == "answer"]
print(ans.groupby("grounded").correct.agg(["size", "mean"]))
ans = ans.assign(long=ans.context_tokens > ans.context_tokens.median())
print(ans.groupby("long").correct.agg(["size", "mean"]))
```

Grounded: 45 at 0.31, ungrounded: 50 at 0.28 (as in `exploratory.csv`). A useful signal would
show a large gap; check your context-length split the same way. Whatever you find, it was chosen
after seeing the data: it would need new development questions to be trusted.

**E5.** `answers_match(0.01164, "0.012")` is True (|diff| = 0.00036 ≤ 0.005);
`answers_match(116.4, "120")` is False (|diff| = 3.6 > 0.005 × 120 = 0.6). Below 1, the tolerance
`0.005 · max(|gold|, 1)` is a fixed 0.005 in absolute terms, i.e. a large relative error on small
values; above 1 it is a true 0.5 % relative tolerance.
