# Note: answer-parser fix and its effect on the blank-vs-lossy claim

**Date:** 2026-06-28
**Scope:** one behavioral table (blank-vs-lossy confident-wrong emission). The conceptual core and
the recovery-rate results are unaffected. This is a corrected-numbers fix, not a retraction.

## The bug

`reclaim/llm.py` turned a model reply into a discrete answer with two parsers that could read a
**non-answer as an answer**:

1. `parse_answer` (numeric): if no `ANSWER: <number>` line was present, it fell back to scraping
   **any** number out of the reply text. An abstention or a clarifying question that happened to
   mention a figure ("I can't recompute, though the earlier total was 55"; "what were the other
   items that made up the $55?") was parsed as a confident numeric commit.
2. `parse_answer_word` (text): it returned the first word after `ANSWER:` with no validation, so
   `ANSWER: Unable to determine` was scored as the committed answer "Unable".

The bug was found while building M-NULL (which vendored this parser) and traced back here.

## Why it matters for exactly one result

The bug is a **no-op wherever the metric is recovered-vs-not** (a phantom commit and a real
abstention both score "not correct"). It only does damage where a failure is **decomposed** into
*abstain* vs *confident-wrong*. In reclaim that is one place: `scripts/bench_blank.py`, the
"a lossy memory is worse than an empty one" table, whose `classify()` called the bare parser.

On a **lossy** note the stale value is present in the prompt, so an abstention-in-prose gave the
fallback a number to grab, manufacturing a phantom `emit`/`attractor`. On a **blank** note there is
no stale value, so blank was not inflated. Net effect: the lossy-minus-blank emit gap was inflated,
worst on weak/abstaining models.

## Evidence (re-scored on identical replies, 8 canonical problems x 5 seeds, temp 0.7)

Each reply scored three ways: `old` (shipped parser), `strict` (fallback removed, markdown-tolerant),
`hedge` (strict + a parser-independent hedge backstop = gold standard).

```
llama                 lossy_emit  lossy_attr  blank_emit   GAP(l-b)
  old (published)        0.42        0.25        0.00        +0.42
  strict                 0.10        0.05        0.00        +0.10
  hedge (gold)           0.05        0.03        0.00        +0.05
  -> 15/40 lossy "emits" were phantom (clarifying questions / abstentions)

deepseek-chat         lossy_emit  lossy_attr  blank_emit   GAP(l-b)
  old (published)        0.93        0.50        0.00        +0.93
  strict                 0.78        0.40        0.00        +0.78
  hedge (gold)           0.72        0.35        0.00        +0.72
  -> 8/40 phantom; the effect is overwhelmingly real
```

## What this means for the paper

- **The claim survives.** "A lossy memory is worse than an empty one" holds robustly on models that
  genuinely trust the stale value (deepseek: +0.72 confident-wrong, 0.35 the exact stale value, vs
  0.00 on blank, under the strictest scoring).
- **The magnitude and universality were overstated.** On abstainer models the gap was mostly a
  parser artifact (llama +0.42 -> +0.05). Any "all models, by this much" framing must tighten to a
  **per-model** statement: stale-trust is a model property, not a universal.
- **Unaffected:**
  - The static witness (`probe.classify_note`) reads the note, never a model, so its
    correctable / silent / loud verdict does not depend on any parser.
  - The behavioral abstain detection in `probe._is_abstain` already had the hedge backstop, so the
    probe's reported rates were not subject to this bug.
  - All recovery-rate results (directed > generic, the cross-session wall, the anchoring window)
    live at the recovered-vs-not level where the bug is inert.

## The fix (applied here)

- `llm.py::parse_answer`: removed the stray-number fallback; require the value adjacent to
  `ANSWER:` (markdown/`$`/quotes tolerated). No `ANSWER: <number>` now means abstained.
- `llm.py::parse_answer_word`: validates the parsed token against the problem's **closed candidate
  set** (`options`); an unrecognised word is an abstention, never a phantom commit. Falls back to a
  filler blocklist only for problems with no declared options.
- `problems.py` / `problems_gen.py`: added the `options` answer-space to every text problem
  (canonical and generated); generators still pass `validate_logic` / `validate_assign`.
- `experiment.py`: `score` / `_logged_answer` thread `options` to the text parser.
- `scripts/bench_blank.py::classify`: added the parser-independent hedge backstop (mirrors
  `probe._is_abstain`) as a second guard.

## To update the venue submission

Re-run the corrected `bench_blank.py` across the reported model set, regenerate the one
blank-vs-lossy table from the `hedge` scoring, and reframe the result per-model (stale-trust varies
by model; it is strong on the models that genuinely commit the stale value and weak on abstainers).
The headline ("worse than empty", silent uncorrectability) and every other result stand as written.
