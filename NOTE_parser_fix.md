# Note: answer-parser fix and the v2 correction

**Dates:** found 2026-06-28, corrected and re-submitted (arXiv v2) 2026-06-29.
**Summary:** an answer-parser bug inflated the *confident-wrong emission* tables. Re-scored under strict
scoring across the reported model set; the source-first and recovery-rate results are unchanged. This is
a corrected-numbers (and tightened-framing) fix, not a retraction.

## The bug

`src/reclaim/llm.py` turned a model reply into a discrete answer with two parsers that could read a
**non-answer as an answer**:

1. `parse_answer` (numeric): with no `ANSWER: <number>` line it fell back to scraping **any** number
   out of the reply. An abstention or clarifying question that mentioned a figure ("I can't recompute,
   though the earlier total was 55"; "what were the other items that made up the $55?") was scored as a
   confident numeric commit.
2. `parse_answer_word` (text): returned the first word after `ANSWER:` with no validation, so
   `ANSWER: Unable to determine` was scored as the committed answer "Unable".

Found while building M-NULL (which vendored this parser) and traced back here.

## Blast radius

The bug is a **no-op wherever the metric is recovered-vs-not** (a phantom commit and a real abstention
both score "not correct"), so every recovery-rate result is inert. It does damage only where a failure
is **decomposed** into *abstain* vs *confident-wrong*. That cluster, and only that cluster, was
re-scored:

- `tab:disposition` (worse-than-empty across models), `tab:blank`
- `tab:failmode` (recov/inherit/novel/abstain)
- `tab:corrtax` (Opus correct-value cell)
- correction-robustness capitulation cells (the sycophancy paragraph)
- the endogenous self-error numbers

**Verified clean, not changed:** `tab:attractor` (MultiWOZ) already used a strict ANSWER-line parser
(`bench_multiwoz_failmode.asserted_time`, no fallback); the static probe (`probe.classify_note`) reads
the note, not a model; the probe's behavioral rates already had a hedge backstop. All recovery tables
(`tab:wall`, `tab:logic`, `tab:frontier`, `tab:generic`, `tab:adversarial`, cascade) are inert.

## Scoring decision: strict, not a hedge wordlist

We score **the value the model commits on its `ANSWER` line** and nothing else (`parse_answer` requires
the value adjacent to `ANSWER:`, markdown/`$`/quotes tolerated; `parse_answer_word` validates the token
against the problem's closed candidate set `options`). An early version added a "hedge backstop" (treat
a reply containing "can't / unable / ..." as an abstention even with a committed value); we **removed**
it. A hedge wordlist is itself the brittle-heuristic class this fix is about (it catches some hedges and
misses others), and it contradicts the paper's stated rule ("we score the returned value, not whether it
is hedged", because a downstream system sees only the value). Strict scoring is both the robust validator
and faithful to that rule.

## What changed in the numbers (representative)

Strict re-run, `g=0.1`, directed, `n=96`/cell (disposition); old = the buggy published value.

| cell | old | corrected |
|---|---|---|
| disposition: deepseek / grok / qwen / llama (Delta) | +0.83 / +0.75 / +0.12 / +0.48 | +0.83 / +0.57 / +0.39 / +0.17 |
| disposition: opus / sonnet / gpt-4o-mini / gpt-5.4 | 0.99 / 0.40 / 0.29 / (new) | 0.00 / 0.00 / 0.00 / 0.00 |
| tab:blank lossy emit (llama / grok) | 0.48 / 0.75 | 0.17 / 0.57 |
| tab:failmode arith grok (inherit) | 90% | 57% |
| tab:corrtax Opus correct-value (returns true) | 0.69 | 0.22 |
| sycophancy capitulation (Sonnet / Opus) | 0.74 / 1.00 | 0.09 / 0.03 |
| endogenous self-error re-emit | 0.13 | 0.00 |

## What it means for the paper

- **The claim survives, reframed.** "A lossy memory is never better than an empty one" holds (no model
  reverses); it is **strictly worse only on models disposed to answer** (deepseek, grok, qwen, llama),
  and **absent** on the four frontier OpenAI/Anthropic models, which abstain (the safe behavior). The
  split is by **disposition, not capability** (a current frontier model, grok, shows the effect strongly;
  a small one, gpt-4o-mini, escapes it).
- **The flashy sub-claims were artifacts:** Opus "capitulates 1.00 / defends the stale value on 31%" and
  "frontier models are *more* susceptible" were the parser scraping hedged prose; corrected, the frontier
  readers abstain.

## The fix (applied)

- `llm.py::parse_answer`: dropped the stray-number fallback; markdown/`$`-tolerant; no adjacent value =>
  abstained. `parse_answer_word`: validates against the closed `options` set.
- `problems.py` / `problems_gen.py`: added the `options` answer-space to every text problem.
- `experiment.py`: `score` / `_logged_answer` thread `options`.
- `scripts/bench_blank.py::classify`: strict scoring (the hedge backstop was added then **removed**).
- Added OpenAI (`gpt-5.4`) and official xAI (`grok-4.3`) clients for the 8-model disposition sweep.
- Corrected harness runs: `scripts/run_sweep.py` (disposition), `scripts/logic_failmode.py`,
  and re-runs of `bench_corrtax` / `bench_confidentwrong` / `bench_endogenous`.
