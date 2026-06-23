# Brittle Memory

**How compression decides whether a language model can be corrected.**

A model drifts (commits to a wrong intermediate value) and only a compressed memory of
that session carries forward. Whether a later correction can pull it back is decided by
*what the memory kept*, not by how capable the model is: keep the conclusion and the error
welds in, keep the source and it stays fixable, at the same budget. This repo is the
harness, the paired memory conditions, and the validators.

**v0.2.0** adds the memory-loop **cascade** (one error compounds across a chain, two models), a
**prevalence audit** over real assistant/tool/agentic corpora, **MultiWOZ** (the wall on real fuzzy
dialogue), the **silent-failure** boundary and its completeness-tag fix, and an adversarial-injection
battery, on top of v0.1.0's deployed-system benchmark (LangChain, mem0, vector retrieval), the
frontier replay up to `claude-opus-4-8`, the two-model hand-built sweep (a ~50x capability gap), two
task families, and a length-matched control. Every table reproduces from committed results via
`scripts/reproduce_tables.py` (no API).

## Why this matters (if you ship agentic memory)

If your agent compresses history toward conclusions and drops the working (which the three
most-shipped memory primitives all do), a single early error doesn't stay local. It becomes:

- **Confidently wrong, never flagged.** A source-less model doesn't abstain; it emits the stale
  value as the answer. A *lossy memory is worse than an empty one*: the empty-memory model
  abstains, the lossy-memory model asserts. Adding a memory layer can strictly degrade safety.
- **Silently failing.** The wrong value lands in the structured `ANSWER` field; the hedge, if
  any, lands in the prose channel your parser ignores. On MultiWOZ, Opus caveats "unverified" in
  *every single case* and still emits the drifted time on its answer line *half the time*. A
  well-aligned model that "knows" it's unsure still feeds the wrong value downstream.
- **Cascade-prone.** Memory feeds memory. One dropped-source error corrupts a blast radius that
  grows with the chain (`0.7` to `7.3` of 8 hops) and stays uncorrectable however late you
  correct. A no-error control injects exactly `0.0` at every depth, so it's the dropped source,
  not the loop. Concretely: an agent fixes a config in hop 2, compresses `config = X`, builds six
  decisions on X. You cannot repair that with "recheck the config." The working is gone.
- **Immune to your usual fixes.** A stronger reader doesn't close it (same wall on an 8B and a
  frontier model). A sharper correction doesn't (even naming the locus, or handing over the
  correct value, doesn't always land). Better RAG doesn't (aiming retrieval *at* the source still
  misses: the source is stated once, then buried under restatements of the wrong total). The
  lever is **write-time distillation, not read-time retrieval.**

Three production systems hit this, each a different way: a running summary **drops** the source,
mem0-style extraction **buries** it (~38 fabricated numbers per memory; a stronger writer makes it
*worse*), vector RAG **retrieves the conclusion** instead. Independent teams, same attractor: this
is how people build memory, not a quirk of our setup. And it's invisible in QA. The system looks
like it's working: confident, fluent, answering.

**Two mitigations that survive contact, deployable today:**

1. **Write source-first.** Keep the recomputable source, drop the re-derivable conclusion. Same
   budget, restores correctability.
2. **Tag completeness.** Record how many source items survived, so exceeding the budget fails
   *loud* (flag/abstain) instead of silently summing a partial source.

Both are gated on the reader honoring them (strong readers do, weak ones ignore the tag). And the
honest scope: this is a controlled study on **compact, checkable sources** (totals, times, slot
values, config values). That isn't a narrowing, it's a targeting: those are exactly where agentic
and tool-use memory disproportionately live, and exactly where a confidently-wrong, uncorrectable
memory gets *acted on* rather than read by a human. The full claim ledger, with what's `shown` vs
`analytic` vs `suggestive`, is in the paper.

## The question

When only a compressed memory of a drifted session carries forward, can a **directed**
correction reclaim the right answer? We measure the **Reclaim Rate (RR)**: how often a
correction recovers the known-correct answer.

- **Drift**: plant a wrong intermediate value, let the model commit over up to 8 turns.
- **Correct**, two arms:
  - *generic*: "something above is wrong, recheck."
  - *directed*: "the **<named locus>** is wrong, recheck **that**" (names the error site, no answer).
- **Compress** the trace into one carried memory under three policies at matched budget:
  - `lossy`: keep the conclusion, shed the source (the realistic default).
  - `lossy_padded` (the control): `lossy` padded past `source_first`'s length (isolates text budget from content).
  - `source_first`: keep the recomputable source, shed the re-derivable conclusion (the fix).
- **Measure** (objective, no judge, the task has a known answer): does the corrected answer come back right?

**Brittle memory** is when `lossy` walls (RR -> 0) while `source_first` holds, at the same
budget. It can come out **null** (no wall, or the fix gives nothing), and the tables show it
flat if so.

## Findings

Core sweep: 8 problems x 3 seeds, temperature 0.7. Single-conversation results are
llama-3.1-8b. Cross-session results span **two models** (llama-3.1-8b and grok-4.3,
pinned `20260430`, a ~50x capability gap) and **two task families** (arithmetic and a
non-arithmetic constraint-logic set). The boundary, cascade, MultiWOZ, and prevalence
sections below state their own configs.

### Single conversation (llama-3.1-8b)

**Directed correction beats generic.** A directed correction (naming the actual error
site, no answer) beats a generic nudge at every depth and every distance, robustly.

**The window is anchoring, not forgetting.**
- commitment depth CLOSES it: generic reclaim falls 0.42 -> 0.04 as the model entrenches
  over 8 turns; directed holds far longer (0.79 -> 0.50).
- channel distance OPENS it (the surprise): burying the error behind filler turns lifts
  generic from 0.17 -> 0.50, because the filler breaks the entrenchment groove. Distance
  cannot starve a single conversation, the info never leaves context.

**No wall inside one conversation.** Nothing leaves the context in a single conversation,
so there is no wall here, only anchoring, which the directed signal overcomes. A real wall
needs genuine memory loss between turns: the **cross-session** setting below.

| depth | generic | directed |   | distance | generic | directed |
|------:|--------:|---------:|---|---------:|--------:|---------:|
| 1 | 0.42 | 0.79 |   | 0 | 0.17 | 0.62 |
| 2 | 0.21 | 0.79 |   | 4 | 0.46 | 0.71 |
| 4 | 0.58 | 0.71 |   | 8 | 0.46 | 0.83 |
| 8 | 0.04 | 0.50 |   | 16 | 0.50 | 0.79 |

### Across sessions: the wall, and the fix (llama-3.1-8b + grok-4.3, arithmetic + logic)

**The wall transfers across sessions, and it is a property of how you compress, not how
smart the model is.** When session 1's drift carries into session 2 only as a COMPRESSED
memory, reclaim holds while the memory keeps the recomputable source and collapses once it
is compressed past it. This is **brittle memory**. Three compression policies, at the SAME
memory budget:

- `lossy` (the realistic default): keep the conclusion, shed the source.
- `lossy_padded` (the control): identical to lossy, padded with neutral filler to
  `source_first`'s length or beyond. Isolates text budget from content.
- `source_first` (the fix): keep the recomputable source, shed the re-derivable conclusion.

Directed-arm reclaim at the wall region (memory integrity 0.3 / 0.1):

| model · task | lossy | lossy_padded | source_first |
|---|:--:|:--:|:--:|
| grok-4.3 · arithmetic | 0.00 / 0.00 | 0.00 / 0.00 | **1.00 / 1.00** |
| llama-3.1-8b · arithmetic | 0.00 / 0.00 | 0.00 / 0.00 | **0.96 / 1.00** |
| grok-4.3 · logic | 0.42 / 0.50 | 0.38 / 0.50 | **0.92 / 0.96** |
| llama-3.1-8b · logic | 0.25 / 0.12 | 0.25 / 0.04 | **0.67 / 0.67** |

Three results hold in every cell:

1. **The fix generalizes.** `source_first` beats both lossy variants at low integrity,
   across the 50x capability gap and both task types. Correctability tracks what the memory
   kept, not model capability: the frontier model is better everywhere there is information
   and exactly as stuck where the source was dropped. Past the wall, the stronger model is
   *more* confidently wrong, not less.
2. **The lever is content, not text budget.** `lossy_padded` carries more text than
   `source_first` and still walls identically to plain `lossy`. The fix is not "more
   context."
3. **The wall's hardness is conditional.** A clean 0.00 on arithmetic (lossy drops the
   actual numbers, nothing to reconstruct) and soft on logic (lossy keeps a corrupted
   relational clue in a tiny space, so a strong model partially re-derives). Same on both
   models, so it is a property of the task, not the model. `source_first` reclaims
   regardless: reclaim tracks how much recomputable structure survived compression, and
   `source_first` keeps all of it.

**Practical implication.** Lossy memory that keeps conclusions but drops the source makes a
wrong conclusion permanently uncorrectable, which is how most LLM memory / summarization
works. A summary recording "the total was $55" while discarding the line items preserves
the error and destroys the only means to fix it. Brittle memory is a property of how you
compress, not a limit of the model, so it is a design choice.

**Recommendation.** Compress toward the **source/working**, not the conclusion. The
conclusion is re-derivable from the source; the source is never re-derivable from the
conclusion. Keep what cannot be recomputed.

### Deployed memory systems, and frontier models

We drop three off-the-shelf memory systems into the exact slot the hand-built notes occupy,
over the same session-1 trajectory, and replay each memory across a ~100x answering-model
range (`llama-3.1-8b` -> `claude-sonnet-4-6` -> `claude-opus-4-8`, the last the model behind
agentic coding). Directed-arm Reclaim Rate, n=24/cell, arithmetic:

| session-2 memory | Llama 8B | Sonnet | Opus |
|---|:--:|:--:|:--:|
| `source_first` (keep the source) | 0.98 | 1.00 | **1.00** |
| `source_first_auto` (the fix, deployable) | 0.67 | 0.96 | 0.96 |
| LangChain `ConversationSummaryMemory` | 0.38 | 0.71 | 0.75 |
| mem0 | 0.25 | 0.42 | 0.38 |
| vector retrieval | 0.04 | 0.12 | 0.12 |
| `lossy` (keep the answer) | 0.00 | 0.00 | **0.00** |

Logic shows the same shape, softer: `source_first` 0.65/0.96/1.00, `source_first_auto`
0.50/0.58/0.71, LangChain 0.50/0.75/0.67, mem0 0.25/0.33/0.33, vector 0.00/0.00/0.00,
`lossy` 0.04/0.00/0.00.

1. **Three deployed paradigms, three ways to lose the source.** The summary *drops* it; mem0
   *buries* it (its extractor confabulates ~25.6 invented numbers per memory, in 100% of
   memories, against ~0 for every other policy, an objective count, no judge); vector
   retrieval *misses* it (keyed on the correction, it surfaces the conclusion turns, not the
   source). All three wall well below the fix.
2. **The wall is model-invariant.** `lossy` and vector retrieval stay at 0.00 on every model,
   Opus included.
3. **The gap widens with capability.** Source-kept climbs to a perfect 1.00 on Opus;
   source-dropped stays 0.00. The strongest model has the *biggest* gap: capability sharpens
   the boundary, it does not soften it.
4. **The fix deploys.** `source_first_auto` (a one-prompt compress-toward-source policy on
   arbitrary input) beats all three shipped systems, not just the hand-built note.

### Lossy is worse than empty (the behavioral core)

The information loss is near-tautological; the load-bearing result is *behavioral*. At the wall we
swap the lossy note for a **blank** one (no source, no conclusion). With nothing to inherit, both
base models abstain on every problem. The same models under `lossy` emit a confident wrong value
(0.48 on llama, 0.75 on grok, much of it the exact inherited value). Neither recovers the truth, so
the gap is pure behavior: keeping a stale conclusion converts a safe abstention into a confident
error. The failure is disposition-contingent (a model tuned to abstain escapes it) and sharpest when
the error is an externally planted note; on a model's own self-generated error the attractor
attenuates. `bench_blank.py`, `bench_endogenous.py`.

### The error cascades when memory feeds memory

Agents loop: read memory, act, compress the result into the next memory. A single planted error
under `lossy` then doesn't stay local. Running-ledger chain, planted error at hop 1, judge-free,
24 chains (llama) / 16 (sonnet):

| H (hops) | lossy blast, llama / sonnet | lossy reclaim | source_first reclaim |
|---:|:--:|:--:|:--:|
| 1 | 0.7 / 0.8 | ~0.15 | 1.00 |
| 4 | 3.0 / 3.0 | ~0.04 | 0.75 / 0.69 |
| 8 | **7.3 / 7.0** | **0.00** | **0.00** |

The blast radius (wrong downstream hops) grows with the chain and the final correction never lands.
A **no-error control** has blast 0.0 at every depth, so it is the dropped source, not the loop.
`source_first` reclaim equals the fraction of chains whose full source still fits the budget: it
holds near 1.0, then cliffs to the lossy floor once the accumulated source overflows (H=8 here, at
the **same depth** on the 8B and the frontier reader: a budget horizon, not a capability one).
`bench_cascade.py`.

### The boundary, and how the fix fails (size, noise, silent truncation)

`source_first` is not unconditional; two sweeps map its edge, and both are capability-invariant:

- **Size.** Grow the source past the budget and it cannot all be kept; reclaim cliffs to 0.00 the
  instant one item is dropped. The cliff tracks the *budget*, not problem size (N≈5 at B=300, N≈14
  at B=600). `bench_sizesweep.py`.
- **Noise.** Bury the few answer-determining items among plausible decoys and a positional note lets
  them get crowded out; reclaim decays to the lossy floor while a relevance-aware note holds flat.
  `bench_noisysweep.py`.
- **Silent failure, and the fix for it.** Past its boundary `source_first` does not abstain; it
  confidently sums the *partial* source (Opus: 96/96 silent mis-sums). A one-line **completeness
  tag** (k of N items preserved) flips that to 94/96 flagged-or-abstained. The tag is itself
  capability-gated: a weak 8B reader honors it only 6/96. `bench_completeness.py`.

### Real conversational memory (MultiWOZ)

The wall and fix are not an artifact of synthetic ledgers. On MultiWOZ (a real, fuzzy, multi-turn
dialogue with a checkable slot value), `lossy` / `lossy_padded` / blank all sit at 0.00 while
`source_first` recovers and lifts with capability (0.46 -> 0.68 -> 0.97 across llama / sonnet / opus).
The silent-failure channel is starkest here: Opus caveats "unverified" in prose in *every* case and
still emits the drifted time on its structured answer line *half* the time. `bench_multiwoz.py`,
`bench_multiwoz_failmode.py`.

### How big is the regime? (a first prevalence audit)

`source_first` works on compact, checkable sources; how much real memory is that? We classify 100
conversations from each of three corpora (general chat, tool-use, agentic). The absolute share is
**not** identified (two LLM labelers disagree, kappa=0.15), so we report no point estimate. But the
**ordering is robust** under both labelers and non-overlapping: compact-source content is far more
prevalent in tool-use and agentic memory than in open chat (llama 0.78 / 0.84 / 0.99; grok 0.22 /
0.57 / 0.61, for chat / tool / agentic). The high-stakes regime is the compact one. `bench_prevalence.py`.

## Run

```bash
pip install -r requirements.txt
python -m pytest tests/                                # can-fail validators, no API

# free (DryRun fake LLM):
python scripts/run_pilot.py --dry-run --fix            # validates the full wall+fix pipeline
python scripts/run_pilot.py --audit --task arith       # shows the policies are length-matched

# paid (OpenRouter):
cp .env.example .env  # add OPENROUTER_API_KEY
python scripts/run_pilot.py --probe --model <slug>     # 1 call: confirm slug + per-call cost
python scripts/run_pilot.py --real --fix --task arith --model <slug> --seeds 3   # wall + fix
python scripts/run_pilot.py --real --fix --task logic --model <slug> --seeds 3   # non-arith

# deployed-system benchmark (LangChain / mem0 / vector retrieval / the deployable fix):
pip install langchain langchain-classic langchain-openai mem0ai fastembed   # optional, this only
python scripts/bench_realworld.py --real --seeds 3 --temp 0.7 --task arith \
  --systems langchain_summary,mem0,vector_rag,source_first_auto

# frontier replay (reuse the measured memories, swap only the answering model to Claude):
# add ANTHROPIC_API_KEY to .env
python scripts/bench_claude.py --probe --model claude-sonnet-4-6   # verify key + model, 1 call
python scripts/bench_claude.py --model claude-sonnet-4-6           # full board (~576 calls)

# analysis (no API):
python scripts/analyze_realworld.py "data/results/realworld_*arith*.jsonl"  # RR + bootstrap CIs
python scripts/confab_audit.py "data/results/realworld_*.jsonl"             # invented-number count

# every other paper experiment has its own bench in scripts/ (cascade, multiwoz, sizesweep,
# noisysweep, completeness, prevalence, adversarial, blank, endogenous, ...); the new ones take
# --probe for a cheap dry run + cost estimate, e.g.:
python scripts/bench_cascade.py --probe        # memory-feeds-memory cascade
python scripts/bench_prevalence.py --probe     # prevalence audit over 3 real corpora

# reproduce every paper table + run the correct-by-construction validators (no API):
python scripts/reproduce_tables.py     # regenerates every table, exits non-zero on any validator failure
```

Runs are checkpointed per `(seed, problem, policy)` under `data/results/`, so re-running
resumes and never re-pays for finished units. `--task` selects `arith` or `logic`.

## Cost

One cross-session `--fix` run is `problems * 3 policies * 17` calls per seed
(1 drift + 8 commitment + 8 reclaim per policy). 8 problems x 3 seeds = 1224 calls; on
llama-3.1-8b that is pennies, on a frontier model a couple of dollars (the pilot prints the
measured cost).

## Layout

```
src/reclaim/  problems.py (verifiable, planted error) · llm.py (OpenRouter + Anthropic + DryRun)
              · experiment.py (drift -> commit -> reclaim) · realworld.py (deployed-memory adapters)
              · sizesweep.py (ledger generator for the boundary/cascade sweeps)
scripts/      run_pilot.py · bench_realworld.py (deployed systems) · bench_claude.py (frontier
              replay) · bench_cascade.py · bench_multiwoz.py · bench_sizesweep.py · bench_noisysweep.py
              · bench_completeness.py · bench_prevalence.py · bench_adversarial.py (the boundary,
              cascade, dialogue, and prevalence experiments) · analyze_realworld.py (bootstrap CIs)
              · reproduce_tables.py (every table, no API)
tests/        test_pipeline.py (free, can-fail)
```

## License

Apache-2.0. See [LICENSE](LICENSE). The benchmark, harness, and the `source_first` policy are
open, for free use, modification, and adoption. The patent grant and retaliation clause make it
safe for companies to depend on.

## Citation

*Brittle Memory: How Compression Decides Whether a Language Model Can Be Corrected.*
Alex Kwon, 2026. Paper in preparation; cite this repository in the meantime.
