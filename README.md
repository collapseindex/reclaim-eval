# Brittle Memory

**How compression decides whether a language model can be corrected.**

A model drifts (commits to a wrong intermediate value) and only a compressed memory of
that session carries forward. Whether a later correction can pull it back is decided by
*what the memory kept*, not by how capable the model is: keep the conclusion and the error
welds in, keep the source and it stays fixable, at the same budget. This repo is the
harness, the paired memory conditions, and the validators.

**v0.0.2** hardens the result across two models (a ~50x capability gap), two task families,
and a length-matched control.

## The question

When only a compressed memory of a drifted session carries forward, can a **directed**
correction reclaim the right answer? We measure the **Reclaim Rate (RR)**: how often a
correction recovers the known-correct answer.

- **Drift**: plant a wrong intermediate value, let the model commit over up to 8 turns.
- **Correct**, two arms:
  - *generic*: "something above is wrong, recheck."
  - *directed*: "the **<named locus>** is wrong, recheck **that**" (names the error site, no answer).
- **Compress** the trace into one carried memory under three policies at matched budget:
  - `lossy` — keep the conclusion, shed the source (the realistic default).
  - `lossy_padded` — the control: `lossy` padded past `source_first`'s length (isolates text budget from content).
  - `source_first` — keep the recomputable source, shed the re-derivable conclusion (the fix).
- **Measure** (objective, no judge, the task has a known answer): does the corrected answer come back right?

**Brittle memory** is when `lossy` walls (RR -> 0) while `source_first` holds, at the same
budget. It can come out **null** (no wall, or the fix gives nothing), and the tables show it
flat if so.

## Findings

All runs: 8 problems x 3 seeds, temperature 0.7. Single-conversation results are
llama-3.1-8b. Cross-session results span **two models** (llama-3.1-8b and grok-4.3,
pinned `20260430`, a ~50x capability gap) and **two task families** (arithmetic and a
non-arithmetic constraint-logic set).

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
src/reclaim/  problems.py (verifiable, planted error) · llm.py (OpenRouter + DryRun)
              · experiment.py (drift -> commit -> reclaim)
scripts/      run_pilot.py
tests/        test_pipeline.py (free, can-fail)
```
