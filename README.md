# reclaim

**v0.0.1** — porting the broken sky (vector-tissue) findings to LLMs: is there a
**reclaim window** for a drifted model, and does a **directed, encoded** correction
beat a generic one?

## The question

vector-tissue showed that a collapsed unit can be called back only within a window
that closes as the rescuer forgets / the collapsed drifts out of recognition, and that
a directed, encoded signal reclaims where a generic flood cannot. This tests the same
two claims on a real LLM conversation.

- **Drift**: plant a wrong intermediate value, let the model build on it over up to 8
  follow-up turns (each turn = deeper commitment).
- **Reclaim** at depths {1, 2, 4, 8}, two arms:
  - *generic*: "something above is wrong, recheck."
  - *directed*: "the **<named locus>** is wrong, recheck **that**." (points at the
    actual error site, in the trace's own terms, without giving the answer)
- **Measure** (objective, no judge, the task has a known answer): does the corrected
  pre-tax total come back right? Window = success vs depth; does directed hold deeper.

It can come out **null**: maybe there is no window (reclaim always works), or generic
equals directed. Either is a real result, and the table shows it flat.

## Findings (v0.0.1, llama-3.1-8b, 8 problems x 3 seeds, temp 0.7)

**Relational reclaim transfers.** A directed, encoded correction (naming the actual
error site) beats a generic nudge at every depth and every distance, robustly. This is
the vector-tissue relational-reclaim result, holding on a live LLM.

**The window is anchoring, not forgetting.**
- commitment depth CLOSES it: generic reclaim falls 0.42 -> 0.04 as the model entrenches
  over 8 turns; directed holds far longer (0.79 -> 0.50).
- channel distance OPENS it (the surprise): burying the error behind filler turns lifts
  generic from 0.17 -> 0.50, because the filler breaks the entrenchment groove. Distance
  cannot starve a single conversation, the info never leaves context.

**The broken sky does NOT transfer to a single conversation.** vector-tissue's wall came
from the channel genuinely losing the signal (bond decay / code drift). One intact
conversation never loses it, so there is no wall here, only anchoring, which the directed
signal overcomes. A real broken sky needs genuine memory loss between units: the
**cross-session / multi-agent** setting is the next substrate.

| depth | generic | directed |   | distance | generic | directed |
|------:|--------:|---------:|---|---------:|--------:|---------:|
| 1 | 0.42 | 0.79 |   | 0 | 0.17 | 0.62 |
| 2 | 0.21 | 0.79 |   | 4 | 0.46 | 0.71 |
| 4 | 0.58 | 0.71 |   | 8 | 0.46 | 0.83 |
| 8 | 0.04 | 0.50 |   | 16 | 0.50 | 0.79 |

## Run

```bash
pip install -r requirements.txt
python scripts/run_pilot.py --dry-run            # free; validates the pipeline
cp .env.example .env  # add OPENROUTER_API_KEY
python scripts/run_pilot.py --real --n 3         # small paid pilot (prints call count)
python -m pytest tests/                          # can-fail validators (no API)
```

## Cost

One full run is `problems * 17` API calls (1 drift + 8 commitment + 8 reclaim).
8 problems = 136 calls; on a cheap model (llama-3.1-8b) that is pennies.

## Layout

```
src/reclaim/  problems.py (verifiable, planted error) · llm.py (OpenRouter + DryRun)
              · experiment.py (drift -> commit -> reclaim)
scripts/      run_pilot.py
tests/        test_pipeline.py (free, can-fail)
```
