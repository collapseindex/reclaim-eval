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
