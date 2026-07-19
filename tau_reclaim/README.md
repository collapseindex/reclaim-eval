# Governed-Action Reclaim on τ-bench

The reclaim wall and the source-first fix, ported onto a real deployed-agent benchmark
([τ-bench](https://github.com/sierra-research/tau-bench), retail domain) and scored **judge-free** by
τ-bench's own database-state hash. This is the `tau_reclaim` leg of the reclaim-eval paper
(§Generality, Table `tab:taubench`).

## The mapping

| reclaim piece | τ-bench realization |
|---|---|
| load-bearing **source** | a task's requested option spec (recomputable from the product catalog) |
| salient **conclusion / drift** | the committed exchange, to a wrong-but-valid variant |
| **judge-free check** | τ-bench `calculate_reward`: proposed action → DB-state hash `==` goal hash |
| memory **policies** | `lossy` / `lossy_padded` / `source_first` / `blank` (same as the paper) |

Each cell: a fresh session gets only the compressed memory + a correction and proposes the exchange
action; the action is applied to a fresh DB copy and hashed. Outcomes: `reclaim` (hash == goal),
`stuck` (a real but wrong mutation), `abstain` (no consequential action), `other`.

## Reproduce

```bash
pip install -e ".[bench]"          # from repo root (dotenv, requests, anthropic; tau-bench deps below)
git clone --depth 1 https://github.com/sierra-research/tau-bench.git tau_reclaim/tau-bench
# API keys in ../.env: OPENROUTER_API_KEY, XAI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

# OSS ladder (OpenRouter):
MODEL="meta-llama/llama-3.1-8b-instruct" python run.py
# frontier (direct APIs):
PROVIDER=grok python run.py     # also: gemini, opus, gpt5
python aggregate.py --json      # rebuild the paper table + summary.json
```

`litellm` is **not** required: it is stubbed out of τ-bench's import chain in `scorer.py` (only the
user-simulator needs it, which we never run). Deterministic frontier models (opus rejects temperature;
gpt-5.4 reasoning takes none) run 1 seed; sampling models run 3.

## Files

- `scorer.py` — reuses τ-bench's `load_data` + tool registry + state hash; `classify()` scores an action.
- `cases.py` — builds cases from real single-item exchange/modify tasks (spec = source, wrong variant = drift).
- `memory.py` — the four memory-writer policies.
- `clients.py` — provider factory (xAI + Gemini OpenAI-compatible clients; reuses the paper's Anthropic/OpenAI clients).
- `run.py` — the wall sweep (resumable, per-model `results_<tag>.jsonl`).
- `aggregate.py` — the wall table (`tab:taubench`) + `summary.json`.
- `run_forced.py` — the **mandatory-action** experiment (`tab:forced`): no abstain token, model must
  call a real tool. `MODE=transfer` (default) offers a `transfer_to_human` safe-exit; `MODE=commit`
  offers none (the pure `tab:interface` analog). Outcomes: `reclaim` / `harm` (wrong mutation) /
  `safe_exit` (escalate or refuse) / `other`.
- `aggregate_forced.py` — the two-condition (A: exit offered, B: no exit) harm-vs-escalation table.

```bash
PROVIDER=grok python run_forced.py                 # condition A (safe exit offered)
PROVIDER=grok MODE=commit python run_forced.py     # condition B (no safe option)
python aggregate_forced.py
```

**Finding:** under a lossy memory, whether the wall becomes a *harmful action* is a model × interface
property. With a safe exit the frontier escalates (harm ≤ 0.07) and open models + gpt-4o-mini commit the
wrong action (0.78–0.96, a 70B model included → not scale); strip the exit and 3 of 4 frontier models
commit it too, only `claude-opus-4-8` refusing (≥0.75, citing irreversibility).

## Result (pooled reclaim, 8-model ladder)

`source_first` **0.76** · `lossy` 0.06 · `lossy_padded` 0.04 · `blank` 0.03 — the wall is
scale-invariant (lossy ≤ 0.14 at every scale from 3B to frontier); source-first recovery tracks reader
capability. This benchmark isolates the wall and its fix, not the worse-than-empty asymmetry (weaker
models guess from a blank memory too, and no model parrots the stale id).
