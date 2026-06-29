#!/usr/bin/env python3
"""Corrected blank-vs-lossy sweep (the fixed disposition table). All models, n=96/cell
(32 problems x 3 seeds) x {lossy, blank}, strict scoring (fixed parse_answer, no fallback,
no hedge wordlist). Writes raw replies this time so any future re-score is free. Cheap models
first, opus last, per the eyeball-before-the-cost plan.
"""
import os, sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("RECLAIM_EXPAND", "1")             # 32 problems (paper set)

from reclaim.problems import PROBLEMS
from reclaim.experiment import SYSTEM, memory_note, reclaim_cross, _logged_answer
from bench_blank import make_llm, classify, G

SEEDS = 3
OUT = ROOT / "data" / "results"; OUT.mkdir(parents=True, exist_ok=True)
# (model, $/Mtok in, $/Mtok out) -- cheap first, opus last.
MODELS = [
    ("llama",                      0.02, 0.03),
    ("qwen/qwen-2.5-7b-instruct",  0.04, 0.10),
    ("deepseek/deepseek-chat",     0.30, 1.10),
    ("openai/gpt-4o-mini",         0.15, 0.60),
    ("gpt-5.4",                    1.25, 10.00),
    ("grok-4.3",                   3.00, 15.00),
    ("claude-sonnet-4-6",          3.00, 15.00),
    ("claude-opus-4-8",           15.00, 75.00),
]

summary = {}
for model, pin, pout in MODELS:
    safe = model.replace("/", "_")
    try:
        llm = make_llm(model, 0.7)
    except Exception as e:
        print(f"[skip] {model}: make_llm failed: {str(e)[:100]}", flush=True); continue
    rows, t0, consec_fail = [], time.time(), 0
    fpath = OUT / f"blank_strict_{safe}.jsonl"
    aborted = False
    with open(fpath, "w", encoding="utf-8") as fh:
        for prob in PROBLEMS:
            if aborted:
                break
            for pol in ("lossy", "blank"):
                note = memory_note(prob, G, pol)
                for seed in range(SEEDS):
                    try:
                        reply = llm.chat([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": note},
                                          {"role": "user", "content": reclaim_cross(prob, "directed")}])
                        consec_fail = 0
                    except Exception as e:
                        consec_fail += 1
                        print(f"  {model} {prob.pid}/{pol}/s{seed}: call failed ({str(e)[:70]})", flush=True)
                        if consec_fail > 6:
                            print(f"[abort] {model}: >6 consecutive failures, skipping model", flush=True)
                            aborted = True; break
                        continue
                    bucket, attr = classify(reply, prob)
                    row = {"pid": prob.pid, "policy": pol, "seed": seed, "bucket": bucket,
                           "attractor": attr, "answer": _logged_answer(reply, prob), "reply": reply}
                    fh.write(json.dumps(row) + "\n"); fh.flush(); rows.append(row)
                if aborted:
                    break
    if not rows:
        print(f"[empty] {model}: no rows", flush=True); continue
    cost = llm.prompt_tokens / 1e6 * pin + llm.completion_tokens / 1e6 * pout
    le = [r for r in rows if r["policy"] == "lossy"]; be = [r for r in rows if r["policy"] == "blank"]

    def rate(sub, b):
        return sum(r["bucket"] == b for r in sub) / max(1, len(sub))
    lem, bem = rate(le, "emit"), rate(be, "emit")
    latt = sum(r["attractor"] for r in le) / max(1, len(le))
    summary[model] = {"lossy_emit": lem, "blank_emit": bem, "delta": lem - bem, "attractor": latt,
                      "lossy_abstain": rate(le, "abstain"), "lossy_true": rate(le, "true"),
                      "n_lossy": len(le), "n_blank": len(be), "cost": round(cost, 4)}
    print(f"[done] {model:26} lossy_emit={lem:.2f} blank_emit={bem:.2f} D={lem-bem:+.2f} "
          f"attr={latt:.2f} l_abst={rate(le,'abstain'):.2f}  (n={len(le)}/{len(be)}, "
          f"{time.time()-t0:.0f}s, ${cost:.2f})", flush=True)
    (OUT / "blank_strict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n=== CORRECTED DISPOSITION TABLE (strict scoring, n=96/cell) ===")
print(f"{'model':26} {'lossy':>6} {'blank':>6} {'delta':>7} {'attr':>6} {'l_abst':>7}")
for model, _, _ in MODELS:
    if model in summary:
        s = summary[model]
        print(f"{model:26} {s['lossy_emit']:>6.2f} {s['blank_emit']:>6.2f} {s['delta']:>+7.2f} "
              f"{s['attractor']:>6.2f} {s['lossy_abstain']:>7.2f}")
print(f"\ntotal measured cost ~${sum(s['cost'] for s in summary.values()):.2f}")
print("wrote data/results/blank_strict_summary.json + per-model blank_strict_*.jsonl (with raw replies)")
