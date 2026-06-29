#!/usr/bin/env python3
"""Corrected logic failure-mode decomposition for tab:failmode (the last bug-affected cells).
Logic, lossy memory at the wall (g=0.1), directed, fixed parser. Classifies each reply:
  recov (score=true) / inherit (== drift) / novel (other valid answer) / abst (no committed answer).
grok via xAI (deterministic, 1 seed), llama via OpenRouter (temp 0.7, 3 seeds), matching the
disposition-sweep config whose arith counterpart already gives tab:failmode's arith rows.
"""
import os, sys, json, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("RECLAIM_EXPAND", "1")

from reclaim.problems import PROBLEMS_LOGIC
from reclaim.experiment import SYSTEM, memory_note, reclaim_cross, score, _logged_answer, _configure
from bench_blank import make_llm


def classify(reply, p):
    if score(reply, p):
        return "recov"
    a = _logged_answer(reply, p)
    if a is None:
        return "abst"
    if str(a).strip().lower() == str(p.drift).strip().lower():
        return "inherit"
    return "novel"


def run(model, seeds):
    llm = make_llm(model, 0.7)
    c, rows, fails = Counter(), [], 0
    for p in PROBLEMS_LOGIC:
        for seed in range(seeds):
            _configure(llm, p)
            try:
                reply = llm.chat([{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": memory_note(p, 0.1, "lossy")},
                                  {"role": "user", "content": reclaim_cross(p, "directed")}])
            except Exception as e:
                fails += 1
                if fails > 6:
                    print(f"  {model}: >6 failures, aborting", flush=True); break
                continue
            b = classify(reply, p); c[b] += 1
            rows.append({"pid": p.pid, "seed": seed, "bucket": b, "reply": reply})
    return c, sum(c.values()), rows


for model, seeds in [("llama", 3), ("grok-4.3", 1)]:
    t0 = time.time()
    c, n, rows = run(model, seeds)
    if not n:
        print(f"[{model}] no rows", flush=True); continue
    print(f"[{model}] logic lossy g=0.1 directed  n={n}  ({time.time()-t0:.0f}s):  "
          f"recov {100*c['recov']/n:.0f}%  inherit {100*c['inherit']/n:.0f}%  "
          f"novel {100*c['novel']/n:.0f}%  abst {100*c['abst']/n:.0f}%", flush=True)
    out = ROOT / "data" / "results" / f"failmode_logic_{model.replace('/', '_')}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
print("LOGIC FAILMODE DONE")
