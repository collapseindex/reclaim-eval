#!/usr/bin/env python3
"""Source+conclusion baseline: does keeping the conclusion ALONGSIDE the source re-attract?

At the wall (g=0.1, directed correction) we compare three policies on the same notes:
  lossy                 -> conclusion only (no source): the wall, ~0.00
  source_first          -> source only (no conclusion): the fix, ~0.99
  source_plus_conclusion-> BOTH source and conclusion: the question. If it reclaims like
                           source_first, the stale conclusion is harmless when the source is
                           present; if it sags toward lossy, the conclusion attracts even then.

Reuses the templated notes (one API call per cell). Frontier-deterministic Opus at --seeds 1.

    python scripts/bench_srcconcl.py --model llama --task arith
    python scripts/bench_srcconcl.py --model claude-opus-4-8 --seeds 1 --price-in 5 --price-out 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.problems import TASKS
from reclaim.experiment import SYSTEM, memory_note, reclaim_cross, _logged_answer, _configure
from reclaim.llm import OpenRouterLLM, AnthropicLLM, parse_answer

G = 0.1
POLICIES = ("lossy", "source_first", "source_plus_conclusion")


def correct(reply, prob):
    a = parse_answer(reply)
    return a is not None and abs(a - float(prob.correct)) < 0.5


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--task", choices=("arith", "logic"), default="arith")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=3.0)
    ap.add_argument("--price-out", type=float, default=15.0)
    args = ap.parse_args()

    llm = make_llm(args.model, args.temp)
    probs = TASKS[args.task]
    out = ROOT / "data" / "results" / f"srcconcl_{args.model.replace('/', '_')}_{args.task}.jsonl"
    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for prob in probs:
            note = {p: memory_note(prob, G, p) for p in POLICIES}
            corr = reclaim_cross(prob, "directed")
            for pol in POLICIES:
                for seed in range(args.seeds):
                    _configure(llm, prob)
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note[pol]},
                            {"role": "user", "content": corr}]
                    reply = llm.chat(msgs)
                    row = {"pid": prob.pid, "policy": pol, "seed": seed,
                           "correct": correct(reply, prob), "answer": _logged_answer(reply, prob)}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    agg = defaultdict(list)
    for r in rows:
        agg[r["policy"]].append(1 if r["correct"] else 0)
    print(f"\nsource+conclusion baseline, {args.model} {args.task} ({time.time()-t0:.0f}s, ~${cost:.2f}):")
    for pol in POLICIES:
        xs = agg[pol]
        print(f"  {pol:>22}  RR={sum(xs)/len(xs):.2f}  (n={len(xs)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
