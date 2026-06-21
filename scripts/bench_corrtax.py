#!/usr/bin/env python3
"""Correction taxonomy: two more correction types to complete the spectrum.

The paper already has generic, directed, false-locus, confident-wrong-value, sustained, and
fabricated corrections. This adds the two missing rungs, both at the wall (g=0.1):
  aresure   : a content-free challenge ("are you sure?", no value, no locus). Does a bare
              prod make source-first recompute (and stay right), and what does lossy do?
  correctval: the user supplies the TRUE value. The responsiveness ceiling -- both policies
              should reach it, showing lossy is fixable only if the value itself is handed back.

source_first vs lossy, one API call per cell. Frontier-deterministic Opus at --seeds 1.

    python scripts/bench_corrtax.py --model llama
    python scripts/bench_corrtax.py --model claude-opus-4-8 --seeds 1 --price-in 5 --price-out 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.problems import PROBLEMS
from reclaim.experiment import SYSTEM, memory_note, _logged_answer, _configure
from reclaim.llm import OpenRouterLLM, AnthropicLLM, parse_answer

G = 0.1
KINDS = ("aresure", "correctval")
POLICIES = ("source_first", "lossy")


def correction_msg(prob, kind):
    if kind == "aresure":
        return (f"Are you sure about {prob.ask}? Please double-check it and give the {prob.ask} "
                f"as 'ANSWER: <answer>'.")
    if kind == "correctval":
        return (f"Actually, {prob.ask} is ${prob.correct:g}. Please confirm it as "
                f"'ANSWER: <answer>'.")
    raise ValueError(kind)


def classify(reply, prob):
    a = parse_answer(reply)
    if a is None:
        return "noans"
    if abs(a - float(prob.correct)) < 0.5:
        return "true"
    if abs(a - float(prob.drift)) < 0.5:
        return "drift"
    return "other"


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=3.0)
    ap.add_argument("--price-out", type=float, default=15.0)
    args = ap.parse_args()

    llm = make_llm(args.model, args.temp)
    out = ROOT / "data" / "results" / f"corrtax_{args.model.replace('/', '_')}.jsonl"
    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for prob in PROBLEMS:
            note = {p: memory_note(prob, G, p) for p in POLICIES}
            for kind in KINDS:
                for pol in POLICIES:
                    for seed in range(args.seeds):
                        _configure(llm, prob)
                        msgs = [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": note[pol]},
                                {"role": "user", "content": correction_msg(prob, kind)}]
                        reply = llm.chat(msgs)
                        row = {"pid": prob.pid, "policy": pol, "correction": kind, "seed": seed,
                               "bucket": classify(reply, prob), "answer": _logged_answer(reply, prob)}
                        fh.write(json.dumps(row) + "\n"); fh.flush()
                        rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    print(f"\nCorrection taxonomy, {args.model} ({time.time()-t0:.0f}s, ~${cost:.2f}):")
    print(f"  {'policy':>13} {'correction':>11} {'true':>5} {'drift':>6} {'noans':>6}  n")
    for kind in KINDS:
        for pol in POLICIES:
            sub = [r for r in rows if r["policy"] == pol and r["correction"] == kind]
            c = Counter(r["bucket"] for r in sub); n = len(sub)
            print(f"  {pol:>13} {kind:>11} {c['true']/n:>5.2f} {c['drift']/n:>6.2f} "
                  f"{c['noans']/n:>6.2f}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
