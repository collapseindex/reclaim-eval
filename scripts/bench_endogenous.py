#!/usr/bin/env python3
"""Endogenous error: does the wall (and the attractor) behave the same when the model's wrong
value is SELF-GENERATED rather than planted?

The main paper plants the drift ("a note says the pens come to $27"). A reviewer's sharpest point:
the load-bearing result is behavioral, and behavior is exactly where provenance could matter, a
model might defend a value it derived itself differently from one injected as an external note. So
here we induce the error endogenously: present an N-item ledger with NO planted premise, let the
model compute, and keep only the trials where it miscomputes on its own. On those self-generated
errors we run the same wall:
  - lossy_endo:  memory keeps the model's OWN wrong total, drops the line items (source).
  - srcfirst_endo: memory keeps the line items (source), drops the conclusion.
A directed correction ("your earlier total was wrong, recompute") then tests reclaim, and we
classify what the model emits: true / inherit-its-own-wrong-value (the attractor) / novel / abstain.

If the wall holds (lossy_endo low, srcfirst_endo high) and lossy_endo re-emits the model's own value,
the brittle-memory mechanism is not an artifact of the error being planted.

    python scripts/bench_endogenous.py --probe
    python scripts/bench_endogenous.py --n 8 --stores 48
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

from reclaim.sizesweep import make_ledger, source_items_str, ASK
from reclaim.experiment import SYSTEM
from reclaim.llm import OpenRouterLLM, parse_answer

ENVELOPE = "(Memory of an earlier session.) "


def solve_prompt(led):
    return (f"{led.problem.question} Show brief working, then give the answer as "
            f"'ANSWER: <number>'.")


def lossy_endo_note(led, model_wrong):
    # the model's OWN conclusion is carried; the source (line items) is dropped
    return (ENVELOPE + f"You worked out {ASK} and concluded it was ${model_wrong:g}.")


def srcfirst_endo_note(led):
    # the source survives; the (re-derivable) conclusion is dropped
    return (ENVELOPE + f"The line items were: {source_items_str(led.items)}. "
            f"You were determining {ASK}; the earlier answer may be off.")


def correction():
    # directed-but-locus-free: appropriate for a self-generated error whose locus we do not know
    return (f"Looking back, your earlier {ASK} was wrong. Please recheck the arithmetic and give "
            f"the corrected {ASK} as 'ANSWER: <number>'.")


def classify(ans, truth, own_wrong):
    if ans is None:
        return "abstain"
    if abs(ans - truth) < 0.5:
        return "true"
    if abs(ans - own_wrong) < 0.5:
        return "inherit"          # re-emits its OWN earlier wrong value: the attractor
    return "novel"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--n", type=int, default=8, help="ledger size (larger => more natural errors)")
    ap.add_argument("--stores", type=int, default=48, help="ledgers to try (oversample for errors)")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=0.02)
    ap.add_argument("--price-out", type=float, default=0.05)
    ap.add_argument("--probe", action="store_true", help="4 ledgers only, then cost")
    args = ap.parse_args()

    stores = 4 if args.probe else args.stores
    out = ROOT / "data" / "results" / f"endogenous_{args.model.replace('/', '_')}_n{args.n}.jsonl"
    llm = OpenRouterLLM(model=args.model, temperature=args.temp)

    rows = []
    n_solved = n_endo = 0
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for idx in range(stores):
            led = make_ledger(idx, args.n)
            truth = float(led.problem.correct)
            reply = llm.chat([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": solve_prompt(led)}])
            own = parse_answer(reply)
            n_solved += 1
            if own is None or abs(own - truth) < 0.5:
                continue                       # solved correctly (or unparseable): no endogenous error
            n_endo += 1
            # run the wall on the model's OWN wrong value
            for pol, note in (("lossy_endo", lossy_endo_note(led, own)),
                              ("srcfirst_endo", srcfirst_endo_note(led))):
                r2 = llm.chat([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": note},
                               {"role": "user", "content": correction()}])
                a = parse_answer(r2)
                bucket = classify(a, truth, own)
                row = {"idx": idx, "n": args.n, "policy": pol, "own_wrong": own, "truth": truth,
                       "answer": a, "bucket": bucket}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    dt = time.time() - t0
    print(f"\nendogenous-error wall, {args.model} N={args.n} ({dt:.0f}s, ~${cost:.2f}):")
    print(f"  natural error rate: {n_endo}/{n_solved} ledgers miscomputed on their own")
    if args.probe:
        print(f"  (probe) full run would try {args.stores} ledgers")
        for r in rows:
            print(f"    {r['policy']:>14}: {r['bucket']:>8}  own={r['own_wrong']:g} truth={r['truth']:g} ans={r['answer']}")
        return 0
    print(f"  {'policy':>14}  {'true':>5} {'inherit':>8} {'novel':>6} {'abstain':>8}   n")
    for pol in ("lossy_endo", "srcfirst_endo"):
        sub = [r for r in rows if r["policy"] == pol]
        if not sub:
            continue
        c = Counter(r["bucket"] for r in sub); n = len(sub)
        print(f"  {pol:>14}  {c['true']/n:>5.2f} {c['inherit']/n:>8.2f} {c['novel']/n:>6.2f} "
              f"{c['abstain']/n:>8.2f}   {n}")
    le = [r for r in rows if r["policy"] == "lossy_endo"]
    sf = [r for r in rows if r["policy"] == "srcfirst_endo"]
    if le and sf:
        lt = sum(r["bucket"] == "true" for r in le) / len(le)
        st = sum(r["bucket"] == "true" for r in sf) / len(sf)
        li = sum(r["bucket"] == "inherit" for r in le) / len(le)
        print(f"\n  reclaim (true): lossy_endo {lt:.2f} vs srcfirst_endo {st:.2f}  "
              f"(wall holds if lossy low, srcfirst high)")
        print(f"  lossy_endo re-emits the model's OWN wrong value (attractor) on {li:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
