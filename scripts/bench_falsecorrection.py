#!/usr/bin/env python3
"""Does source-first make the model act on a FALSE correction, or reject it?

The symmetric question to reclaim. A directed correction normally names the true error locus.
Here it names a \emph{correct} component as the error ("the notebooks subtotal is wrong" when it
is fine). With the source present (source-first), a robust model recomputes, finds nothing wrong,
and returns the true total; a misled model "fixes" the correctly-named component and returns a
wrong one. lossy has no source, so it can act on neither a true nor a false correction.

    python scripts/bench_falsecorrection.py --model llama
    python scripts/bench_falsecorrection.py --model x-ai/grok-4.3-20260430
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
from reclaim.experiment import SYSTEM, memory_note, score, _logged_answer, _configure
from reclaim.llm import OpenRouterLLM, AnthropicLLM, parse_answer

# the non-error (correct) component of each arithmetic problem: a false locus to name
FALSE_LOCUS = {
    "notebooks": "the notebooks subtotal",
    "tiles": "the grout bags subtotal",
    "fruit": "the apples subtotal",
    "tickets": "the adult-ticket subtotal",
    "paint": "the paint cans subtotal",
    "coffee": "the filter subtotal",
    "lumber": "the screws subtotal",
    "flowers": "the lilies subtotal",
}
G = 0.1   # low integrity: lossy = conclusion only; source-first = clean source, no conclusion


def reclaim_msg(prob, locus):
    return (f"Looking back at that earlier conclusion, I think {locus} was wrong. Please recheck "
            f"that and give the corrected {prob.ask} as 'ANSWER: <answer>'.")


def classify(reply, prob):
    a = parse_answer(reply)
    if a is None:
        return "noans"
    if abs(a - float(prob.correct)) < 0.5:
        return "true"          # recomputed the truth (robust under false correction)
    if abs(a - float(prob.drift)) < 0.5:
        return "drift"         # inherited the planted wrong value
    return "other"             # a different wrong value (misled / miscomputed)


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=15.0)
    ap.add_argument("--price-out", type=float, default=75.0)
    args = ap.parse_args()

    temp = 0.0 if args.model.startswith("claude") else args.temp
    llm = make_llm(args.model, temp)
    out = ROOT / "data" / "results" / f"falsecorr_{args.model.replace('/', '_')}.jsonl"
    rows = []
    conds = [("source_first", "true"), ("source_first", "false"),
             ("lossy", "true"), ("lossy", "false")]
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for prob in PROBLEMS:
            note = {p: memory_note(prob, G, p) for p in ("source_first", "lossy")}
            for pol, corr in conds:
                locus = prob.locus if corr == "true" else FALSE_LOCUS[prob.pid]
                for seed in range(args.seeds):
                    _configure(llm, prob)
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note[pol]},
                            {"role": "user", "content": reclaim_msg(prob, locus)}]
                    reply = llm.chat(msgs)
                    b = classify(reply, prob)
                    row = {"pid": prob.pid, "policy": pol, "correction": corr, "seed": seed,
                           "bucket": b, "answer": _logged_answer(reply, prob)}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    print(f"\nFalse-correction robustness, {args.model} ({time.time()-t0:.0f}s):")
    print(f"  {'policy':>13} {'correction':>10} {'true':>5} {'drift':>6} {'other':>6} {'noans':>6}")
    for pol, corr in conds:
        sub = [r for r in rows if r["policy"] == pol and r["correction"] == corr]
        c = Counter(r["bucket"] for r in sub)
        n = len(sub)
        print(f"  {pol:>13} {corr:>10} {c['true']/n:>5.2f} {c['drift']/n:>6.2f} "
              f"{c['other']/n:>6.2f} {c['noans']/n:>6.2f}")
    sf_false = [r for r in rows if r["policy"] == "source_first" and r["correction"] == "false"]
    rob = sum(1 for r in sf_false if r["bucket"] == "true") / len(sf_false)
    print(f"\n  source-first ROBUSTNESS to a false correction (returns truth): {rob:.2f} "
          f"(n={len(sf_false)})")
    if args.model.startswith("claude"):
        print(f"  cost ~${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
