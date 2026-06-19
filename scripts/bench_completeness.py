#!/usr/bin/env python3
"""Completeness-signal probe: does telling the model the source is truncated stop the silent
'sum the partial source' failure?

At the cliff (k<N) plain source-first fails silently: the model confidently sums the k visible
items and asserts the result, never flagging the missing item(s). This re-runs the same cliff
cells with an identical note plus an explicit 'preserved only k of N items' marker, and records
the raw reply so we can classify abstain vs. confident mis-sum.

    python scripts/bench_completeness.py --model claude-opus-4-8
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.sizesweep import make_ledger, build_note
from reclaim.experiment import SYSTEM, reclaim_cross, score, _logged_answer
from reclaim.llm import OpenRouterLLM, AnthropicLLM, parse_answer

N_STORES = 8
CLIFF_CELLS = [(300, 6), (300, 8), (600, 16), (600, 20)]   # k<N on all of these
ABSTAIN_RE = re.compile(
    r"(can(?:not|'t)|unable|insufficient|not enough|missing|incomplete|truncat|dropped|"
    r"not shown|isn't shown|of the original|were dropped|was dropped|caveat|caution|"
    r"can't be certain|cannot be certain|would need|without the|do(?:n't| not) have|"
    r"unknown|impossible to)", re.I)


def classify(reply, ans, true_total, sum_visible):
    """Behavioral bucket for one reply at the cliff."""
    if ans is not None and abs(ans - true_total) < 0.5:
        return "correct"
    flagged = bool(ABSTAIN_RE.search(reply))
    summed_vis = ans is not None and abs(ans - sum_visible) < 0.5
    if ans is None and flagged:
        return "abstain"
    if summed_vis and flagged:
        return "flagged_partial"     # gave the partial sum but warned it is incomplete
    if summed_vis:
        return "silent_missum"       # confident partial sum, no warning (the silent failure)
    if flagged:
        return "abstain"
    return "other_wrong"


def make_llm(model):
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=0.0,
                             max_tokens=1000)
    return AnthropicLLM(model=model, temperature=0.0, max_tokens=1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--price-in", type=float, default=15.0)
    ap.add_argument("--price-out", type=float, default=75.0)
    args = ap.parse_args()

    out_path = ROOT / "data" / "results" / f"completeness_{args.model.replace('/', '_')}.jsonl"
    llm = make_llm(args.model)
    t0 = time.time()
    rows = []
    with open(out_path, "w", encoding="utf-8") as out:
        for b, n in CLIFF_CELLS:
            for s in range(N_STORES):
                led = make_ledger(s, n)
                note, k, _ = build_note(led, b, "source_first_complete")
                true_total = float(led.problem.correct)
                sum_visible = sum(p * q for _, p, q in led.items[:k])
                for seed in range(args.seeds):
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note},
                            {"role": "user", "content": reclaim_cross(led.problem, "directed")}]
                    reply = llm.chat(msgs)
                    ans = parse_answer(reply)
                    bucket = classify(reply, ans, true_total, sum_visible)
                    row = {"budget": b, "n": n, "k": k, "store": s, "seed": seed,
                           "answer": ans, "bucket": bucket, "correct": score(reply, led.problem),
                           "reply": reply}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)
            cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
            print(f"  B{b} N{n} done  (~${cost:.2f}, {time.time()-t0:.0f}s)")

    # summary
    from collections import Counter
    print(f"\nCompleteness-signal probe, {args.model}, cliff cells, n={len(rows)}:")
    for b, n in CLIFF_CELLS:
        c = Counter(r["bucket"] for r in rows if r["budget"] == b and r["n"] == n)
        print(f"  B{b} N{n}: {dict(c)}")
    overall = Counter(r["bucket"] for r in rows)
    print(f"\n  overall: {dict(overall)}")
    print(f"  -> abstain or flagged: "
          f"{sum(overall[k] for k in ('abstain','flagged_partial'))}/{len(rows)}; "
          f"silent mis-sum: {overall['silent_missum']}/{len(rows)}")
    print(f"  log -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
