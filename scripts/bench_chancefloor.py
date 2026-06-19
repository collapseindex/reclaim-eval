#!/usr/bin/env python3
"""Chance-floor control for the logic soft wall.

The logic tasks have single-token answers over small candidate sets (3-5 options), so part of
the soft-wall "recovery" could be free guessing rather than re-derivation from the surviving
clue. This measures the empirical free-guess rate: a carried note that gives ONLY the candidate
set (no clues, no premise, no conclusion), then the same correction. The generic-arm number is
the conservative chance floor; the directed arm adds only the locus-naming signal (no clue), so
(directed soft-wall with clue) - (directed blank) isolates the clue's genuine contribution.

    python scripts/bench_chancefloor.py --model llama
    python scripts/bench_chancefloor.py --model claude-opus-4-8
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

from reclaim.problems import PROBLEMS_LOGIC
from reclaim.experiment import SYSTEM, reclaim_cross, score, _logged_answer
from reclaim.llm import OpenRouterLLM, AnthropicLLM

# candidate sets per logic problem: what the answer is freely drawn from (no constraints)
CANDIDATES = {
    "roles":   ["Ana", "Ben", "Cleo"],
    "seating": ["Dee", "Eve", "Fia", "Gus"],
    "race":    ["Hal", "Ira", "Jo", "Kit", "Lee"],
    "ages":    ["Mae", "Ned", "Ola"],
    "pets":    ["Pam", "Quincy", "Rosa"],
    "days":    ["Monday", "Tuesday", "Wednesday"],
    "height":  ["Sam", "Tom", "Uma", "Val"],
    "houses":  ["red", "blue", "green"],
}


def blank_note(prob) -> str:
    opts = ", ".join(CANDIDATES[prob.pid])
    return (f"(Memory of an earlier session.) You were determining {prob.ask}; the only "
            f"options were: {opts}. No other details were preserved.")


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)   # any OpenRouter id, e.g. grok


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
    out_path = ROOT / "data" / "results" / f"chancefloor_{args.model.replace('/', '_')}.jsonl"
    rows = []
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for prob in PROBLEMS_LOGIC:
            note = blank_note(prob)
            for arm in ("generic", "directed"):
                for seed in range(args.seeds):
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note},
                            {"role": "user", "content": reclaim_cross(prob, arm)}]
                    reply = llm.chat(msgs)
                    row = {"pid": prob.pid, "n_options": len(CANDIDATES[prob.pid]), "arm": arm,
                           "seed": seed, "model": args.model,
                           "answer": _logged_answer(reply, prob), "correct": score(reply, prob)}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)

    agg = defaultdict(list)
    for r in rows:
        agg[r["arm"]].append(1 if r["correct"] else 0)
    analytic = sum(1 / len(CANDIDATES[p.pid]) for p in PROBLEMS_LOGIC) / len(PROBLEMS_LOGIC)
    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    print(f"\nChance floor, {args.model}, logic blank note (n={len(rows)//2}/arm):")
    for arm in ("generic", "directed"):
        xs = agg[arm]
        print(f"  {arm:8} free-guess RR = {sum(xs)/len(xs):.2f}  (n={len(xs)})")
    print(f"  analytic uniform chance ~ {analytic:.2f}")
    if args.model.startswith("claude"):
        print(f"  cost ~${cost:.2f}")
    print(f"  ({time.time()-t0:.0f}s)  log -> {out_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
