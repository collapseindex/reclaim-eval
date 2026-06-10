#!/usr/bin/env python3
"""Reclaim-window pilot. Dry-run validates the pipeline for free; --real spends API.

    python scripts/run_pilot.py --dry-run          # zero cost, fake LLM with a window
    python scripts/run_pilot.py --real --n 3       # small paid pilot (prints call count)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reclaim.problems import PROBLEMS
from reclaim.experiment import DEPTHS, run_problem
from reclaim.llm import OpenRouterLLM, DryRunLLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--n", type=int, default=len(PROBLEMS))
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    if not (args.dry_run or args.real):
        ap.error("pass --dry-run or --real")

    problems = PROBLEMS[: args.n]
    # success[arm][depth] -> [bools]
    succ = {a: defaultdict(list) for a in ("generic", "directed")}
    total_calls = 0
    for s in range(args.seeds):
        llm = DryRunLLM(seed=s) if args.dry_run else OpenRouterLLM(model=args.model)
        for p in problems:
            for row in run_problem(llm, p):
                succ[row["arm"]][row["depth"]].append(row["correct"])
        total_calls += getattr(llm, "calls", 0)

    print(f"\n{'mode':<8} {args.model if args.real else 'dry-run'}   "
          f"problems={len(problems)} seeds={args.seeds}  api_calls={total_calls}\n")
    print("reclaim success rate vs drift depth (the window):")
    print(f"  {'depth':>6} {'generic':>9} {'directed':>9}")
    for d in DEPTHS:
        g = succ["generic"][d]
        di = succ["directed"][d]
        gm = (sum(g) / len(g)) if g else float("nan")
        dm = (sum(di) / len(di)) if di else float("nan")
        print(f"  {d:>6} {gm:>9.2f} {dm:>9.2f}")

    # window summary: where each arm drops below 0.5
    def edge(arm):
        for d in DEPTHS:
            v = succ[arm][d]
            if v and (sum(v) / len(v)) < 0.5:
                return d
        return None
    ge, de = edge("generic"), edge("directed")
    print(f"\n  generic window closes at depth: {ge if ge else '> max'}")
    print(f"  directed window closes at depth: {de if de else '> max'}")
    print("\n  (a window = directed holding deeper than generic; null = they match"
          " or neither closes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
