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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.problems import PROBLEMS
from reclaim.experiment import DEPTHS, DISTANCES, run_problem, run_problem_distance
from reclaim.llm import OpenRouterLLM, DryRunLLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--n", type=int, default=len(PROBLEMS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--degrade", action="store_true",
                    help="vary channel distance (filler) instead of commitment depth")
    args = ap.parse_args()
    if not (args.dry_run or args.real):
        ap.error("pass --dry-run or --real")

    problems = PROBLEMS[: args.n]
    axis = "distance" if args.degrade else "depth"
    levels = DISTANCES if args.degrade else DEPTHS
    succ = {a: defaultdict(list) for a in ("generic", "directed")}
    total_calls = 0
    for s in range(args.seeds):
        llm = (DryRunLLM(seed=s) if args.dry_run
               else OpenRouterLLM(model=args.model, temperature=args.temp))
        for p in problems:
            rows = run_problem_distance(llm, p) if args.degrade else run_problem(llm, p)
            for row in rows:
                succ[row["arm"]][row[axis]].append(row["correct"])
        total_calls += getattr(llm, "calls", 0)

    print(f"\n{'mode':<8} {args.model if args.real else 'dry-run'}   "
          f"problems={len(problems)} seeds={args.seeds}  api_calls={total_calls}\n")
    label = ("reclaim success vs CHANNEL DISTANCE (filler turns; the sky diluting)"
             if args.degrade else "reclaim success vs drift depth (the window)")
    print(label + ":")
    print(f"  {axis:>9} {'generic':>9} {'directed':>9}")
    for d in levels:
        g, di = succ["generic"][d], succ["directed"][d]
        gm = (sum(g) / len(g)) if g else float("nan")
        dm = (sum(di) / len(di)) if di else float("nan")
        print(f"  {d:>9} {gm:>9.2f} {dm:>9.2f}")

    def edge(arm):
        for d in levels:
            v = succ[arm][d]
            if v and (sum(v) / len(v)) < 0.5:
                return d
        return None
    ge, de = edge("generic"), edge("directed")
    print(f"\n  generic falls below 0.5 at {axis}: {ge if ge is not None else '> max'}")
    print(f"  directed falls below 0.5 at {axis}: {de if de is not None else '> max'}")
    print("\n  (signal = directed holding past generic; null = they match or"
          " neither falls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
