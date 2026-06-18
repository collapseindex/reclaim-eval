#!/usr/bin/env python3
"""Reclaim Rate with bootstrap confidence intervals, from realworld checkpoints.

Pure analysis, no API. Hardens the benchmark for a venue: every cell gets a 95% CI and an
n, so the n=24 / no-CI reviewer objection is answered.

    python scripts/analyze_realworld.py data/results/realworld_*arith*leaderboard*.jsonl
    python scripts/analyze_realworld.py --latex <files>      # emit LaTeX table rows
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

ARMS = ("generic", "directed")


def load(paths):
    rows = []
    for pat in paths:
        for f in glob.glob(pat):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def boot_ci(xs, n=5000, seed=0):
    """Mean and percentile bootstrap 95% CI of a 0/1 list."""
    if not xs:
        return float("nan"), float("nan"), float("nan")
    r = random.Random(seed)
    k = len(xs)
    means = []
    for _ in range(n):
        means.append(sum(xs[r.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return sum(xs) / k, means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--latex", action="store_true", help="emit LaTeX table rows")
    args = ap.parse_args()

    rows = load(args.paths)
    cells = defaultdict(list)
    order = []
    for r in rows:
        v, a = r["variant"], r["arm"]
        if v not in order:
            order.append(v)
        cells[(v, a)].append(1 if r["correct"] else 0)

    if args.latex:
        for v in order:
            out = [v.replace("_", "\\_")]
            for a in ARMS:
                m, lo, hi = boot_ci(cells[(v, a)])
                out.append(f"{m:.2f}\\,[{lo:.2f},{hi:.2f}]")
            print(" & ".join(out) + " \\\\")
        return 0

    print(f"\nReclaim Rate with bootstrap 95% CI  ({len(rows)} rows)\n")
    print(f"  {'memory system':>22} {'n':>4}   {'generic [95% CI]':>22}   "
          f"{'directed [95% CI]':>22}")
    for v in order:
        line = f"  {v:>22}"
        n = len(cells[(v, ARMS[0])])
        line += f" {n:>4}  "
        for a in ARMS:
            m, lo, hi = boot_ci(cells[(v, a)])
            line += f"  {m:.2f} [{lo:.2f},{hi:.2f}]".rjust(23)
        print(line)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
