#!/usr/bin/env python3
"""Decay curve for the source-size sweep: where the source-first law ends.

Pure analysis (+ optional figure), no API. Aggregates directed Reclaim Rate by
(budget, N, policy) with bootstrap 95% CIs, locates the crossover where source-first falls to
the lossy floor, and checks the mechanism: reclaim tracks whether the answer-determining source
survived the budget, not N per se.

    python scripts/analyze_sizesweep.py data/results/sizesweep_llama.jsonl
    python scripts/analyze_sizesweep.py --fig figures/sizesweep.pdf <files>
    python scripts/analyze_sizesweep.py --latex <files>
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path


def load(paths):
    rows = []
    for pat in paths:
        for f in glob.glob(pat):
            for line in open(f, encoding="utf-8"):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def boot_ci(xs, n=5000, seed=0):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    r = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[r.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(xs) / k, means[int(0.025 * n)], means[int(0.975 * n)]


def aggregate(rows, arm="directed"):
    """(budget, policy, N) -> list[0/1]."""
    cell = defaultdict(list)
    for r in rows:
        if r["arm"] != arm:
            continue
        cell[(r["budget"], r["policy"], r["n"])].append(1 if r["correct"] else 0)
    return cell


def crossover(cell, budget, ns, thresh=0.5):
    """Largest N at which source_first is still above thresh (the boundary of the law)."""
    above = [n for n in ns if cell.get((budget, "source_first", n))
             and sum(cell[(budget, "source_first", n)]) / len(cell[(budget, "source_first", n)]) > thresh]
    return max(above) if above else None


def mechanism(rows, arm="directed"):
    """source_first reclaim split by whether the FULL source survived the budget (k==N).
    A sum needs every item, so full-source is the right variable: partial source (k<N) is
    the hard information wall; within full-source, residual variation is capability."""
    full, partial = [], []
    for r in rows:
        if r["arm"] != arm or r["policy"] != "source_first":
            continue
        (full if r["k_items"] == r["n"] else partial).append(1 if r["correct"] else 0)
    return full, partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fig", help="write a decay-curve figure to this path")
    ap.add_argument("--overlay", help="second model's jsonl, plotted as a confirm overlay")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows = load(args.paths)
    cell = aggregate(rows)
    budgets = sorted({b for b, _, _ in cell})
    ns = sorted({n for _, _, n in cell})

    if args.latex:
        for b in budgets:
            print(f"\\multicolumn{{2}}{{l}}{{\\textit{{budget $B={b}$}}}} \\\\")
            for pol, lab in (("source_first", "\\srcfirst{}"), ("lossy_padded", "\\padded{}")):
                out = [lab]
                for n in ns:
                    xs = cell.get((b, pol, n), [])
                    m, lo, hi = boot_ci(xs)
                    out.append(f"${m:.2f}$" if xs else "--")
                print(" & ".join(out) + " \\\\")
        return 0

    print(f"\nDirected Reclaim Rate by budget x N  ({len(rows)} rows)\n")
    for b in budgets:
        print(f"=== budget B={b} ===")
        print(f"  {'N':>4}  {'source_first [95% CI]':>26}  {'lossy_padded [95% CI]':>26}  {'k items':>8}")
        for n in ns:
            sf = cell.get((b, "source_first", n), [])
            lp = cell.get((b, "lossy_padded", n), [])
            ks = [r["k_items"] for r in rows if r["budget"] == b
                  and r["policy"] == "source_first" and r["n"] == n]
            kmed = sorted(ks)[len(ks) // 2] if ks else 0
            m1, lo1, hi1 = boot_ci(sf)
            m2, lo2, hi2 = boot_ci(lp)
            print(f"  {n:>4}  {m1:>6.2f} [{lo1:.2f},{hi1:.2f}] n{len(sf):<3}"
                  f"  {m2:>6.2f} [{lo2:.2f},{hi2:.2f}] n{len(lp):<3}  {kmed:>3}/{n}")
        xo = crossover(cell, b, ns)
        print(f"  crossover (source_first > 0.5 up to): N={xo}\n")

    full, partial = mechanism(rows)
    mf, lof, hif = boot_ci(full)
    mp, lop, hip = boot_ci(partial)
    print("Mechanism: source_first reclaim, split by whether the FULL source fit the budget")
    print(f"  full source (k=N)    : {mf:.2f} [{lof:.2f},{hif:.2f}]  n={len(full)}")
    print(f"  partial source (k<N) : {mp:.2f} [{lop:.2f},{hip:.2f}]  n={len(partial)}")
    print("  (k<N is the hard information wall; full-source variation is capability)\n")

    if args.fig:
        overlay = aggregate(load([args.overlay])) if args.overlay else None
        make_fig(cell, budgets, ns, args.fig, overlay)
        print(f"figure -> {args.fig}")
    return 0


def make_fig(cell, budgets, ns, path, overlay=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    colors = {budgets[0]: "#1f77b4", budgets[-1]: "#d62728"}
    for b in budgets:
        c = colors.get(b, "#555555")
        xs_sf, ys_sf, lo_sf, hi_sf = [], [], [], []
        xs_lp, ys_lp = [], []
        for n in ns:
            sf = cell.get((b, "source_first", n), [])
            if sf:
                m, lo, hi = boot_ci(sf)
                xs_sf.append(n); ys_sf.append(m); lo_sf.append(m - lo); hi_sf.append(hi - m)
            lp = cell.get((b, "lossy_padded", n), [])
            if lp:
                xs_lp.append(n); ys_lp.append(sum(lp) / len(lp))
        ax.errorbar(xs_sf, ys_sf, yerr=[lo_sf, hi_sf], color=c, marker="o", ms=4,
                    lw=1.8, capsize=2, label=f"source-first, B={b}")
        ax.plot(xs_lp, ys_lp, color=c, marker="x", ms=4, lw=1.0, ls="--", alpha=0.6,
                label=f"lossy (budget-matched), B={b}")
        if overlay is not None:
            xo, yo = [], []
            for n in ns:
                ov = overlay.get((b, "source_first", n), [])
                if ov:
                    xo.append(n); yo.append(sum(ov) / len(ov))
            if xo:
                ax.plot(xo, yo, color=c, marker="s", ms=4, lw=1.0, ls=":", alpha=0.9,
                        label=f"source-first, B={b} (Opus confirm)")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns], fontsize=7)
    ax.set_xlabel("source size $N$ (line items)")
    ax.set_ylabel("directed Reclaim Rate")
    ax.set_ylim(-0.03, 1.05)
    ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(0.0, 0.42))
    ax.set_title("Source-first leads until the source outgrows the budget", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
