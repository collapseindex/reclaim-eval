#!/usr/bin/env python3
"""Noisy-source analysis: keeping the source is not enough; you must keep the RIGHT source.

Aggregates directed Reclaim Rate by (policy, decoy count) with bootstrap 95% CIs, and checks
the mechanism: naive source-first reclaims iff every bought item survived the budget, so noise
that crowds a bought item out is the same hard information wall as dropping it.

    python scripts/analyze_noisysweep.py data/results/noisysweep_llama.jsonl
    python scripts/analyze_noisysweep.py --fig figures/noisysweep.pdf \
        --overlay data/results/noisysweep_claude-opus-4-8.jsonl <llama.jsonl>
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

POLICIES = ("source_first_naive", "source_first_denoised", "lossy_padded")
LAB = {"source_first_naive": "source-first (naive)",
       "source_first_denoised": "source-first (denoised)",
       "lossy_padded": "lossy (budget-matched)"}


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
    cell = defaultdict(list)
    for r in rows:
        if r["arm"] == arm:
            cell[(r["policy"], r["decoys"])].append(1 if r["correct"] else 0)
    return cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--fig")
    ap.add_argument("--overlay", help="frontier model's noisysweep jsonl, naive overlay")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    rows = load(args.paths)
    cell = aggregate(rows)
    decoys = sorted({d for _, d in cell})

    if args.latex:
        for pol in POLICIES:
            out = [LAB[pol]]
            for d in decoys:
                m, lo, hi = boot_ci(cell.get((pol, d), []))
                out.append(f"${m:.2f}$" if cell.get((pol, d)) else "--")
            print(" & ".join(out) + " \\\\")
        return 0

    print(f"\nDirected Reclaim Rate by policy x decoy count  ({len(rows)} rows)\n")
    print(f"  {'policy':>26}  " + "  ".join(f"d{d:>2}" for d in decoys))
    for pol in POLICIES:
        line = f"  {pol:>26}  "
        line += "  ".join(f"{boot_ci(cell.get((pol, d), []))[0]:.2f}" for d in decoys)
        print(line)

    # mechanism: naive reclaim split by whether ALL bought items survived the budget
    full, partial = [], []
    for r in rows:
        if r["arm"] == "directed" and r["policy"] == "source_first_naive":
            (full if r["all_kept"] else partial).append(1 if r["correct"] else 0)
    mf, lof, hif = boot_ci(full)
    mp, lop, hip = boot_ci(partial)
    print("\nMechanism: naive source-first, split by whether ALL bought items fit the budget")
    print(f"  all bought kept    : {mf:.2f} [{lof:.2f},{hif:.2f}]  n={len(full)}")
    print(f"  a bought item lost : {mp:.2f} [{lop:.2f},{hip:.2f}]  n={len(partial)}")
    print("  (noise crowding a bought item out is the same information wall as dropping it)\n")

    if args.fig:
        overlay = aggregate(load([args.overlay])) if args.overlay else None
        make_fig(cell, decoys, args.fig, overlay)
        print(f"figure -> {args.fig}")
    return 0


def make_fig(cell, decoys, path, overlay=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    style = {"source_first_naive": ("#d62728", "o", "-"),
             "source_first_denoised": ("#2ca02c", "s", "-"),
             "lossy_padded": ("#777777", "x", "--")}
    for pol in POLICIES:
        c, mk, ls = style[pol]
        xs, ys, lo, hi = [], [], [], []
        for d in decoys:
            v = cell.get((pol, d), [])
            if v:
                m, l, h = boot_ci(v)
                xs.append(d); ys.append(m); lo.append(m - l); hi.append(h - m)
        ax.errorbar(xs, ys, yerr=[lo, hi], color=c, marker=mk, ms=4, lw=1.6,
                    ls=ls, capsize=2, label=LAB[pol])
    if overlay is not None:
        xo, yo = [], []
        for d in decoys:
            v = overlay.get(("source_first_naive", d), [])
            if v:
                xo.append(d); yo.append(sum(v) / len(v))
        if xo:
            ax.plot(xo, yo, color="#d62728", marker="D", ms=4, lw=1.0, ls=":",
                    alpha=0.9, label="source-first (naive), frontier confirm")
    ax.set_xlabel("decoy (noise) items added to a 4-item source")
    ax.set_ylabel("directed Reclaim Rate")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=7, loc="center right")
    ax.set_title("Noise crowds the source out of a fixed budget", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
