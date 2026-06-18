#!/usr/bin/env python3
"""Objective confabulation audit: how often a memory system INVENTS a fact during compression.

Pure analysis, no API, no LLM judge (keeping the paper's objective-scoring stance). A memory
``confabulates'' when its carried text states a number that never appeared in session 1: it
was not recalled, it was manufactured. mem0's ``the number of pens is 13'' is the canonical
case, a value invented to make the wrong premise self-consistent. Verbatim-retrieval memory
(vector_rag) cannot confabulate by construction; LLM-rewriting memory can.

    python scripts/confab_audit.py data/results/realworld_*leaderboard*.jsonl

Reports, per system: the share of memories that invent >=1 number, and the mean count of
invented numbers per memory (the noise the source must survive).
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict


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


def nums(text):
    import re
    return {m.group() for m in re.finditer(r"\d+(?:\.\d+)?", text or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--examples", type=int, default=2,
                    help="show this many worst (most-invented) memories per system")
    args = ap.parse_args()

    rows = load(args.paths)
    # one memory per (seed, pid, variant): both arms share memory_text, so dedup
    seen = {}
    for r in rows:
        if "src_nums" not in r:
            continue
        key = (r["seed"], r["pid"], r["variant"])
        if key not in seen:
            seen[key] = r

    if not seen:
        print("No rows with src_nums (re-run the benchmark after the confab patch).")
        return 1

    by_var = defaultdict(list)
    for (seed, pid, v), r in seen.items():
        invented = sorted(nums(r["memory_text"]) - set(r["src_nums"]),
                          key=lambda x: float(x))
        by_var[v].append((len(invented), invented, pid, r["memory_text"]))

    print(f"\nConfabulation audit ({len(seen)} memories): numbers invented during "
          f"compression (not present in session 1)\n")
    print(f"  {'memory system':>22} {'n':>4} {'%>=1 invented':>14} {'mean invented':>14}")
    order = sorted(by_var, key=lambda v: sum(c for c, *_ in by_var[v]) / len(by_var[v]))
    for v in order:
        items = by_var[v]
        share = sum(1 for c, *_ in items if c) / len(items)
        mean = sum(c for c, *_ in items) / len(items)
        print(f"  {v:>22} {len(items):>4} {share:>13.0%} {mean:>14.1f}")

    if args.examples:
        print("\nworst examples (most invented numbers):")
        for v in order:
            worst = sorted(by_var[v], reverse=True)[: args.examples]
            for cnt, inv, pid, mem in worst:
                if not cnt:
                    continue
                print(f"  [{v} / {pid}] invented {cnt}: {inv}")
                print(f"      {mem[:240]}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
