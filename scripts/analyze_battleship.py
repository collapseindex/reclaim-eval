#!/usr/bin/env python3
"""Aggregate the Battleship behavioral sweep into the showcase table + a JSON for the website.

Structural metrics only (no prose-regex): re-fire rate (the brittle-memory signal), abstain rate
(the 'does not abstain' finding), and coverage/hits per game. Reads every
data/results/battleship_behavior_*.jsonl produced by bench_battleship_behavior.py.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"
ORDER = ["claude-opus-4-8", "claude-sonnet-4-6", "llama"]
PRETTY = {"claude-opus-4-8": "Opus 4.8", "claude-sonnet-4-6": "Sonnet 4.6", "llama": "Llama-3.1-8B"}


def load():
    rows = []
    for p in RESULTS.glob("battleship_behavior_*.jsonl"):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    rows = load()
    if not rows:
        print("no battleship_behavior_*.jsonl found"); return 1
    cell = defaultdict(list)
    boards = defaultdict(set)
    for r in rows:
        cell[(r["model"], r["policy"])].append(r)
        boards[(r["model"], r["policy"])].add(r["board"])

    models = [m for m in ORDER if any(k[0] == m for k in cell)]
    summary = {}
    print(f"\n{'model':>13}  {'policy':>12}  {'turns':>5}  {'re-fire':>8}  {'abstain':>8}  "
          f"{'distinct/game':>13}  {'hits/game':>9}")
    for m in models:
        for pol in ("source_first", "lossy", "lossy_padded"):
            xs = cell.get((m, pol))
            if not xs:
                continue
            n = len(xs)
            nb = len(boards[(m, pol)])
            refire = sum(r["bucket"] == "refire" for r in xs)
            abstain = sum(r["bucket"] == "invalid" for r in xs)
            distinct = sum(r["bucket"] == "new" for r in xs)
            hits = sum(r["result"] in ("HIT", "SUNK") for r in xs)
            rec = {"turns": n, "boards": nb, "refire_rate": refire / n, "abstain_rate": abstain / n,
                   "distinct_per_game": distinct / nb, "hits_per_game": hits / nb}
            summary[f"{m}/{pol}"] = rec
            print(f"{PRETTY.get(m,m):>13}  {pol:>12}  {n:>5}  {refire/n:>7.0%}  {abstain/n:>7.0%}  "
                  f"{distinct/nb:>13.1f}  {hits/nb:>9.1f}")

    out = RESULTS / "battleship_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nheadline — re-fire rate (source_first -> lossy), capability-invariant:")
    for m in models:
        sf = summary.get(f"{m}/source_first", {}).get("refire_rate")
        lo = summary.get(f"{m}/lossy", {}).get("refire_rate")
        if sf is not None and lo is not None:
            print(f"  {PRETTY.get(m,m):>13}:  {sf:.0%}  ->  {lo:.0%}")
    print(f"\nabstain rate at the wall (lossy) — the 'does not abstain' finding:")
    for m in models:
        lo = summary.get(f"{m}/lossy", {}).get("abstain_rate")
        if lo is not None:
            print(f"  {PRETTY.get(m,m):>13}:  {lo:.0%}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
