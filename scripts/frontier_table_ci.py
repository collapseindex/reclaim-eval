#!/usr/bin/env python3
"""Frontier-replay table (tab:frontier) with bootstrap 95% CIs, pure analysis, no API.

Answers the "n=24 / no CI on the deployed board" reviewer objection by attaching a 95%
bootstrap interval and an explicit n to every cell of the deployed/frontier table.

Canonical, uniform n=24 per cell (no double-counting):
  * Llama answering model: the realworld t0.7 runs are themselves the session-2 Llama pass.
    - langchain_summary, mem0, lossy@0.1, source_first@0.1  <- main realworld file
    - source_first_auto, vector_rag                         <- leaderboard file (only there)
    The two hand anchors appear in BOTH files as a re-run of the same 24 (seed,pid)
    conditions; we take the main file's 24 and do NOT union to 48. This makes the Llama
    source-first cell match tab:wall / tab:logic (1.00 arith, 0.67 logic) instead of the
    n=48 union artifact (0.98 / 0.65) the first draft reported.
  * Sonnet / Opus answering models: claude_<model>.jsonl, task-split, n=24.

Usage:
    python scripts/frontier_table_ci.py            # human-readable
    python scripts/frontier_table_ci.py --latex    # LaTeX body rows for tab:frontier
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"

# current problem ids, used to ignore orphaned rows from superseded generated problem sets
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT / "src"))
from reclaim.problems import TASKS as _PROBS  # noqa: E402
PID2PROB = {p.pid for fam in _PROBS.values() for p in fam}

LLAMA_MAIN = "realworld_meta-llama_llama-3.1-8b-instruct_{task}_t0.7.jsonl"
LLAMA_LB = "realworld_meta-llama_llama-3.1-8b-instruct_{task}_t0.7_leaderboard.jsonl"
CLAUDE = "claude_{model}.jsonl"

# Which file is canonical for each variant on the Llama pass (main unless leaderboard-only).
LLAMA_SRC = {
    "langchain_summary": "main",
    "mem0": "main",
    "lossy@0.1": "main",
    "source_first@0.1": "main",
    "source_first_auto": "main",   # n96 crank wrote auto/vector to the main file (matches the
    "vector_rag": "main",          # Sonnet/Opus replay memories); leaderboard file was the old n24
}

# Paper row order and labels.
ROWS = [
    ("source_first@0.1", "\\srcfirst{} (hand)"),
    ("source_first_auto", "\\srcfirst{}-auto (fix)"),
    ("langchain_summary", "LangChain summary"),
    ("mem0", "\\texttt{mem0}"),
    ("vector_rag", "naive vector retrieval"),
    ("lossy@0.1", "\\lossy{} (hand)"),
]
TASKS = ("arith", "logic")
ARM = "directed"


def _read(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
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
    means = sorted(sum(xs[r.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(xs) / k, means[int(0.025 * n)], means[int(0.975 * n)]


def _b(x):
    """Abbreviate a CI bound to match tab:wall style: drop leading 0, render 1.0 as 1."""
    if x >= 0.9995:
        return "1"
    if x <= 0.0005:
        return "0"
    return f"{x:.2f}".lstrip("0")


def llama_cells(task):
    main = _read(RES / LLAMA_MAIN.format(task=task))
    lb = _read(RES / LLAMA_LB.format(task=task))
    by = {"main": main, "lb": lb}
    cells = {}
    for var, _ in ROWS:
        src = by[LLAMA_SRC[var]]
        cells[var] = [1 if r["correct"] else 0 for r in src
                      if r["variant"] == var and r["arm"] == ARM and r.get("pid") in PID2PROB]
    return cells


def claude_cells(model, task):
    rows = _read(RES / CLAUDE.format(model=model))
    cells = defaultdict(list)
    for r in rows:
        if r["arm"] == ARM and r.get("task") == task and r.get("pid") in PID2PROB:
            cells[r["variant"]].append(1 if r["correct"] else 0)
    return cells


def collect(task):
    return {
        "Llama": llama_cells(task),
        "Sonnet": claude_cells("claude-sonnet-4-6", task),
        "Opus": claude_cells("claude-opus-4-8", task),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    data = {t: collect(t) for t in TASKS}

    # Cells to bold: the Opus column extremes that carry the headline
    # (source-kept -> 1.00, source-dropped -> 0.00).
    BOLD = {("source_first@0.1", "arith", "Opus"), ("source_first@0.1", "logic", "Opus"),
            ("lossy@0.1", "arith", "Opus"), ("lossy@0.1", "logic", "Opus")}

    if args.latex:
        for var, label in ROWS:
            parts = [label]
            for task in TASKS:
                for model in ("Llama", "Sonnet", "Opus"):
                    m, lo, hi = boot_ci(data[task][model].get(var, []))
                    body = f"{m:.2f}_{{[{_b(lo)},{_b(hi)}]}}"
                    if (var, task, model) in BOLD:
                        body = "\\mathbf{" + body + "}"
                    parts.append(f"${body}$")
                if task == "arith":
                    parts.append("")  # spacer column between tasks
            print(" & ".join(parts) + " \\\\")
        return 0

    for task in TASKS:
        print(f"\n=== {task} directed RR, bootstrap 95% CI (n per cell) ===")
        hdr = f"  {'memory':>24}" + "".join(f"{m:>20}" for m in ("Llama", "Sonnet", "Opus"))
        print(hdr)
        for var, label in ROWS:
            line = f"  {var:>24}"
            for model in ("Llama", "Sonnet", "Opus"):
                xs = data[task][model].get(var, [])
                m, lo, hi = boot_ci(xs)
                line += f"  {m:.2f}[{lo:.2f},{hi:.2f}]n{len(xs):>2}".rjust(20)
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
