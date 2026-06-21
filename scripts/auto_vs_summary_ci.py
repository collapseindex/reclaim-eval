#!/usr/bin/env python3
"""Paired \\srcfirst-auto minus LangChain-summary difference with a paired bootstrap 95% CI.

Pure analysis, no API. Reads the same stored frontier files frontier_table_ci.py uses, pairs
rows by (seed, pid) within each (model, task) cell, and bootstraps the mean paired difference.
Also prints the auto-over-{mem0,vector} gaps and the auto RR range for the prose.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
import sys as _sys
_sys.path.insert(0, str(ROOT / "src"))
from reclaim.problems import TASKS as _PROBS  # noqa: E402
PID2PROB = {p.pid for fam in _PROBS.values() for p in fam}

LLAMA_MAIN = "realworld_meta-llama_llama-3.1-8b-instruct_{task}_t0.7.jsonl"
CLAUDE = "claude_{model}.jsonl"
ARM = "directed"


def _read(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def keyed(rows, variant):
    """{(seed,pid): correct} for one variant, current problems only."""
    out = {}
    for r in rows:
        if (r["variant"] == variant and r["arm"] == ARM and r.get("pid") in PID2PROB):
            out[(r["seed"], r["pid"])] = 1 if r["correct"] else 0
    return out


def rows_for(model, task):
    if model == "Llama":
        return _read(RES / LLAMA_MAIN.format(task=task))
    fname = {"Sonnet": "claude-sonnet-4-6", "Opus": "claude-opus-4-8"}[model]
    return [r for r in _read(RES / CLAUDE.format(model=fname)) if r.get("task") == task]


def paired_diff_ci(a, b, n=5000, seed=0):
    """Mean(a-b) over shared keys, paired percentile bootstrap."""
    keys = sorted(set(a) & set(b))
    diffs = [a[k] - b[k] for k in keys]
    if not diffs:
        return float("nan"), float("nan"), float("nan"), 0
    r = random.Random(seed)
    k = len(diffs)
    means = sorted(sum(diffs[r.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(diffs) / k, means[int(0.025 * n)], means[int(0.975 * n)], k


def mean(d):
    return sum(d.values()) / len(d) if d else float("nan")


for task in ("arith", "logic"):
    print(f"\n=== {task} ===")
    for model in ("Llama", "Sonnet", "Opus"):
        rows = rows_for(model, task)
        auto = keyed(rows, "source_first_auto")
        summ = keyed(rows, "langchain_summary")
        mem0 = keyed(rows, "mem0")
        vec = keyed(rows, "vector_rag")
        d, lo, hi, k = paired_diff_ci(auto, summ)
        print(f"  {model:>6}  auto={mean(auto):.2f}  summary={mean(summ):.2f}  "
              f"auto-summary={d:+.2f} [{lo:+.2f},{hi:+.2f}] (n={k})  "
              f"auto-mem0={mean(auto)-mean(mem0):+.2f}  auto-vec={mean(auto)-mean(vec):+.2f}")
