#!/usr/bin/env python3
"""tab:wall / tab:logic (integrity sweep) with bootstrap 95% CIs at the cranked N.

Reads fix_<model>_<task>.jsonl. Filters to current problem ids, so llama reads n=96 (32 problems
x 3 seeds) and grok reads its clean canonical n=24. Directed arm. Prints the table and the
paste-ready LaTeX body rows.

    python scripts/integrity_table_ci.py            # human-readable
    python scripts/integrity_table_ci.py --latex    # LaTeX rows for tab:wall / tab:logic
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
sys.path.insert(0, str(ROOT / "src"))
from reclaim.problems import TASKS as _PROBS  # noqa: E402
PID2PROB = {p.pid for fam in _PROBS.values() for p in fam}

FILES = {"llama": "fix_meta-llama_llama-3.1-8b-instruct_{task}.jsonl",
         "grok": "fix_x-ai_grok-4.3-20260430_{task}.jsonl"}
INTEG = (1.0, 0.6, 0.3, 0.1)
POLICIES = ("lossy", "lossy_padded", "source_first")
PCOL = {"lossy": "\\lossy", "lossy_padded": "\\padded", "source_first": "\\srcfirst"}
ARM = "directed"


def boot_ci(xs, n=5000, seed=0):
    if not xs:
        return float("nan"), float("nan"), float("nan")
    r = random.Random(seed)
    k = len(xs)
    means = sorted(sum(xs[r.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return sum(xs) / k, means[int(0.025 * n)], means[int(0.975 * n)]


def _b(x):
    if x >= 0.9995:
        return "1"
    if x <= 0.0005:
        return "0"
    return f"{x:.2f}".lstrip("0")


def cells(model, task):
    p = RES / FILES[model].format(task=task)
    if not p.exists():
        return {}
    out = defaultdict(list)
    for line in open(p, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["arm"] == ARM and r.get("pid") in PID2PROB:
            out[(r["integrity"], r["policy"])].append(1 if r["correct"] else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()
    for task in ("arith", "logic"):
        data = {m: cells(m, task) for m in FILES}
        n_llama = len(data["llama"].get((0.1, "source_first"), []))
        n_grok = len(data["grok"].get((0.1, "source_first"), []))
        if args.latex:
            print(f"% --- {task}: tab:{'wall' if task=='arith' else 'logic'} body (llama n={n_llama}, grok n={n_grok}) ---")
            for g in INTEG:
                row = [f"{g:.1f}"]
                for m in ("llama", "grok"):
                    for pol in POLICIES:
                        mean, lo, hi = boot_ci(data[m].get((g, pol), []))
                        body = f"{mean:.2f}_{{[{_b(lo)},{_b(hi)}]}}"   # full value, abbreviated CI
                        cell = (f"$\\mathbf{{{body}}}$" if pol == "source_first" and g <= 0.3
                                else f"${body}$")
                        row.append(cell)
                print(" & ".join(row) + " \\\\")
        else:
            print(f"\n=== {task} directed RR, bootstrap 95% CI  (llama n={n_llama}, grok n={n_grok}) ===")
            print(f"{'g':>5} | " + "  ".join(f"{m[:4]}:{pol[:4]:>4}" for m in FILES for pol in POLICIES))
            for g in INTEG:
                parts = [f"{g:>5.1f}"]
                for m in FILES:
                    for pol in POLICIES:
                        mean, lo, hi = boot_ci(data[m].get((g, pol), []))
                        parts.append(f"{mean:.2f}[{_b(lo)},{_b(hi)}]")
                print(" | ".join([parts[0], "  ".join(parts[1:])]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
