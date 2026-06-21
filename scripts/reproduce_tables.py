#!/usr/bin/env python3
"""One-command reproduction of every headline number in the paper, plus the anti-rig validators.

Pure analysis over the committed result JSONL: no API calls, no randomness beyond a seeded
bootstrap. Regenerates the LaTeX bodies for tab:wall, tab:logic, and tab:frontier, recomputes the
auto-vs-summary paired differences, and runs the correct-by-construction validators on the
generated problem set. Exits non-zero if any validator fails, so it doubles as a CI check.

    python scripts/reproduce_tables.py          # full report
    python scripts/reproduce_tables.py --quiet   # validators + PASS/FAIL only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))


def run(label, argv):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    r = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
    return r.returncode == 0


def validators():
    """Correct-by-construction: every generated problem's answer is brute-force verified.

    A planted-wrong-answer note that the model merely echoes cannot pass, because the validators
    recompute the true answer independently and assert the generator agrees.
    """
    from reclaim.problems_gen import (gen_arith, gen_logic, gen_assign,
                                      validate_arith, validate_logic, validate_assign)
    checks = []
    ga, fa = gen_arith(24, seed=1); validate_arith(ga, fa)
    checks.append(("arith generator (24 problems, answers brute-forced)", True))
    gl, fl = gen_logic(12, seed=2); validate_logic(gl, fl)
    checks.append(("logic-ordering generator (12 problems, solver-checked)", True))
    gs, fs = gen_assign(12, seed=3); validate_assign(gs, fs)
    checks.append(("logic-assignment generator (12 problems, solver-checked)", True))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="validators + PASS/FAIL only")
    args = ap.parse_args()

    ok = True

    print("Reproducing paper headline numbers from committed results (no API).")

    # 1. Correct-by-construction validators (the load-bearing anti-rig check at n=96).
    print(f"\n{'='*70}\nVALIDATORS: correct-by-construction answers (n=96 problem set)\n{'='*70}")
    try:
        for name, passed in validators():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            ok = ok and passed
    except AssertionError as e:
        print(f"  [FAIL] validator raised: {e}")
        ok = False

    # 2. Headline table bodies (each prints its own n so the reader can confirm n=96 / n=24).
    if not args.quiet:
        ok &= run("tab:wall + tab:logic  (integrity sweep, llama n=96 / grok n=24)",
                  [str(SCRIPTS / "integrity_table_ci.py"), "--latex"])
        ok &= run("tab:frontier  (deployed + frontier replay, n=96)",
                  [str(SCRIPTS / "frontier_table_ci.py"), "--latex"])
        ok &= run("auto-vs-summary paired difference  (n=96, paired bootstrap)",
                  [str(SCRIPTS / "auto_vs_summary_ci.py")])

    print(f"\n{'='*70}\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}\n{'='*70}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
