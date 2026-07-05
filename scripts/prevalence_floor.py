#!/usr/bin/env python3
"""A judge-free, deterministic LOWER BOUND on compact-source prevalence per domain.

The LLM prevalence audit (bench_prevalence.py) orders the domains robustly but cannot identify an
absolute level (two labelers disagree, kappa=0.15). This script sidesteps the fuzzy compact/diffuse
boundary the same way the MultiWOZ replication does: it only claims cases that are compact *by
construction*. A trace counts as compact iff a checkable value (a number of >=2 digits, a time, or a
currency amount) is stated and then *recurs* later in the trace, the signature of a concrete value being
carried and referenced. This is a token test: no model call, no annotator, no inter-labeler disagreement.

Because the test is conservative (verbatim match + recurrence, and it misses paraphrased values), the
fraction it flags is a LOWER BOUND on the compact-source share. We validate that it is a genuine floor,
not a loose proxy, by aligning to the audit's exact sample (by index) and measuring precision: of the
traces the token test flags, what fraction the LLM audit also labelled COMPACT. High precision means few
false positives, so the flagged fraction really is a floor.

Reproduces the audit's sampling exactly (seed 0, 100/domain), so it can be matched to
`data/results/prevalence_audit.jsonl` by index. Re-streams the three corpora from the HF Hub (no API key,
no cost). A deployer can point the same test at their own traffic.

    python scripts/prevalence_floor.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from bench_prevalence import load_domain  # reuse the exact corpus loaders / sampling

# a checkable value: currency ($12.50), a clock time (16:15), or a >=2-digit number
VALUE_RE = re.compile(r"\$\d[\d,]*\.?\d*|\b\d{1,2}:\d{2}\b|\b\d{2,}(?:[.,]\d+)?\b")


def _norm(tok: str) -> str:
    return tok.replace(",", "").rstrip(".")


def is_compact_floor(conv: str) -> bool:
    """Compact by construction: a checkable value stated and referenced again (recurs >=2x)."""
    counts = Counter(_norm(t) for t in VALUE_RE.findall(conv))
    return any(v >= 2 for v in counts.values())


def load_audit_labels():
    p = ROOT / "data" / "results" / "prevalence_audit.jsonl"
    labels = {}
    if p.exists():
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                labels[(r["domain"], int(r["i"]))] = (r["label"], int(r["chars"]))
    return labels


def main():
    audit = load_audit_labels()
    print("Judge-free deterministic compact-source floor (seed 0, 100/domain)\n" + "-" * 62)
    print(f"{'domain':8s} {'n':>3} {'floor':>7} {'aligned':>9} {'precision(vs LLM COMPACT)':>26}")
    for dom in ("chat", "tool", "agentic"):
        rows = load_domain(dom, 100, 0)
        flagged = aligned = tp = tp_compact = 0
        for i, conv in enumerate(rows):
            f = is_compact_floor(conv)
            flagged += f
            lab_ch = audit.get((dom, i))
            if lab_ch is not None:
                lab, ch = lab_ch
                if abs(len(conv) - ch) <= 2:
                    aligned += 1
                if f:
                    tp += 1
                    tp_compact += int(lab == "COMPACT")
        n = len(rows)
        prec = tp_compact / tp if tp else float("nan")
        al = f"{aligned}/{n}" if audit else "n/a"
        pr = f"{prec:.2f} ({tp_compact}/{tp})" if audit and tp else "n/a"
        print(f"{dom:8s} {n:>3} {flagged / n:>7.2f} {al:>9} {pr:>26}")
    print("\nFloor = fraction with a checkable value stated and recurring (a conservative lower bound).")
    print("Precision = of flagged traces, fraction the LLM audit also called COMPACT (validates the floor).")


if __name__ == "__main__":
    main()
