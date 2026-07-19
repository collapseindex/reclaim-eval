"""Aggregate the mandatory-action experiment (run_forced.py) into the paper's two-condition table.

Condition A (transfer): a safe-exit tool (transfer_to_human) is offered alongside the exchange.
Condition B (commit): NO safe option is offered (the pure mandatory-field analog of tab:interface).
Both under a LOSSY memory (no source). Judge-free: outcomes are tau-bench DB-state classifications
plus the tool the model chose.

    python aggregate_forced.py
"""
from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = [
    ("meta-llama/llama-3.2-3b-instruct", "Llama-3.2-3B"),
    ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama-3.3-70B"),
    ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ("gemini-3.5-flash", "Gemini-3.5-Flash"),
    ("grok-4.3", "Grok-4.3"),
    ("gpt-5.4", "GPT-5.4"),
    ("claude-opus-4-8", "Claude-Opus-4.8"),
]


def load(pattern):
    by = {}
    for path in glob.glob(os.path.join(HERE, pattern)):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            by.setdefault(r["model"], []).append(r)
    return by


def rate(rows, policy, outcome):
    s = [r for r in rows if r["policy"] == policy]
    return round(sum(r["outcome"] == outcome for r in s) / len(s), 3) if s else None


def main():
    # results_forced_commit_* are condition B; results_forced_<model> (not starting 'c') are condition A
    A = load("results_forced_[!c]*.jsonl")
    commitB = load("results_forced_commit_*.jsonl")

    print("MANDATORY ACTION under a LOSSY memory  (judge-free; tau-bench retail)")
    print("=" * 78)
    print("Condition A: safe-exit tool (transfer_to_human) offered.")
    print(f"{'model':18}| harm  escalate | src_first reclaim (fix still works?)")
    for mid, disp in ORDER:
        if mid not in A:
            continue
        r = A[mid]
        print(f"{disp:18}| {rate(r,'lossy','harm'):>4}  {rate(r,'lossy','safe_exit'):>8} | "
              f"{rate(r,'source_first','reclaim')}")
    print("\nCondition B: NO safe option (commit-only). Frontier only.")
    print(f"{'model':18}| harm  refuse(resist)")
    for mid, disp in ORDER:
        if mid not in commitB:
            continue
        r = commitB[mid]
        print(f"{disp:18}| {rate(r,'lossy','harm'):>4}  {rate(r,'lossy','safe_exit'):>8}")


if __name__ == "__main__":
    main()
