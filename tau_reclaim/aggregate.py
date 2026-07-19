"""Aggregate all results_*.jsonl into the paper's tau-bench table + a summary.json.
Judge-free throughout: outcomes are tau-bench DB-state-hash classifications.

    python aggregate.py           # print the table
    python aggregate.py --json    # also (re)write summary.json
"""
from __future__ import annotations

import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# display order: OSS scale ladder, then frontier, low -> high capability within each block
ORDER = [
    ("meta-llama/llama-3.2-3b-instruct", "Llama-3.2-3B"),
    ("meta-llama/llama-3.1-8b-instruct", "Llama-3.1-8B"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama-3.3-70B"),
    ("openai/gpt-4o-mini", "GPT-4o-mini"),
    ("gemini-3.5-flash", "Gemini-3.5-Flash"),
    ("grok-4.3", "Grok-4.3"),
    ("claude-opus-4-8", "Claude-Opus-4.8"),
    ("gpt-5.4", "GPT-5.4"),
]
POLICIES = ["source_first", "lossy", "lossy_padded", "blank"]


def load():
    by = {}
    for path in glob.glob(os.path.join(HERE, "results_*.jsonl")):
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


def summarize():
    by = load()
    out = {"models": {}, "policies": POLICIES}
    for mid, disp in ORDER:
        if mid not in by:
            continue
        rows = by[mid]
        out["models"][disp] = {
            "model_id": mid,
            "n_per_policy": len([r for r in rows if r["policy"] == "lossy"]),
            "reclaim": {p: rate(rows, p, "reclaim") for p in POLICIES},
            "lossy_stuck": rate(rows, "lossy", "stuck"),
            "blank_stuck": rate(rows, "blank", "stuck"),
        }
    # pooled reclaim per policy
    allrows = [r for rows in by.values() for r in rows]
    out["pooled_reclaim"] = {p: rate(allrows, p, "reclaim") for p in POLICIES}
    out["n_total"] = len(allrows)
    return out


def main():
    s = summarize()
    print(f"tau-bench governed-action reclaim  ({s['n_total']} rows, {len(s['models'])} models)")
    print(f"{'model':18s}| {'src_first':>9} {'lossy':>6} {'lossy_pad':>9} {'blank':>6} | n")
    print("-" * 62)
    for disp, m in s["models"].items():
        rc = m["reclaim"]
        print(f"{disp:18s}| {rc['source_first']:>9} {rc['lossy']:>6} {rc['lossy_padded']:>9} "
              f"{rc['blank']:>6} | {m['n_per_policy']}")
    p = s["pooled_reclaim"]
    print(f"{'POOLED':18s}| {p['source_first']:>9} {p['lossy']:>6} {p['lossy_padded']:>9} {p['blank']:>6} |")
    if "--json" in sys.argv:
        with open(os.path.join(HERE, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        print("\nwrote summary.json")


if __name__ == "__main__":
    main()
