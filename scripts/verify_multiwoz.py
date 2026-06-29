#!/usr/bin/env python3
"""Verify tab:attractor was scored strictly (not via the buggy _ANY fallback). Re-score the logged
MultiWOZ failmode replies two ways: STRICT (asserted_time = ANSWER-line only, what the paper used)
vs BUGGY (extract_time, with the grab-any-time fallback). If the table matches strict and buggy is
higher, the table is clean and the strict choice mattered.
"""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from reclaim.multiwoz import load_targets, extract_time, _canon
from bench_multiwoz_failmode import asserted_time

tgt = {t.dialogue_id: t for t in load_targets(str(ROOT / "data" / "multiwoz" / "dev_001.json"), max_n=60)}


def bucket(got, t):
    if got is None: return "abstain"
    if got == _canon(t.true_value): return "correct"
    if got == _canon(t.drift_value): return "inherit"
    return "novel"


for model in ["llama", "claude-opus-4-8", "claude-sonnet-4-6"]:
    f = ROOT / "data" / "results" / f"multiwoz_failmode_{model}.jsonl"
    if not f.exists():
        print(f"== {model}: FILE MISSING"); continue
    rows = [json.loads(l) for l in f.open(encoding="utf-8")]
    print(f"== {model} (n={len(rows)}) ==")
    for pol in ["lossy", "blank"]:
        sub = [r for r in rows if r["policy"] == pol and r["dialogue_id"] in tgt]
        n = len(sub)
        if not n:
            print(f"  {pol}: no rows"); continue
        logged = sum(r["bucket"] in ("inherit", "novel") for r in sub) / n
        strict = sum(bucket(asserted_time(r["reply"]), tgt[r["dialogue_id"]]) in ("inherit", "novel") for r in sub) / n
        buggy = sum(bucket(extract_time(r["reply"]), tgt[r["dialogue_id"]]) in ("inherit", "novel") for r in sub) / n
        print(f"  {pol:6} n={n}:  logged_emit={logged:.2f}  strict_emit={strict:.2f}  BUGGY_emit={buggy:.2f}")
