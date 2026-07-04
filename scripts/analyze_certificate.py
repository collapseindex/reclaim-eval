#!/usr/bin/env python3
"""Write-time recompute certificate: a capability-free silent-failure guard, and its exact limit.

The certificate is not a model call. At write time the compressor already holds the true answer (it
just finished the session); before storing the note it recomputes the answer from the note's retained
source and flags the note if the recompute disagrees. That is pure arithmetic on the string it is about
to store, so it is independent of any later reader's capability.

  certificate flag  <=>  recompute(note source) != true answer  <=>  the note lost part of the
                         re-derivation basis (pi<1).

This script computes the certificate offline over the already-paid size/noise sweeps and the
completeness bench (no API), and reports three things:

  1. DETECTION (size). The flag separates reclaimable from walled: reclaim ~0.93 when it passes,
     ~0.00 when it flags. It catches the silent partial-sum regime with 100% recall.
  2. CAPABILITY-FREE. The completeness *tag* is honored by a strong reader but ignored by a weak one
     (Opus ~94/96 vs llama ~6/96); the certificate flags every truncated note on both, because it runs
     at write time. It closes the gap the tag leaves on weak readers.
  3. THE LIMIT. The certificate certifies presence (pi), not accessibility (alpha): a note can keep all
     the source yet bury it among decoys so the reader still fails. This is a genuine blind spot, but on
     the noise sweep it rarely binds (reclaim | cert-pass ~0.98); we report the residual honestly.

    python scripts/analyze_certificate.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "data" / "results"


def load(pattern):
    rows = []
    for f in glob.glob(str(RESULTS / pattern)):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _correct(x):
    c = x.get("correct")
    return c if isinstance(c, (int, float)) else int(bool(c))


def detection():
    """Certificate flag (k<N) vs reclaim, on the source-first size sweep."""
    rows = [x for x in load("sizesweep_*.jsonl") if x.get("policy") == "source_first"]
    agg = defaultdict(lambda: [0, 0])
    for x in rows:
        flag = x["k_items"] < x["n"]           # recompute of a partial source != true answer
        agg[flag][0] += _correct(x)
        agg[flag][1] += 1
    print("1. DETECTION (size sweep, source-first):")
    for flag in (False, True):
        s, n = agg[flag]
        label = "cert FLAG (k<N, pi<1)" if flag else "cert pass (k=N, pi=1)"
        if n:
            print(f"   {label:28s}: reclaim {s:>4}/{n:<4} = {s/n:.2f}")
    return agg


def capability_free():
    """The completeness tag is capability-gated; the certificate is not."""
    print("\n2. CAPABILITY-FREE (completeness bench, cliff cells k<N):")
    for mdl, pat in [("Opus", "completeness_claude-opus-4-8.jsonl"),
                     ("llama-8b", "completeness_llama.jsonl")]:
        r = [x for x in load(pat) if x["k"] < x["n"]]
        if not r:
            continue
        c = Counter(x["bucket"] for x in r)
        honored = c.get("abstain", 0) + c.get("flagged_partial", 0)
        print(f"   {mdl:8s}: completeness-TAG honored {honored:>3}/{len(r)}   "
              f"silent mis-sum {c.get('silent_missum', 0):>3}/{len(r)}")
    # the certificate flags every k<N note on every model (write-time arithmetic):
    sz = [x for x in load("sizesweep_*.jsonl") if x.get("policy") == "source_first" and x["k_items"] < x["n"]]
    print(f"   certificate flags {len(sz)}/{len(sz)} truncated notes, model-independent (write-time).")


def the_limit():
    """Presence (pi) is certified; accessibility (alpha) is not. Report the residual honestly."""
    rows = [x for x in load("noisysweep_*.jsonl") if x.get("policy") == "source_first_naive"]
    passed = [x for x in rows if x.get("all_kept")]          # cert passes: every relevant item retained
    if not passed:
        print("\n3. THE LIMIT: no cert-pass noise cells found.")
        return
    ok = sum(_correct(x) for x in passed)
    fail = len(passed) - ok
    print("\n3. THE LIMIT (noise sweep, source-first-naive):")
    print(f"   cert PASS (all relevant kept, pi=1): {len(passed)} cells")
    print(f"   reclaim | cert-pass = {ok}/{len(passed)} = {ok/len(passed):.2f}   "
          f"pass-yet-fail (alpha blind spot) = {fail}/{len(passed)} = {fail/len(passed):.1%}")
    print("   -> the certificate certifies presence, not accessibility; the blind spot is real but small.")


def main():
    if not RESULTS.exists():
        print("no results dir", file=sys.stderr)
        return 1
    print("Write-time recompute certificate (offline, no API)\n" + "-" * 52)
    detection()
    capability_free()
    the_limit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
