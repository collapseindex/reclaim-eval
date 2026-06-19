#!/usr/bin/env python3
"""Noisy-source sweep: keeping the source is not enough, you must keep the RIGHT source.

The answer-determining (bought) items are buried among plausible 'considered, not bought'
decoys. At a fixed memory budget, a positional source-first note (naive) spends budget on
noise and crowds the bought items out, while a relevance-aware note (denoised) keeps only the
bought items and holds. The total is the exact sum over bought items, so scoring stays
objective (no judge). We fix the bought count and budget and sweep the decoy count.

    python scripts/bench_noisysweep.py --dry                      # free validators, no API
    python scripts/bench_noisysweep.py --model llama
    python scripts/bench_noisysweep.py --model claude-opus-4-8 --decoys 0,4,8,16,28 \
        --policies source_first_naive,source_first_denoised        # frontier confirm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.sizesweep import make_noisy_ledger, build_noisy_note, _clause
from reclaim.experiment import SYSTEM, reclaim_cross, score, _logged_answer
from reclaim.llm import OpenRouterLLM, AnthropicLLM

N_STORES = 8
N_RELEVANT = 4
BUDGET = 420
DEF_DECOYS = [0, 2, 4, 6, 8, 12, 16, 24, 32]
POLICIES = ("source_first_naive", "source_first_denoised", "lossy_padded")
LLAMA = "meta-llama/llama-3.1-8b-instruct"


class NoisyFake:
    """Validator model: recomputes correctly IFF every bought-item clause survived in the
    note. Naive notes that drop a bought item to noise therefore fail; denoised never do."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = self.completion_tokens = 0

    def configure(self, rel_clauses, drift, correct):
        self._rel, self._drift, self._correct = rel_clauses, drift, correct

    def chat(self, messages):
        self.calls += 1
        ctx = "\n".join(m["content"] for m in messages).lower()
        full = all(c.lower() in ctx for c in self._rel)
        return f"ANSWER: {self._correct if full else self._drift:g}"


def make_llm(model, temp):
    if model in ("llama", LLAMA):
        return OpenRouterLLM(model=LLAMA, temperature=temp)
    return AnthropicLLM(model=model, temperature=temp)


def ckpt_path(model):
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"noisysweep_{model.replace('/', '_')}.jsonl"


def cells(decoys, policies, seeds):
    for s in range(N_STORES):
        for d in decoys:
            nl = make_noisy_ledger(s, N_RELEVANT, d)
            for pol in policies:
                note, rel_kept, all_kept = build_noisy_note(nl, BUDGET, pol)
                for seed in range(seeds):
                    yield (s, d, pol, seed, nl, note, rel_kept, all_kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--decoys", default=",".join(map(str, DEF_DECOYS)))
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--arm", default="directed", choices=["directed", "generic"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=15.0)
    ap.add_argument("--price-out", type=float, default=75.0)
    args = ap.parse_args()

    decoys = [int(x) for x in args.decoys.split(",")]
    policies = tuple(p.strip() for p in args.policies.split(","))

    if args.dry:
        return run_dry(decoys, args.seeds, args.arm)

    model = args.model
    ckpt = ckpt_path(model)
    done = set()
    if ckpt.exists():
        for line in open(ckpt, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["store"], r["decoys"], r["policy"], r["seed"], r["arm"]))

    temp = 0.0 if model.startswith("claude") else args.temp
    llm = make_llm(model, temp)
    todo = [c for c in cells(decoys, policies, args.seeds)
            if (c[0], c[1], c[2], c[3], args.arm) not in done]
    print(f"model={model}  cells to run={len(todo)}  checkpoint={ckpt.name}")
    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8") as out:
        for i, (s, d, pol, seed, nl, note, rel_kept, all_kept) in enumerate(todo, 1):
            prob = nl.problem
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": note},
                    {"role": "user", "content": reclaim_cross(prob, args.arm)}]
            reply = llm.chat(msgs)
            row = {"store": s, "decoys": d, "n_relevant": N_RELEVANT, "budget": BUDGET,
                   "policy": pol, "seed": seed, "arm": args.arm, "rel_kept": rel_kept,
                   "all_kept": all_kept, "model": model,
                   "answer": _logged_answer(reply, prob), "correct": score(reply, prob)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 24 == 0 or i == len(todo):
                cost = (llm.prompt_tokens / 1e6 * args.price_in
                        + llm.completion_tokens / 1e6 * args.price_out)
                tag = f"~${cost:.2f}" if model.startswith("claude") else f"{llm.calls} calls"
                print(f"  {i}/{len(todo)}  {tag}  ({time.time()-t0:.0f}s)")
    print(f"done. {llm.calls} calls.")
    return 0


def run_dry(decoys, seeds, arm):
    fake = NoisyFake()
    rows = []
    for s, d, pol, seed, nl, note, rel_kept, all_kept in cells(decoys, POLICIES, seeds):
        prob = nl.problem
        rel_clauses = [_clause(nl.rows[i]) for i in nl.relevant_idx]
        fake.configure(rel_clauses, prob.drift, prob.correct)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": note},
                {"role": "user", "content": reclaim_cross(prob, arm)}]
        reply = fake.chat(msgs)
        rows.append({"decoys": d, "policy": pol, "all_kept": all_kept,
                     "correct": score(reply, prob)})

    by = defaultdict(list)
    for r in rows:
        by[(r["policy"], r["decoys"])].append(r["correct"])
    rr = lambda pol, d: sum(by[(pol, d)]) / len(by[(pol, d)])

    v1 = all(not c for r in rows if r["policy"] == "lossy_padded" for c in [r["correct"]])
    v2 = all(rr("source_first_denoised", d) > 0.99 for d in decoys)
    # naive reclaims iff all bought items survived the budget
    v3 = all(r["correct"] == r["all_kept"]
             for r in rows if r["policy"] == "source_first_naive")
    # naive degrades with noise: high at d=0, low at the largest decoy count
    v4 = rr("source_first_naive", min(decoys)) > 0.9 and rr("source_first_naive", max(decoys)) < 0.5

    print("\nreclaim (fake) by policy x decoy count:")
    for pol in POLICIES:
        print(f"  {pol:24} " + "  ".join(f"d{d}:{rr(pol,d):.2f}" for d in decoys))
    print(f"\nV1 lossy never reclaims:                  {'PASS' if v1 else 'FAIL'}")
    print(f"V2 denoised always reclaims:              {'PASS' if v2 else 'FAIL'}")
    print(f"V3 naive reclaims iff all bought kept:    {'PASS' if v3 else 'FAIL'}")
    print(f"V4 naive degrades as noise grows:         {'PASS' if v4 else 'FAIL'}")
    ok = v1 and v2 and v3 and v4
    print(f"\n{'ALL PASS' if ok else 'FAILED'} ({sum([v1,v2,v3,v4])}/4)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
