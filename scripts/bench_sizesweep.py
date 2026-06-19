#!/usr/bin/env python3
"""Source-size decay sweep: where the source-first law ends.

Fixes the carried-memory character budget B and sweeps the ledger size N. For each
(store, N, budget, policy, seed) it builds the carried memory, delivers one directed
correction, and scores the recomputed total objectively. As N outgrows B, the source-first
note can keep only k<N items and an exact sum needs all N, so its reclaim advantage decays to
the lossy floor. Two budgets show the crossover moves with B.

    python scripts/bench_sizesweep.py --dry                       # free validators, no API
    python scripts/bench_sizesweep.py --model llama               # full sweep on llama (OpenRouter)
    python scripts/bench_sizesweep.py --model claude-opus-4-8 \
        --ns 4,8,16,32 --budgets 220                              # frontier confirm points

Writes data/results/sizesweep_<model>.jsonl (checkpointed, resumable).
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

from reclaim.sizesweep import make_ledger, build_note
from reclaim.experiment import SYSTEM, reclaim_cross, score, _logged_answer

N_STORES = 8
DEF_NS = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 32]
DEF_BUDGETS = [300, 600]
POLICIES = ("source_first", "lossy_padded")
LLAMA = "meta-llama/llama-3.1-8b-instruct"


class SizeFake:
    """Deterministic validator model: recomputes the correct total IFF the carried note
    contains every line-item clause (the full source). It therefore reclaims only when the
    budget kept all N items; past the cliff (k<N) it returns the inherited wrong total. A
    real model scoring above this past the cliff is confabulating, not recomputing."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = self.completion_tokens = 0
        self._clauses = self._drift = self._correct = None

    def configure(self, clauses, drift, correct):
        self._clauses, self._drift, self._correct = clauses, drift, correct

    def chat(self, messages):
        self.calls += 1
        ctx = "\n".join(m["content"] for m in messages).lower()
        full = all(c.lower() in ctx for c in self._clauses)
        return f"ANSWER: {self._correct if full else self._drift:g}"


def make_llm(model, temp):
    if model in ("llama", LLAMA):
        from reclaim.llm import OpenRouterLLM
        return OpenRouterLLM(model=LLAMA, temperature=temp)
    from reclaim.llm import AnthropicLLM
    return AnthropicLLM(model=model, temperature=temp)


def ckpt_path(model):
    safe = model.replace("/", "_")
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"sizesweep_{safe}.jsonl"


def cells(ns, budgets, policies, seeds):
    for s in range(N_STORES):
        for n in ns:
            led = make_ledger(s, n)
            for b in budgets:
                for pol in policies:
                    note, k, locus_kept = build_note(led, b, pol)
                    for seed in range(seeds):
                        yield (s, n, b, pol, seed, led, note, k, locus_kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--dry", action="store_true", help="deterministic validators, no API")
    ap.add_argument("--ns", default=",".join(map(str, DEF_NS)))
    ap.add_argument("--budgets", default=",".join(map(str, DEF_BUDGETS)))
    ap.add_argument("--policies", default=",".join(POLICIES),
                    help="comma list; restrict to source_first for cheap frontier confirms")
    ap.add_argument("--arm", default="directed", choices=["directed", "generic"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=15.0)
    ap.add_argument("--price-out", type=float, default=75.0)
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    policies = tuple(p.strip() for p in args.policies.split(","))

    if args.dry:
        return run_dry(ns, budgets, args.seeds, args.arm)

    model = args.model
    ckpt = ckpt_path(model)
    done = set()
    if ckpt.exists():
        for line in open(ckpt, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["store"], r["n"], r["budget"], r["policy"], r["seed"], r["arm"]))

    temp = 0.0 if model.startswith("claude") else args.temp
    llm = make_llm(model, temp)
    todo = [c for c in cells(ns, budgets, policies, args.seeds)
            if (c[0], c[1], c[2], c[3], c[4], args.arm) not in done]
    print(f"model={model}  cells to run={len(todo)}  checkpoint={ckpt.name}")
    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8") as out:
        for i, (s, n, b, pol, seed, led, note, k, locus_kept) in enumerate(todo, 1):
            prob = led.problem
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": note},
                    {"role": "user", "content": reclaim_cross(prob, args.arm)}]
            reply = llm.chat(msgs)
            row = {"store": s, "n": n, "budget": b, "policy": pol, "seed": seed,
                   "arm": args.arm, "k_items": k, "locus_kept": locus_kept,
                   "model": model, "answer": _logged_answer(reply, prob),
                   "correct": score(reply, prob)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 24 == 0 or i == len(todo):
                cost = (llm.prompt_tokens / 1e6 * args.price_in
                        + llm.completion_tokens / 1e6 * args.price_out)
                tag = f"~${cost:.2f}" if model.startswith("claude") else f"{llm.calls} calls"
                print(f"  {i}/{len(todo)}  {tag}  ({time.time()-t0:.0f}s)")
    print(f"done. {llm.calls} calls.")
    return 0


def run_dry(ns, budgets, seeds, arm):
    """Validators built to come out false. The fake recomputes only with the full source."""
    rows = []
    fake = SizeFake()
    for s, n, b, pol, seed, led, note, k, locus_kept in cells(ns, budgets, POLICIES, seeds):
        prob = led.problem
        clauses = [f"{noun} at ${p} each ({q} bought)" for noun, p, q in led.items]
        fake.configure(clauses, prob.drift, prob.correct)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": note},
                {"role": "user", "content": reclaim_cross(prob, arm)}]
        reply = fake.chat(msgs)
        rows.append({"n": n, "budget": b, "policy": pol, "k_items": k,
                     "full": k == n, "correct": score(reply, prob)})

    by = defaultdict(list)
    for r in rows:
        by[(r["budget"], r["policy"], r["n"])].append(r["correct"])

    # V1: lossy never reclaims (no source at any N/budget).
    v1 = all(not c for r in rows if r["policy"] == "lossy_padded" for c in [r["correct"]])
    # V2: source_first reclaims IFF the full source fit the budget (k==n).
    v2 = all(r["correct"] == (r["k_items"] == r["n"])
             for r in rows if r["policy"] == "source_first")
    # V3: there exists a cliff in range, source_first goes 1.0 -> 0.0 as N grows (per budget).
    cliff = {}
    for b in budgets:
        rr = {n: sum(by[(b, "source_first", n)]) / len(by[(b, "source_first", n)])
              for n in ns}
        cliff[b] = rr
    v3 = all(max(c.values()) > 0.9 and min(c.values()) < 0.1 for c in cliff.values())
    # V4: the cliff moves right with budget (larger B keeps the source to a larger N).
    def last_full(rr):  # largest N still fully recovered
        full = [n for n in ns if rr[n] > 0.5]
        return max(full) if full else 0
    v4 = (len(budgets) < 2) or (last_full(cliff[max(budgets)]) > last_full(cliff[min(budgets)]))

    print("\nsource_first reclaim (fake) by budget x N  [should fall 1->0]:")
    for b in budgets:
        print(f"  B={b}: " + "  ".join(f"N{n}:{cliff[b][n]:.2f}" for n in ns))
    print(f"\nV1 lossy never reclaims:                 {'PASS' if v1 else 'FAIL'}")
    print(f"V2 source_first reclaims iff full source: {'PASS' if v2 else 'FAIL'}")
    print(f"V3 a 1->0 cliff exists in range:          {'PASS' if v3 else 'FAIL'}")
    print(f"V4 cliff moves right with budget:         {'PASS' if v4 else 'FAIL'}")
    ok = v1 and v2 and v3 and v4
    print(f"\n{'ALL PASS' if ok else 'FAILED'} ({sum([v1,v2,v3,v4])}/4)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
