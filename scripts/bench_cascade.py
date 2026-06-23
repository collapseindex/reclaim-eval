#!/usr/bin/env python3
"""Cascade: does a single brittle-memory error compound when memory feeds memory?

The main paper measures ONE hop: session-1 drift -> compressed memory -> session-2 wall. Deployed
agents run a loop: they read their own memory, act, and compress the result into the NEXT memory. If
the compression is lossy (keeps the conclusion, drops the source), the mechanism is one-directional
(a conclusion never regenerates its source), so an error planted once should (a) corrupt every
downstream step and (b) stay uncorrectable no matter how late the correction arrives. This bench
runs that loop and measures the staircase.

Task: a running-ledger agent. Hop k reveals purchase k and asks for the running total
(truth_k = exact sum of purchases 1..k, brute-forced -> judge-free). A wrong subtotal is planted at
hop 1. After each hop the interaction is compressed into the carried memory under a policy, and that
memory is the only thing hop k+1 inherits. After H hops we deliver ONE directed correction and
measure whether the true final total is reclaimed.

  lossy:        memory keeps the running total (conclusion), drops the items (source).
  source_first: memory keeps the items (source), drops the total. Past the budget the item list is
                truncated -> the source-size cliff reappears as a CASCADE HORIZON.
  padded:       lossy + neutral filler to source_first length (budget-matched control).

Metrics per H: per-hop correctness, blast radius (# wrong downstream states), final reclaim after
correction. The cascade signature is lossy: blast radius ~ H and final reclaim 0 regardless of H;
source_first: blast radius ~ 0 and reclaim high until the budget horizon, then it cliffs too.

    python scripts/bench_cascade.py --probe
    python scripts/bench_cascade.py --hops 1,2,4,8 --budget 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.sizesweep import make_ledger
from reclaim.experiment import SYSTEM
from reclaim.llm import OpenRouterLLM, AnthropicLLM, parse_answer

ENVELOPE = "(Memory of an earlier session.) "
PREFIX_SF = ENVELOPE + "Purchases so far: "
SUFFIX_SF = " You are tracking a running total; an early figure may be off."
PAD = " (Archived running-ledger entry; no further working was preserved with it.)"


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)


def truths(items):
    """Running totals: truth[k] = sum of price*qty over the first k+1 items."""
    out, run = [], 0
    for noun, p, q in items:
        run += p * q
        out.append(run)
    return out  # out[k-1] is the total through purchase k


def clause(item):
    noun, p, q = item
    return f"{q} {noun} at ${p} each"


def sf_note(items_so_far, budget):
    """Source-first memory: as many whole purchase clauses (from the start) as fit `budget`.
    Returns (note, k_kept, complete). Past the budget the earliest-needed items drop -> the
    cascade horizon: the running total is no longer recomputable."""
    cl = [clause(it) for it in items_so_far]
    kept, used = [], len(PREFIX_SF) + len(SUFFIX_SF)
    for c in cl:
        add = len(c) + (2 if kept else 0)
        if used + add > budget:
            break
        kept.append(c); used += add
    note = PREFIX_SF + "; ".join(kept) + "." + SUFFIX_SF
    return note, len(kept), len(kept) == len(cl)


def compress(policy, items_so_far, last_total, k, budget):
    if policy == "source_first":
        note, kk, complete = sf_note(items_so_far, budget)
        return note, kk, complete
    if policy == "lossy":
        note = ENVELOPE + f"The running total through purchase {k} is ${last_total:g}."
        return note, 0, False
    if policy == "padded":
        base = ENVELOPE + f"The running total through purchase {k} is ${last_total:g}."
        target = len(sf_note(items_so_far, budget)[0])
        while len(base) < target:
            base += PAD
        return base, 0, False
    raise ValueError(policy)


def hop_prompt(memory, k, item, planted_wrong=None):
    noun, p, q = item
    if k == 1:
        note = (f"You are tracking a running total of purchases. Purchase 1: {q} {noun} at ${p} each."
                + (f" A note says they come to ${planted_wrong:g}." if planted_wrong is not None else ""))
        return note + " What is the running total so far? End with 'ANSWER: <number>'."
    return (f"{memory}\nNext, purchase {k}: {q} {noun} at ${p} each. "
            f"What is the running total now? End with 'ANSWER: <number>'.")


def run_chain(llm, items, H, policy, budget, planted):
    """One chain. Returns per-hop correctness, blast radius, final reclaim, memory trace."""
    tr = truths(items)
    i0 = items[0]
    planted_wrong = (i0[1] * i0[2]) + 7 if planted else None  # +$7 on purchase 1
    memory, kk, complete = "", 0, True
    per_hop, mem_has_source = [], []
    for k in range(1, H + 1):
        prompt = hop_prompt(memory, k, items[k - 1], planted_wrong if k == 1 else None)
        ans = parse_answer(llm.chat([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": prompt}]))
        per_hop.append(ans is not None and abs(ans - tr[k - 1]) < 0.5)
        memory, kk, complete = compress(policy, items[:k], ans if ans is not None else tr[k - 1], k, budget)
        mem_has_source.append(items[0][0].lower() in memory.lower())  # purchase-1 noun present?
    # one directed correction at the end
    corr = (f"{memory}\nLooking back, the early figure was wrong. Recompute the final running total "
            f"over all {H} purchases from scratch. End with 'ANSWER: <number>'.")
    fa = parse_answer(llm.chat([{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": corr}]))
    reclaim = fa is not None and abs(fa - tr[-1]) < 0.5
    blast = sum(1 for c in per_hop if not c)
    return {"per_hop": per_hop, "blast": blast, "reclaim": reclaim,
            "source_complete": complete, "mem_has_source_final": mem_has_source[-1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--hops", default="1,2,4,8")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--chains", type=int, default=8)
    ap.add_argument("--policies", default="lossy,padded,source_first",
                    help="comma list; drop padded for a leaner frontier confirm")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=0.02)
    ap.add_argument("--price-out", type=float, default=0.05)
    ap.add_argument("--probe", action="store_true", help="1 chain, H=4, lossy+source_first +control")
    args = ap.parse_args()

    Hs = [int(x) for x in args.hops.split(",")]
    POLICIES = [p.strip() for p in args.policies.split(",")]
    chains = 1 if args.probe else args.chains
    llm = make_llm(args.model, args.temp)
    out = ROOT / "data" / "results" / f"cascade_{args.model.replace('/', '_')}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.probe:
        H = 4
        items = make_ledger(0, H).items
        print(f"\nPROBE: one chain, H={H}, budget={args.budget}")
        print(f"  truths (running totals): {truths(items)}")
        rows = []
        for policy in ("lossy", "source_first"):
            for planted in (True, False):
                r = run_chain(llm, items, H, policy, args.budget, planted)
                tag = f"{policy}{'' if planted else ' (no-error control)'}"
                print(f"  {tag:<34} per-hop={['T' if c else 'F' for c in r['per_hop']]} "
                      f"blast={r['blast']}/{H} reclaim={int(r['reclaim'])}")
                rows.append((policy, planted, r))
        # validators (each can fail)
        lossy_p = next(r for pol, pl, r in rows if pol == "lossy" and pl)
        sf_p = next(r for pol, pl, r in rows if pol == "source_first" and pl)
        lossy_c = next(r for pol, pl, r in rows if pol == "lossy" and not pl)
        v1 = not lossy_p["mem_has_source_final"]      # lossy genuinely dropped the source
        v2 = sf_p["mem_has_source_final"]              # source_first kept it
        v3 = lossy_c["per_hop"][0]                      # no-error control: hop-1 correct (chaining adds no error)
        print(f"\n  VALIDATORS  (V1) lossy memory has NO source: {v1}  "
              f"(V2) source_first KEEPS source: {v2}  (V3) no-error control hop1 correct: {v3}")
        cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
        full_calls = len(Hs) * args.chains * 3 * (sum(Hs) / len(Hs) + 1) * 2
        print(f"  ({time.time()-t0:.0f}s, ${cost:.4f}) -> full (~{int(full_calls)} calls) ~"
              f"${cost/ max(1,(2*(H+1)*2)) * full_calls:.2f}")
        if not (v1 and v2 and v3):
            print("  !! a validator failed; the cascade reading is not yet trustworthy.")
        return 0

    # full sweep: policies x planted x H x chains
    print(f"\nCascade sweep, {args.model}, budget={args.budget}")
    agg = {}
    with open(out, "w", encoding="utf-8") as fh:
        for H in Hs:
            for policy in POLICIES:
                for planted in (True, False):
                    rs = []
                    for s in range(chains):
                        items = make_ledger(s, H).items
                        r = run_chain(llm, items, H, policy, args.budget, planted)
                        fh.write(json.dumps({"H": H, "policy": policy, "planted": planted,
                                             "store": s, **r}) + "\n"); fh.flush()
                        rs.append(r)
                    agg[(H, policy, planted)] = (
                        sum(x["blast"] for x in rs) / len(rs),
                        sum(x["reclaim"] for x in rs) / len(rs),
                        sum(x["source_complete"] for x in rs) / len(rs))

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    print(f"\n  {'H':>3} {'policy':>13} {'planted':>8}  {'blast/H':>8} {'final reclaim':>14} {'src-complete':>12}")
    for H in Hs:
        for policy in POLICIES:
            for planted in (True, False):
                b, rc, sc = agg[(H, policy, planted)]
                print(f"  {H:>3} {policy:>13} {str(planted):>8}  {b:>5.1f}/{H:<2} {rc:>14.2f} {sc:>12.2f}")
    print(f"\n  cascade signature: lossy planted -> blast~H, reclaim 0 at every H; "
          f"source_first -> blast~0, reclaim high until the budget horizon (src-complete falls).")
    print(f"  ({time.time()-t0:.0f}s, ${cost:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
