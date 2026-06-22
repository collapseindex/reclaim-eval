#!/usr/bin/env python3
"""Can the DEPLOYABLE distiller locate the answer-determining source under noise?

The noise sweep contrasts a positional source-first note (naive, crowds the bought items out as
decoys grow) with a relevance-aware one (denoised, the ORACLE that knows which items are bought).
The denoised arm "has that knowledge by construction," so locating is untested deployably. Here we
add the deployable arm: hand source-first-auto the full noisy ledger (bought items interleaved with
'considered, not bought' decoys) and let the LLM distiller decide what to keep, then reclaim.

  auto >= denoised  => the LLM can do the locating the oracle does (a deployable positive)
  auto ~= naive     => locating is the real wall, now quantified with a deployable method

We log both reclaim and a structural locating metric: how many of the bought items survive in the
distilled note (independent of whether the answer is then recomputed correctly).

    python scripts/bench_locating.py --probe
    python scripts/bench_locating.py --decoys 0,4,8,16
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

from reclaim.sizesweep import make_noisy_ledger, _clause, ASK
from reclaim.realworld import source_first_auto
from reclaim.experiment import SYSTEM, reclaim_cross, score, _logged_answer
from reclaim.llm import OpenRouterLLM

N_STORES = 8
N_RELEVANT = 4
ENVELOPE = "(Memory of an earlier session.) "


def session1(nl):
    """A session-1 transcript of the noisy shopping problem: every item (bought + decoy) is stated,
    then the model commits to the planted wrong total. This is what the distiller must compress."""
    items = "; ".join(_clause(r) for r in nl.rows)
    p = nl.problem
    return [
        {"role": "user", "content": f"{p.question} The store listed: {items}. "
         f"{p.wrong_premise}. What is {ASK}?"},
        {"role": "assistant", "content": f"The {ASK} is ${p.drift:g}."},
    ]


def bought_kept(nl, note: str) -> int:
    """Structural locating metric: how many bought-item nouns survive in the distilled note."""
    note_l = (note or "").lower()
    return sum(1 for i in nl.relevant_idx if nl.rows[i][0].lower() in note_l)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--decoys", default="0,4,8,16")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=0.02)
    ap.add_argument("--price-out", type=float, default=0.05)
    ap.add_argument("--probe", action="store_true", help="1 store x 2 decoy points, then cost")
    args = ap.parse_args()

    decoys = [int(x) for x in args.decoys.split(",")]
    stores = 1 if args.probe else N_STORES
    if args.probe:
        decoys = decoys[:2]
    out = ROOT / "data" / "results" / f"locating_{args.model.replace('/', '_')}.jsonl"
    llm = OpenRouterLLM(model=args.model, temperature=args.temp)

    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for d in decoys:
            for s in range(stores):
                nl = make_noisy_ledger(s, N_RELEVANT, d)
                note_body = source_first_auto(session1(nl), args.model)
                note = ENVELOPE + note_body
                kept = bought_kept(nl, note_body)
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": note},
                        {"role": "user", "content": reclaim_cross(nl.problem, "directed")}]
                reply = llm.chat(msgs)
                row = {"store": s, "decoys": d, "policy": "source_first_auto",
                       "bought_kept": kept, "n_relevant": N_RELEVANT,
                       "answer": _logged_answer(reply, nl.problem),
                       "correct": score(reply, nl.problem)}
                fh.write(json.dumps(row) + "\n"); fh.flush()
                rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    dt = time.time() - t0
    if args.probe:
        print(f"\nPROBE: {len(rows)} cell(s), {dt:.0f}s, ~${cost:.4f} -> "
              f"full ({len(decoys)*N_STORES} cells) scales up")
        for r in rows:
            print(f"  decoys={r['decoys']}: correct={r['correct']} bought_kept={r['bought_kept']}/{N_RELEVANT} ans={r['answer']!r}")
        return 0

    print(f"\nDeployable locating (source-first-auto under noise), {args.model} "
          f"({dt:.0f}s, ~${cost:.2f}):")
    print(f"  {'decoys':>7}  {'auto reclaim':>12}  {'bought-kept':>11}   n")
    by = defaultdict(list)
    for r in rows:
        by[r["decoys"]].append(r)
    for d in decoys:
        sub = by[d]
        rr = sum(r["correct"] for r in sub) / len(sub)
        bk = sum(r["bought_kept"] for r in sub) / len(sub)
        print(f"  {d:>7}  {rr:>12.2f}  {bk:>7.1f}/{N_RELEVANT}   {len(sub)}")
    print("  (paper: naive 0.96->0.00 by 8 decoys; denoised oracle holds 1.00)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
