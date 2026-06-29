#!/usr/bin/env python3
"""Is lossy memory worse than EMPTY memory on arithmetic? (the behavioral headline, off MultiWOZ)

The paper leads with "a memory that kept the wrong value is worse than one that kept nothing."
That strict head-to-head was previously shown only on MultiWOZ; on arithmetic we had inherit rates
and a 0.00 reclaim wall, but no matched empty-memory baseline. This bench adds it.

At the wall (g=0.1) we deliver the SAME directed reclaim correction under two policies:
  - lossy: keeps the stale wrong conclusion, drops the source (the realistic default).
  - blank: keeps NEITHER source nor conclusion (the empty-memory baseline; nothing to inherit).

Neither policy retains the source, so neither can recompute the truth: reclaim rate is ~0 for both.
The DIFFERENCE is behavioral, in what the model emits when it cannot recover the answer:
  - emit  : returns a confident numeric value anyway (wrong). Under lossy this is the inherited
            attractor (the planted drift value); under blank there is no value to inherit.
  - abstain: declines / flags that it cannot determine the answer (the safe behavior).

"Worse than empty" = lossy emits a confident wrong value where blank abstains. Run on the two base
models where the inherit rates are established (llama 43%, grok 90%).

    python scripts/bench_blank.py --model llama --probe
    python scripts/bench_blank.py --model llama
    python scripts/bench_blank.py --model x-ai/grok-4.3-20260430 --price-in 3 --price-out 15
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.problems import PROBLEMS
from reclaim.experiment import SYSTEM, memory_note, reclaim_cross, _logged_answer, _configure
from reclaim.llm import OpenRouterLLM, AnthropicLLM, OpenAILLM, XAILLM, parse_answer

G = 0.1   # the wall: lossy = conclusion only; blank = nothing retained


def classify(reply, prob):
    """emit (confident wrong number) vs abstain (no value) vs true (recovered), plus whether an
    emitted value is the inherited attractor (drift) -- only meaningful for lossy.

    Scores the RETURNED value (the committed ANSWER line), matching the paper's stated methodology:
    we score the value the model hands back, not whether the surrounding prose hedges, because a
    downstream system sees only the value. The fixed parse_answer requires the number to sit on the
    ANSWER line and no longer scrapes a stray number out of refusal prose, so a reply with no
    committed value is an abstention and a reply that commits a wrong value is an emit. We do NOT
    gate on a hedge wordlist: that would depart from the stated methodology and is itself the brittle
    heuristic this fix removes (a wordlist catches some hedges and misses others). See
    NOTE_parser_fix.md."""
    a = parse_answer(reply)
    if a is None:
        return "abstain", False               # no committed value: the safe behavior
    if abs(a - float(prob.correct)) < 0.5:
        return "true", False                  # recomputed truth (rare with no source)
    is_attractor = abs(a - float(prob.drift)) < 0.5
    return "emit", is_attractor               # a confident wrong value


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model.startswith("gpt-5"):              # OpenAI direct reasoning model (deterministic, no temp)
        return OpenAILLM(model=model)
    if model.startswith("grok"):               # official xAI endpoint (deterministic frontier reader)
        return XAILLM(model=model)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=0.3)
    ap.add_argument("--price-out", type=float, default=0.3)
    ap.add_argument("--probe", action="store_true", help="one cell only, then report cost")
    args = ap.parse_args()

    llm = make_llm(args.model, args.temp)
    out = ROOT / "data" / "results" / f"blank_{args.model.replace('/', '_')}.jsonl"
    policies = ["lossy", "blank"]
    probs = PROBLEMS[:1] if args.probe else PROBLEMS
    seeds = 1 if args.probe else args.seeds
    pols_run = ["lossy"] if args.probe else policies

    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for prob in probs:
            for pol in pols_run:
                note = memory_note(prob, G, pol)
                for seed in range(seeds):
                    _configure(llm, prob)
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note},
                            {"role": "user", "content": reclaim_cross(prob, "directed")}]
                    reply = llm.chat(msgs)
                    bucket, attractor = classify(reply, prob)
                    row = {"pid": prob.pid, "policy": pol, "seed": seed, "bucket": bucket,
                           "attractor": attractor, "answer": _logged_answer(reply, prob)}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    dt = time.time() - t0
    if args.probe:
        per = cost / max(1, len(rows))
        full = per * len(PROBLEMS) * args.seeds * len(policies)
        print(f"\nPROBE {args.model}: {len(rows)} call(s), {dt:.0f}s, bucket={rows[0]['bucket']}")
        print(f"  tokens in/out: {llm.prompt_tokens}/{llm.completion_tokens}")
        print(f"  cost this probe ~${cost:.4f}  ->  full run "
              f"({len(PROBLEMS)*args.seeds*len(policies)} cells) ~${full:.2f}")
        print(f"  raw reply: {rows[0].get('answer')!r}")
        return 0

    print(f"\nlossy vs blank at the wall, {args.model} ({dt:.0f}s, ~${cost:.2f}):")
    print(f"  {'policy':>7} {'emit':>6} {'abstain':>8} {'true':>5}  {'attractor':>9}   n")
    for pol in policies:
        sub = [r for r in rows if r["policy"] == pol]
        c = Counter(r["bucket"] for r in sub); n = len(sub)
        attr = sum(r["attractor"] for r in sub)
        print(f"  {pol:>7} {c['emit']/n:>6.2f} {c['abstain']/n:>8.2f} {c['true']/n:>5.2f}  "
              f"{attr/n:>9.2f}   {n}")
    le = [r for r in rows if r["policy"] == "lossy"]
    be = [r for r in rows if r["policy"] == "blank"]
    l_emit = sum(r["bucket"] == "emit" for r in le) / len(le)
    b_emit = sum(r["bucket"] == "emit" for r in be) / len(be)
    print(f"\n  confident-wrong-emission: lossy {l_emit:.2f} vs blank {b_emit:.2f} "
          f"(delta {l_emit - b_emit:+.2f}) -- lossy worse than empty by this much")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
