#!/usr/bin/env python3
"""Reclaim on real conversational memory (MultiWOZ slot recovery).

Cross-session reclaim where the source is a real chatty dialogue and the target is a checkable
time slot. A lossy memory keeps a corrupted confirmation (walls); a source-first memory keeps
the user's verbatim utterance (recovers). Objective slot scoring, no judge.

    python scripts/bench_multiwoz.py --dry                     # free validators, no API
    python scripts/bench_multiwoz.py --model llama
    python scripts/bench_multiwoz.py --model claude-opus-4-8 --policies source_first,lossy
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

from reclaim.multiwoz import load_targets, build_note, reclaim_msg, score
from reclaim.experiment import SYSTEM
from reclaim.llm import OpenRouterLLM, AnthropicLLM

DATA = ROOT / "data" / "multiwoz" / "dev_001.json"
POLICIES = ("source_first", "lossy", "lossy_padded", "blank")


class WozFake:
    """Validator: recovers the true slot value IFF that value's token is in the note (i.e. the
    source survived). Otherwise returns the drift (lossy) or a non-answer (blank)."""

    def __init__(self):
        self.calls = 0
        self.prompt_tokens = self.completion_tokens = 0

    def configure(self, true_v, drift_v):
        self._t, self._d = true_v, drift_v

    def chat(self, messages):
        self.calls += 1
        ctx = "\n".join(m["content"] for m in messages)
        return f"ANSWER: {self._t}" if self._t in ctx else f"ANSWER: {self._d}"


def make_llm(model, temp):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp)
    return OpenRouterLLM(model=model, temperature=temp)


def ckpt(model):
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"multiwoz_{model.replace('/', '_')}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--n-dialogues", type=int, default=30)
    ap.add_argument("--slots", default="", help="comma list to restrict slot types "
                    "(e.g. restaurant-booktime for the unambiguous exact-value subset)")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--arm", default="directed", choices=["directed", "generic"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=15.0)
    ap.add_argument("--price-out", type=float, default=75.0)
    args = ap.parse_args()

    from reclaim.multiwoz import SLOT_DESC
    slots = tuple(s.strip() for s in args.slots.split(",") if s.strip()) or tuple(SLOT_DESC)
    targets = load_targets(DATA, slots=slots, max_n=args.n_dialogues)
    policies = tuple(p.strip() for p in args.policies.split(","))

    if args.dry:
        return run_dry(targets, policies, args.seeds, args.arm)

    model = args.model
    path = ckpt(model)
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["dialogue_id"], r["policy"], r["seed"], r["arm"]))

    temp = 0.0 if model.startswith("claude") else args.temp
    llm = make_llm(model, temp)
    todo = [(t, pol, s) for t in targets for pol in policies for s in range(args.seeds)
            if (t.dialogue_id, pol, s, args.arm) not in done]
    print(f"model={model}  targets={len(targets)}  cells to run={len(todo)}")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as out:
        for i, (tgt, pol, seed) in enumerate(todo, 1):
            note = build_note(tgt, pol)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": note},
                    {"role": "user", "content": reclaim_msg(tgt, args.arm)}]
            reply = llm.chat(msgs)
            row = {"dialogue_id": tgt.dialogue_id, "slot": tgt.slot, "policy": pol, "seed": seed,
                   "arm": args.arm, "true": tgt.true_value, "drift": tgt.drift_value,
                   "model": model, "correct": score(reply, tgt)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 30 == 0 or i == len(todo):
                cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
                tag = f"~${cost:.2f}" if model.startswith("claude") else f"{llm.calls} calls"
                print(f"  {i}/{len(todo)}  {tag}  ({time.time()-t0:.0f}s)")
    summarize(path, policies, args.arm)
    return 0


def run_dry(targets, policies, seeds, arm):
    fake = WozFake()
    cell = defaultdict(list)
    for t in targets:
        fake.configure(t.true_value, t.drift_value)
        for pol in policies:
            note = build_note(t, pol)
            for _ in range(seeds):
                msgs = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": note},
                        {"role": "user", "content": reclaim_msg(t, arm)}]
                cell[pol].append(score(fake.chat(msgs), t))
    rr = {p: sum(cell[p]) / len(cell[p]) for p in policies}
    print("\nfake reclaim by policy:", {p: round(v, 2) for p, v in rr.items()})
    v1 = rr.get("source_first", 0) > 0.99           # source present -> recovers
    v2 = rr.get("lossy", 1) < 0.01                   # only drift -> never the truth
    v3 = rr.get("lossy_padded", 1) < 0.01
    v4 = rr.get("blank", 1) < 0.01                   # no value -> never the truth
    print(f"V1 source_first recovers (source in note): {'PASS' if v1 else 'FAIL'}")
    print(f"V2 lossy never returns truth:              {'PASS' if v2 else 'FAIL'}")
    print(f"V3 lossy_padded never returns truth:       {'PASS' if v3 else 'FAIL'}")
    print(f"V4 blank never returns truth:              {'PASS' if v4 else 'FAIL'}")
    ok = v1 and v2 and v3 and v4
    print(f"\n{'ALL PASS' if ok else 'FAILED'} ({sum([v1,v2,v3,v4])}/4)")
    return 0 if ok else 1


def summarize(path, policies, arm):
    cell = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r["arm"] == arm:
                cell[r["policy"]].append(1 if r["correct"] else 0)
    print(f"\nMultiWOZ reclaim ({arm} arm):")
    for p in policies:
        xs = cell.get(p, [])
        if xs:
            print(f"  {p:14} RR = {sum(xs)/len(xs):.2f}  (n={len(xs)})")


if __name__ == "__main__":
    raise SystemExit(main())
