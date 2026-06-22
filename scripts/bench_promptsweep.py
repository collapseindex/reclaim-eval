#!/usr/bin/env python3
"""Prompt-sensitivity of source-first-auto: does the deployable number ride on one prompt wording?

The paper's deployable fix (source-first-auto) compresses a transcript toward its recomputable
source via ONE distillation prompt. A reviewer's pointed objection: the whole thesis is that the
compression step's content decides everything, so the wording of OUR compression prompt is the most
obvious thing to sweep. This bench swaps the distillation prompt for several intent-preserving
rewordings and re-measures directed reclaim on arithmetic. A tight cluster hardens the deployable
0.49-0.88; a wide swing is the honest fragility finding.

    python scripts/bench_promptsweep.py --probe
    python scripts/bench_promptsweep.py            # llama, arith, 32 problems, directed, temp 0
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

from reclaim.problems import TASKS
from reclaim.experiment import (DEPTHS, SYSTEM, build_trajectory, reclaim_cross,
                                score, _logged_answer)
from reclaim.realworld import SOURCE_FIRST_PROMPT
from reclaim.llm import OpenRouterLLM

ENVELOPE = "(Memory of an earlier session.) "  # same framing the hand-built notes carry

# Intent-preserving rewordings of the distillation instruction: each says "keep the source/working
# needed to recompute, drop the concluded answer." v0 is the paper's verbatim prompt.
PROMPTS = {
    "v0_paper": SOURCE_FIRST_PROMPT,
    "v1_terse": (
        "Compress this conversation into a brief note for a future session that may need to fix an "
        "error in it. Record every number, fact, and unit needed to redo the calculation from "
        "scratch. Do not state the final answer as established fact. Be brief.\n\nConversation:\n"
        "{conv}\n\nNote:"),
    "v2_checklist": (
        "Write a short memory note from the conversation below. List all input quantities, prices, "
        "and units exactly as given, so the result can be recomputed later. Omit the concluded "
        "total. Keep it short.\n\nConversation:\n{conv}\n\nNote:"),
    "v3_archivist": (
        "You archive conversations so they can be audited later. Preserve the raw inputs and "
        "working (every figure needed to recompute the outcome) and drop derived conclusions, which "
        "can be recalculated from the inputs. Be concise.\n\nConversation:\n{conv}\n\nArchive note:"),
}


def distill(llm, transcript, prompt_tmpl: str) -> str:
    conv = "\n".join(f'{m["role"]}: {m["content"]}' for m in transcript
                     if m.get("role") in ("user", "assistant"))
    return llm.chat([{"role": "user", "content": prompt_tmpl.format(conv=conv)}])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--task", default="arith")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=0.02)
    ap.add_argument("--price-out", type=float, default=0.05)
    ap.add_argument("--probe", action="store_true", help="1 problem x all prompts, then cost")
    args = ap.parse_args()

    problems = TASKS[args.task]
    if args.probe:
        problems, args.seeds = problems[:1], 1
    out = ROOT / "data" / "results" / f"promptsweep_{args.model.replace('/', '_')}_{args.task}.jsonl"

    llm = OpenRouterLLM(model=args.model, temperature=args.temp)
    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for s in range(args.seeds):
            for prob in problems:
                transcript = build_trajectory(llm, prob)[max(DEPTHS)]
                for name, tmpl in PROMPTS.items():
                    note = ENVELOPE + distill(llm, transcript, tmpl)
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note},
                            {"role": "user", "content": reclaim_cross(prob, "directed")}]
                    reply = llm.chat(msgs)
                    row = {"seed": s, "pid": prob.pid, "prompt": name,
                           "answer": _logged_answer(reply, prob), "correct": score(reply, prob)}
                    fh.write(json.dumps(row) + "\n"); fh.flush()
                    rows.append(row)

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    dt = time.time() - t0
    if args.probe:
        per = cost / max(1, len(rows))
        full = per * len(TASKS[args.task]) * args.seeds * len(PROMPTS)
        print(f"\nPROBE: {len(rows)} prompt(s), {dt:.0f}s, cost ${cost:.4f} -> "
              f"full ({len(TASKS[args.task])*args.seeds*len(PROMPTS)} cells) ~${full:.2f}")
        for r in rows:
            print(f"  {r['prompt']:>14}: correct={r['correct']}  answer={r['answer']!r}")
        return 0

    by = defaultdict(list)
    for r in rows:
        by[r["prompt"]].append(r["correct"])
    print(f"\nsource-first-auto prompt sensitivity, {args.model} {args.task} "
          f"({dt:.0f}s, ~${cost:.2f}):")
    print(f"  {'prompt':>14}  reclaim   n")
    rates = {}
    for name in PROMPTS:
        xs = by.get(name, [])
        r = sum(xs) / len(xs) if xs else float("nan")
        rates[name] = r
        print(f"  {name:>14}  {r:>6.2f}  {len(xs)}")
    vals = [v for v in rates.values() if v == v]
    if vals:
        print(f"\n  spread across prompt wordings: {min(vals):.2f}-{max(vals):.2f} "
              f"(range {max(vals)-min(vals):.2f}); paper's auto-arith on this model is 0.53")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
