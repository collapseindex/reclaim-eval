#!/usr/bin/env python3
"""Reclaim-window pilot. Dry-run validates the pipeline for free; --real spends API.

    python scripts/run_pilot.py --dry-run          # zero cost, fake LLM with a window
    python scripts/run_pilot.py --real --n 3       # small paid pilot (prints call count)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ── checkpointing (Law #1: never lose paid work; resume skips what is already done) ──
def _ckpt_path(tag: str, mode: str = "fix") -> Path:
    """Stable per-model checkpoint file so a resume run lands in the same place.
    Outputs live under data/results/ (Law #6), not in source."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", tag)
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{mode}_{safe}.jsonl"


def _load_rows(path: Path) -> list:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _append_rows(path: Path, rows: list) -> None:
    """Append-and-flush after each unit of work, so a crash loses at most one unit."""
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.problems import PROBLEMS, TASKS
from reclaim.experiment import (DEPTHS, DISTANCES, INTEGRITY, run_problem,
                                run_problem_distance, run_problem_crosssession,
                                memory_note)
from reclaim.llm import OpenRouterLLM, DryRunLLM

# memory-compression policies compared in --fix. lossy_padded is the budget-match
# control: same content as lossy, padded to source_first's length.
FIX_POLICIES = ("lossy", "lossy_padded", "source_first")


def run_audit(args, problems) -> int:
    """Free, no API: show that lossy / lossy_padded / source_first notes are matched in
    length, so any source_first advantage is content, not text budget."""
    print(f"\nmemory-note length audit  task={args.task}  (chars, ~tokens=chars/4)\n")
    print(f"  {'integrity':>9} {'lossy':>16} {'lossy_padded':>16} {'source_first':>16}")
    for g in INTEGRITY:
        means = []
        for pol in FIX_POLICIES:
            lens = [len(memory_note(p, g, pol)) for p in problems]
            means.append(sum(lens) / len(lens))
        print(f"  {g:>9} {means[0]:>10.0f} ch {means[1]:>13.0f} ch {means[2]:>13.0f} ch")
    print("\n  (lossy_padded should track source_first; lossy may be shorter at low g."
          " The control is lossy_padded vs source_first at equal length.)")
    return 0


def run_fix(args, problems) -> int:
    """Compare lossy / lossy_padded / source_first memory compression across integrity.
    The wall is a choice: keep the source and it stays reclaimable where keeping the
    conclusion walls, and the lossy_padded control shows it is not just text length.

    Checkpointed and resumable: every (seed, problem, policy) unit is written to
    data/results/ as it finishes. Re-running skips finished units (no API spend), so
    `--seeds 1` (smoke) then `--seeds 3` (resume) only pays for the missing seeds.
    """
    from reclaim.experiment import INTEGRITY

    tag = f"{args.model}_{args.task}" if args.real else f"dry-run_{args.task}"
    ckpt = _ckpt_path(tag, "fix")
    all_rows = _load_rows(ckpt)
    done = {(r["seed"], r["pid"], r["policy"]) for r in all_rows}
    resumed = len(done)

    new_calls = 0
    pt = ct = 0
    t0 = time.time()
    units_run = 0
    for s in range(args.seeds):
        llm = None  # build lazily so a fully-resumed seed needs no client/key
        for prob in problems:
            for policy in FIX_POLICIES:
                if (s, prob.pid, policy) in done:
                    continue
                if llm is None:
                    llm = (DryRunLLM(seed=s) if args.dry_run
                           else OpenRouterLLM(model=args.model, temperature=args.temp))
                rows = run_problem_crosssession(llm, prob, policy=policy)
                for r in rows:
                    r["seed"], r["model"] = s, tag
                _append_rows(ckpt, rows)
                all_rows.extend(rows)
                done.add((s, prob.pid, policy))
                units_run += 1
        if llm is not None:
            new_calls += getattr(llm, "calls", 0)
            pt += getattr(llm, "prompt_tokens", 0)
            ct += getattr(llm, "completion_tokens", 0)

    # aggregate over the FULL checkpoint (resumed + new) for the table
    succ = {p: {a: defaultdict(list) for a in ("generic", "directed")}
            for p in FIX_POLICIES}
    for r in all_rows:
        if r["policy"] in succ:
            succ[r["policy"]][r["arm"]][r["integrity"]].append(r["correct"])

    def rate(p, a, g):
        v = succ[p][a][g]
        return (sum(v) / len(v)) if v else float("nan")

    print(f"\n{tag}  task={args.task}  problems={len(problems)} seeds={args.seeds}")
    print(f"  checkpoint: {ckpt}")
    print(f"  units resumed (free): {resumed}   units run this pass: {units_run}"
          f"   new api calls: {new_calls}")
    if args.real and new_calls:
        cost = pt / 1e6 * args.price_in + ct / 1e6 * args.price_out
        print(f"  measured this pass: prompt={pt:,} tok  completion={ct:,} tok  "
              f"= ${cost:.3f}  ({time.time() - t0:.0f}s)")
        if units_run:
            per_unit = cost / units_run
            total_units = args.seeds * len(problems) * len(FIX_POLICIES)
            remaining = max(0, total_units - len(done))
            print(f"  per-unit cost: ${per_unit:.4f}  -> remaining {remaining} units "
                  f"to finish seeds={args.seeds}: ~${per_unit * remaining:.2f}")
    for arm in ("directed", "generic"):
        ref = "" if arm == "directed" else " (reference)"
        print(f"\nreclaim success vs memory integrity, {arm} arm{ref}:")
        print(f"  {'integrity':>9} {'LOSSY':>8} {'LOSSY_PAD':>10} {'SOURCE_1st':>11}")
        for g in INTEGRITY:
            print(f"  {g:>9} {rate('lossy',arm,g):>8.2f} "
                  f"{rate('lossy_padded',arm,g):>10.2f} "
                  f"{rate('source_first',arm,g):>11.2f}")
    print("\n  (the wall is a choice: source_first stays reclaimable at low integrity"
          " where BOTH lossy and the length-matched lossy_padded collapse, so the lever"
          " is the source, not the text budget)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--n", type=int, default=len(PROBLEMS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--degrade", action="store_true",
                    help="vary channel distance (filler) instead of commitment depth")
    ap.add_argument("--cross", action="store_true",
                    help="cross-session: reclaim through a compressed memory (the wall)")
    ap.add_argument("--fix", action="store_true",
                    help="cross-session, comparing lossy vs source_first memory policy")
    ap.add_argument("--probe", action="store_true",
                    help="send ONE real call to confirm the model slug resolves + "
                         "measure tokens-per-call, then exit")
    ap.add_argument("--task", choices=sorted(TASKS), default="arith",
                    help="problem family: arith (numeric) or logic (constraint, non-numeric)")
    ap.add_argument("--audit", action="store_true",
                    help="free: print the lossy/lossy_padded/source_first note lengths "
                         "(budget-match control), then exit")
    ap.add_argument("--price-in", type=float, default=1.25,
                    help="$/1M input tokens for cost reporting (default grok-4.3)")
    ap.add_argument("--price-out", type=float, default=2.50,
                    help="$/1M output tokens for cost reporting (default grok-4.3)")
    args = ap.parse_args()
    problems = TASKS[args.task][: args.n]

    if args.audit:
        return run_audit(args, problems)
    if not (args.dry_run or args.real or args.probe):
        ap.error("pass --dry-run, --real, --probe, or --audit")

    if args.probe:
        llm = OpenRouterLLM(model=args.model, temperature=args.temp)
        reply = llm.chat([{"role": "user", "content":
                           "Reply with exactly: OK. Then 'ANSWER: 2' on a new line."}])
        print(f"slug OK: {args.model}")
        print(f"reply  : {reply!r}")
        print(f"tokens : prompt={llm.prompt_tokens} completion={llm.completion_tokens}")
        cost1 = (llm.prompt_tokens / 1e6 * args.price_in
                 + llm.completion_tokens / 1e6 * args.price_out)
        print(f"this 1 call cost ~${cost1:.5f}  (reasoning tokens, if any, are in "
              f"completion)")
        return 0

    if args.fix:
        return run_fix(args, problems)
    axis = "integrity" if args.cross else ("distance" if args.degrade else "depth")
    levels = INTEGRITY if args.cross else (DISTANCES if args.degrade else DEPTHS)
    runner = (run_problem_crosssession if args.cross else
              run_problem_distance if args.degrade else run_problem)
    succ = {a: defaultdict(list) for a in ("generic", "directed")}
    total_calls = 0
    for s in range(args.seeds):
        llm = (DryRunLLM(seed=s) if args.dry_run
               else OpenRouterLLM(model=args.model, temperature=args.temp))
        for p in problems:
            for row in runner(llm, p):
                succ[row["arm"]][row[axis]].append(row["correct"])
        total_calls += getattr(llm, "calls", 0)

    print(f"\n{'mode':<8} {args.model if args.real else 'dry-run'}   "
          f"problems={len(problems)} seeds={args.seeds}  api_calls={total_calls}\n")
    label = ("reclaim success vs MEMORY INTEGRITY (cross-session; the channel losing "
             "the source)" if args.cross else
             "reclaim success vs CHANNEL DISTANCE (filler turns; the sky diluting)"
             if args.degrade else "reclaim success vs drift depth (the window)")
    print(label + ":")
    print(f"  {axis:>9} {'generic':>9} {'directed':>9}")
    for d in levels:
        g, di = succ["generic"][d], succ["directed"][d]
        gm = (sum(g) / len(g)) if g else float("nan")
        dm = (sum(di) / len(di)) if di else float("nan")
        print(f"  {d:>9} {gm:>9.2f} {dm:>9.2f}")

    def edge(arm):
        for d in levels:
            v = succ[arm][d]
            if v and (sum(v) / len(v)) < 0.5:
                return d
        return None
    ge, de = edge("generic"), edge("directed")
    print(f"\n  generic falls below 0.5 at {axis}: {ge if ge is not None else '> max'}")
    print(f"  directed falls below 0.5 at {axis}: {de if de is not None else '> max'}")
    print("\n  (signal = directed holding past generic; null = they match or"
          " neither falls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
