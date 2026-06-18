#!/usr/bin/env python3
"""Real-world memory benchmark: does an off-the-shelf memory keep the source or wall?

Drops a published memory system (LangChain's genuine ConversationSummaryMemory) into the
exact seam the paper's hand-built memory uses, against the same session-1 trajectory, and
measures Reclaim Rate (RR) under a directed correction. Two hand-built anchors run beside
it on the SAME trajectory so the number is apples-to-apples:
    - lossy@0.1        the paper's predicted wall (keep conclusion, shed source)  -> RR~0
    - source_first@0.1 the paper's fix            (keep source)                   -> RR~1
The empirical question: where does LangChain's summary land?

    python scripts/bench_realworld.py --smoke           # 1 problem, 1 seed (~28 calls)
    python scripts/bench_realworld.py --real --seeds 3  # full, checkpointed/resumable

Honest by construction: nothing is rigged to make the library fail. If its summary keeps
the line-item source, it reclaims and that is the finding; if it keeps only the
conclusion, it walls and that is the finding.
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

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from reclaim.problems import TASKS
from reclaim.experiment import (DEPTHS, SYSTEM, build_trajectory, reclaim_cross,
                                score, memory_note, _logged_answer)
from reclaim.llm import OpenRouterLLM
from reclaim.realworld import BUILDERS

ENVELOPE = "(Memory of an earlier session.) "
ARMS = ("generic", "directed")
# default off-the-shelf systems under test; --systems overrides. The two hand-built anchors
# at the low-integrity wall region always run beside them as the known floor and ceiling.
DEFAULT_SYSTEMS = "langchain_summary,mem0"
ANCHORS = ("lossy@0.1", "source_first@0.1")


def _ckpt_path(tag: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", tag)
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"realworld_{safe}.jsonl"


def _load_rows(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _append_rows(path: Path, rows: list) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        f.flush()


def _numbers(text: str) -> list:
    """Distinct numeric tokens in a text (for the objective confabulation audit)."""
    return sorted({m.group() for m in re.finditer(r"\d+(?:\.\d+)?", text or "")})


def _carryover(variant: str, problem, transcript, model: str) -> str:
    """The session-2 memory text for a variant (with the shared envelope prefix)."""
    if variant in BUILDERS:
        return ENVELOPE + BUILDERS[variant](transcript, model)
    pol, g = variant.split("@")
    return memory_note(problem, float(g), pol)  # already carries its own envelope


def _reclaim(llm, problem, carryover: str, arm: str) -> dict:
    """Session 2: only context is the carry-over memory, then a directed/generic reclaim."""
    base = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": carryover}]
    msgs = base + [{"role": "user", "content": reclaim_cross(problem, arm)}]
    reply = llm.chat(msgs)
    return {"arm": arm, "answer": _logged_answer(reply, problem),
            "correct": score(reply, problem)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 problem, 1 seed: confirm the pipeline + show what the "
                         "summary drops, cheap")
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--task", choices=sorted(TASKS), default="arith")
    ap.add_argument("--n", type=int, default=0, help="problems (0 = all)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=0.02,
                    help="$/1M input tokens (default llama-3.1-8b on OpenRouter)")
    ap.add_argument("--price-out", type=float, default=0.05,
                    help="$/1M output tokens")
    ap.add_argument("--show-memory", action="store_true",
                    help="print each variant's carry-over text (the qualitative result)")
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS,
                    help="comma list of real-world builders to bench "
                         f"(default: {DEFAULT_SYSTEMS}; available: see realworld.BUILDERS)")
    ap.add_argument("--label", default="",
                    help="checkpoint tag suffix, so a different system set is its own file "
                         "(e.g. --label leaderboard)")
    args = ap.parse_args()

    if not (args.real or args.smoke):
        ap.error("pass --smoke (cheap 1-problem run) or --real (full run)")

    from reclaim.realworld import BUILDERS as _B
    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    unknown = [s for s in systems if s not in _B]
    if unknown:
        ap.error(f"unknown systems {unknown}; available: {sorted(_B)}")
    VARIANTS = systems + ANCHORS

    problems = TASKS[args.task]
    if args.smoke:
        problems, args.seeds = problems[:1], 1
        args.show_memory = True
    elif args.n:
        problems = problems[: args.n]

    tag = f"{args.model}_{args.task}_t{args.temp}" + (f"_{args.label}" if args.label else "")
    ckpt = _ckpt_path(tag)
    all_rows = _load_rows(ckpt)
    done = {(r["seed"], r["pid"], r["variant"], r["arm"]) for r in all_rows}
    resumed = len(done)

    t0 = time.time()
    calls = pt = ct = 0
    units_run = 0
    for s in range(args.seeds):
        llm = None
        for prob in problems:
            # what is missing for this (seed, problem)?
            need = [(v, a) for v in VARIANTS for a in ARMS
                    if (s, prob.pid, v, a) not in done]
            if not need:
                continue
            if llm is None:
                llm = OpenRouterLLM(model=args.model, temperature=args.temp)
            transcript = build_trajectory(llm, prob)[max(DEPTHS)]
            # numbers that actually appeared in session 1; anything a memory states that is
            # NOT here was invented during compression (objective confabulation signal).
            src_nums = _numbers(" ".join(m.get("content", "") for m in transcript))
            # build each needed variant's carry-over once, reuse across its arms
            carry = {}
            for v in {v for v, _ in need}:
                carry[v] = _carryover(v, prob, transcript, args.model)
            rows = []
            for v, a in need:
                res = _reclaim(llm, prob, carry[v], a)
                rows.append({"seed": s, "pid": prob.pid, "variant": v,
                             "memory_text": carry[v], "src_nums": src_nums,
                             "model": tag, **res})
                units_run += 1
            _append_rows(ckpt, rows)
            all_rows.extend(rows)
            for r in rows:
                done.add((r["seed"], r["pid"], r["variant"], r["arm"]))
            if args.show_memory:
                print(f"\n[{prob.pid}] carry-over memory each system hands to session 2:")
                for v in VARIANTS:
                    if v in carry:
                        print(f"  --- {v} ---\n    {carry[v]}")
        if llm is not None:
            calls += getattr(llm, "calls", 0)
            pt += getattr(llm, "prompt_tokens", 0)
            ct += getattr(llm, "completion_tokens", 0)

    # aggregate Reclaim Rate over the full checkpoint
    rr = {v: {a: [] for a in ARMS} for v in VARIANTS}
    for r in all_rows:
        if r["variant"] in rr:
            rr[r["variant"]][r["arm"]].append(r["correct"])

    def rate(v, a):
        xs = rr[v][a]
        return (sum(xs) / len(xs)) if xs else float("nan")

    print(f"\n{tag}  task={args.task}  problems={len(problems)} seeds={args.seeds}")
    print(f"  checkpoint: {ckpt}")
    print(f"  units resumed (free): {resumed}   run this pass: {units_run}"
          f"   api calls: {calls}")
    if calls and pt:
        cost = pt / 1e6 * args.price_in + ct / 1e6 * args.price_out
        print(f"  this pass: prompt={pt:,} tok  completion={ct:,} tok  ~${cost:.4f}"
              f"  ({time.time() - t0:.0f}s)")

    print("\nReclaim Rate (fraction recovering the correct answer in session 2):")
    print(f"  {'memory system':>22} {'generic':>9} {'directed':>9}")
    for v in VARIANTS:
        print(f"  {v:>22} {rate(v,'generic'):>9.2f} {rate(v,'directed'):>9.2f}")
    print("\n  source_first is the paper's fix (keep source); lossy is the predicted wall"
          " (keep conclusion).\n  Where langchain_summary lands is the measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
