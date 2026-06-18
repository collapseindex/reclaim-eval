#!/usr/bin/env python3
"""Frontier answering-model pass, by replay.

Holds each memory fixed (reusing the exact carry-overs already measured on llama) and swaps
ONLY the session-2 answering model to Claude, then re-scores. This isolates the pitch claim:
the memory is broken; can a *frontier* model recover through it? It adds no memory-construction
cost (no rebuild, no mem0), just one Claude call per stored (system, problem, seed, arm).

    python scripts/bench_claude.py --model claude-sonnet-4-6                 # full board
    python scripts/bench_claude.py --model claude-opus-4-8 --task arith --arm directed
    python scripts/bench_claude.py --probe                                  # 1 call, verify key+model

Reads data/results/realworld_*_t0.7*.jsonl, writes data/results/claude_<model>.jsonl.
"""
from __future__ import annotations

import argparse
import glob
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
from reclaim.experiment import SYSTEM, reclaim_cross, score, _logged_answer
from reclaim.llm import AnthropicLLM

PID2PROB = {p.pid: p for fam in TASKS.values() for p in fam}
SRC_FILES = "data/results/realworld_*_t0.7.jsonl", "data/results/realworld_*_t0.7_leaderboard.jsonl"


def _task_of(fname: str) -> str:
    return "logic" if "_logic_" in fname else "arith"


def load_memories(systems, tasks, arms):
    """Distinct (task, variant, pid, seed, arm) -> memory_text, deduped across files."""
    out = {}
    for pat in SRC_FILES:
        for f in glob.glob(str(ROOT / pat)):
            task = _task_of(Path(f).name)
            if task not in tasks:
                continue
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r["variant"] not in systems or r["arm"] not in arms:
                        continue
                    key = (task, r["variant"], r["pid"], r["seed"], r["arm"])
                    out.setdefault(key, r["memory_text"])
    return out


def _ckpt(model):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"claude_{safe}.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--systems", default="langchain_summary,mem0,vector_rag,"
                                         "source_first_auto,lossy@0.1,source_first@0.1")
    ap.add_argument("--task", choices=["arith", "logic", "both"], default="both")
    ap.add_argument("--arm", choices=["generic", "directed", "both"], default="both")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=3.0, help="$/1M input (sonnet default)")
    ap.add_argument("--price-out", type=float, default=15.0, help="$/1M output")
    ap.add_argument("--probe", action="store_true", help="1 call to verify key+model, exit")
    args = ap.parse_args()

    if args.probe:
        llm = AnthropicLLM(model=args.model, temperature=args.temp)
        reply = llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": "What is 6 times 7? Give 'ANSWER: <n>'."}])
        print(f"model OK: {args.model}")
        print(f"reply: {reply!r}")
        print(f"tokens: in={llm.prompt_tokens} out={llm.completion_tokens}")
        return 0

    systems = tuple(s.strip() for s in args.systems.split(","))
    tasks = ("arith", "logic") if args.task == "both" else (args.task,)
    arms = ("generic", "directed") if args.arm == "both" else (args.arm,)

    mems = load_memories(systems, tasks, arms)
    ckpt = _ckpt(args.model)
    done = set()
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["task"], r["variant"], r["pid"], r["seed"], r["arm"]))
    todo = [(k, m) for k, m in mems.items() if k not in done]
    print(f"memories: {len(mems)}  already done: {len(done)}  to run: {len(todo)}")
    print(f"checkpoint: {ckpt}\nmodel: {args.model}\n")

    llm = AnthropicLLM(model=args.model, temperature=args.temp)
    t0 = time.time()
    with open(ckpt, "a", encoding="utf-8") as out:
        for i, ((task, variant, pid, seed, arm), mem) in enumerate(todo, 1):
            prob = PID2PROB[pid]
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": mem},
                    {"role": "user", "content": reclaim_cross(prob, arm)}]
            reply = llm.chat(msgs)
            row = {"task": task, "variant": variant, "pid": pid, "seed": seed, "arm": arm,
                   "model": args.model, "answer": _logged_answer(reply, prob),
                   "correct": score(reply, prob)}
            out.write(json.dumps(row) + "\n")
            out.flush()
            if i % 24 == 0 or i == len(todo):
                cost = (llm.prompt_tokens / 1e6 * args.price_in
                        + llm.completion_tokens / 1e6 * args.price_out)
                print(f"  {i}/{len(todo)}  calls={llm.calls}  "
                      f"~${cost:.2f}  ({time.time() - t0:.0f}s)")

    # aggregate
    rows = [json.loads(l) for l in open(ckpt, encoding="utf-8") if l.strip()]
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in rows:
        agg[r["task"]][r["variant"]][r["arm"]].append(1 if r["correct"] else 0)
    for task in tasks:
        print(f"\n=== {args.model} reclaim, {task} ===")
        print(f"  {'memory system':>22} {'generic':>9} {'directed':>9}")
        for v in systems:
            g = agg[task][v]["generic"]
            d = agg[task][v]["directed"]
            gm = sum(g) / len(g) if g else float("nan")
            dm = sum(d) / len(d) if d else float("nan")
            print(f"  {v:>22} {gm:>9.2f} {dm:>9.2f}")
    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    print(f"\nthis pass: {llm.calls} calls, in={llm.prompt_tokens} out={llm.completion_tokens} "
          f"tok, ~${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
