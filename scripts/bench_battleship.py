#!/usr/bin/env python3
"""Brittle memory in a sequential agentic task: a real LLM plays Battleship from a compressed memory.

Every turn the model gets a fresh context carrying only the memory of the game so far, compressed
under a policy at a matched budget. source_first keeps the shot coordinates; lossy keeps a prose
summary and drops them. Prediction: at equal budget, source_first sinks the fleet with few wasted
shots while lossy re-fires dead water and stalls, on the SAME model.

    python scripts/bench_battleship.py --dry                      # free validators, no API
    python scripts/bench_battleship.py --model claude-sonnet-4-6 --boards 2   # smoke test
    python scripts/bench_battleship.py --model claude-opus-4-8 --boards 8 --seeds 3
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

from reclaim.battleship import place_fleet, play_game, FakeCommander
from reclaim.llm import OpenRouterLLM, AnthropicLLM

POLICIES = ("source_first", "lossy", "lossy_padded")


def make_llm(model, temp, max_tokens=200):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=temp, max_tokens=max_tokens)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=temp, max_tokens=max_tokens)
    return OpenRouterLLM(model=model, temperature=temp, max_tokens=max_tokens)


def ckpt(model):
    d = ROOT / "data" / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"battleship_{model.replace('/', '_')}.jsonl"


def run_dry(boards, budget, max_turns):
    """The fake plays from the note only: source_first must win without re-fires; lossy must re-fire
    and stall; the padded control must behave like lossy. Each check can fail."""
    agg = defaultdict(list)
    cap = max(max_turns, 90)                              # the naive fake sweeps the board; give it room
    for ships in boards:
        for pol in POLICIES:
            agg[pol].append(play_game(FakeCommander(), ships, pol, budget, cap))
    def rate(pol, f): return sum(f(g) for g in agg[pol]) / len(agg[pol])
    sf_wins   = rate("source_first", lambda g: g.won)
    sf_refire = sum(g.redundant for g in agg["source_first"])
    lo_wins   = rate("lossy", lambda g: g.won)
    lo_refire = sum(g.redundant for g in agg["lossy"])
    lp_refire = sum(g.redundant for g in agg["lossy_padded"])
    v1 = sf_wins > 0.99
    v2 = sf_refire == 0
    v3 = lo_wins < 0.01 and lo_refire > 0
    v4 = lp_refire > 0
    print("\nDry validators (deterministic fake, plays from the note only):")
    print(f"  source_first: win {sf_wins:.2f}, re-fires {sf_refire}")
    print(f"  lossy:        win {lo_wins:.2f}, re-fires {lo_refire}")
    print(f"  lossy_padded: re-fires {lp_refire}")
    for ok, name in [(v1, "V1 source_first wins (source in note)"),
                     (v2, "V2 source_first never re-fires"),
                     (v3, "V3 lossy re-fires and never wins (source absent)"),
                     (v4, "V4 padded control still re-fires (budget != fix)")]:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    ok = v1 and v2 and v3 and v4
    print(f"\n{'ALL PASS' if ok else 'FAILED'} ({sum([v1,v2,v3,v4])}/4)")
    return 0 if ok else 1


def summarize(path, policies):
    cell = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            cell[r["policy"]].append(r)
    print("\nBattleship reclaim (per policy):")
    print(f"  {'policy':>13}  {'win':>5}  {'turns(win)':>10}  {'re-fire/turn':>12}  {'n':>3}")
    for p in policies:
        xs = cell.get(p, [])
        if not xs:
            continue
        win = sum(g["won"] for g in xs) / len(xs)
        wins = [g for g in xs if g["won"]]
        mt = sum(g["turns"] for g in wins) / len(wins) if wins else float("nan")
        rr = sum(g["redundant"] / max(1, g["turns"]) for g in xs) / len(xs)
        print(f"  {p:>13}  {win:>5.2f}  {mt:>10.1f}  {rr:>12.2f}  {len(xs):>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--boards", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--max-turns", type=int, default=50)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--price-in", type=float, default=3.0)
    ap.add_argument("--price-out", type=float, default=15.0)
    args = ap.parse_args()

    boards = [place_fleet(b) for b in range(args.boards)]
    policies = tuple(p.strip() for p in args.policies.split(","))

    if args.dry:
        return run_dry(boards, args.budget, args.max_turns)

    model = args.model
    path = ckpt(model)
    done = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["board"], r["policy"], r["seed"]))

    temp = 0.0 if model.startswith("claude") else args.temp
    llm = make_llm(model, temp)
    todo = [(b, pol, s) for b in range(args.boards) for pol in policies for s in range(args.seeds)
            if (b, pol, s) not in done]
    print(f"model={model}  boards={args.boards}  policies={policies}  games to run={len(todo)}")
    t0 = time.time()
    with open(path, "a", encoding="utf-8") as out:
        for i, (b, pol, s) in enumerate(todo, 1):
            g = play_game(llm, boards[b], pol, args.budget, args.max_turns)
            out.write(json.dumps({"board": b, "policy": pol, "seed": s, "model": model,
                                  "won": g.won, "turns": g.turns, "distinct": g.distinct,
                                  "redundant": g.redundant, "invalid": g.invalid, "hits": g.hits}) + "\n")
            out.flush()
            cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
            tag = f"~${cost:.2f}" if model.startswith("claude") else f"{llm.calls} calls"
            print(f"  {i}/{len(todo)}  b{b} {pol} s{s}: won={g.won} turns={g.turns} "
                  f"refire={g.redundant}  [{tag}, {time.time()-t0:.0f}s]")
    summarize(path, policies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
