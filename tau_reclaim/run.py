"""Governed-action reclaim on tau-bench: does a lossy memory of a real support case drop the
load-bearing spec and make the exchange uncorrectable (re-asserting a wrong variant), where a
source-first memory recovers and an empty memory merely abstains?

Session 2 only: the model is given the product catalog (shared, recomputable), a compressed memory
(policy) of the earlier session, and a correction (arm). It picks the variant to exchange to. The
action is scored judge-free by tau-bench's DB-state hash. Resumable JSONL.

    python run.py            run/continue the sweep    (MODEL, N_CASES, SEEDS via env)
    python run.py --analyze  aggregate results.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cases import build_cases, catalog_text  # noqa: E402
from memory import memory_note, correction  # noqa: E402
from scorer import classify  # noqa: E402
from reclaim.llm import OpenRouterLLM  # noqa: E402
from clients import make_client  # noqa: E402

# Routing: PROVIDER in {grok, gemini, opus, gpt5} hits a direct frontier API; otherwise the OpenRouter
# OSS path with MODEL (e.g. meta-llama/llama-3.1-8b-instruct). MODEL also overrides a frontier model id.
PROVIDER = os.environ.get("PROVIDER", "").strip().lower()
MODEL = os.environ.get("MODEL", "openai/gpt-4o-mini")
N_CASES = int(os.environ.get("N_CASES", "12"))
POLICIES = ["lossy", "lossy_padded", "source_first", "blank"]
ARMS = ["generic", "directed"]
INTEGRITY = 0.3   # tight-budget regime (the wall regime from the paper)
HERE = os.path.dirname(os.path.abspath(__file__))


def _sanitize(s):
    return s.replace("/", "_").replace(":", "_").replace(".", "")


if PROVIDER:
    _probe, SEEDS_DEFAULT, TAG = make_client(PROVIDER, MODEL if os.environ.get("MODEL") else None)
    MODEL_NAME = _probe.model

    def make_llm():
        c, _, _ = make_client(PROVIDER, MODEL if os.environ.get("MODEL") else None)
        return c
else:
    SEEDS_DEFAULT, TAG, MODEL_NAME = 3, _sanitize(MODEL), MODEL

    def make_llm():
        return OpenRouterLLM(model=MODEL, temperature=0.7)

SEEDS = int(os.environ.get("SEEDS", str(SEEDS_DEFAULT)))
OUT = os.path.join(HERE, f"results_{TAG}.jsonl")

SYSTEM = ("You are a retail support agent resuming an unfinished exchange from an earlier session. "
          "Use ONLY the memory and the catalog below; do not invent item ids. Decide which single "
          "catalog item the product should be exchanged to. Reply with ONLY a JSON object: "
          '{"new_item_id": "<item id from the catalog>"}  or  {"new_item_id": "none"} if the memory '
          "is insufficient to determine it.")


def ask(llm, case, policy, arm):
    user = (f"{catalog_text(case)}\n\n{memory_note(case, INTEGRITY, policy)}\n\n{correction(case, arm)}")
    reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    m = re.search(r'"new_item_id"\s*:\s*"?([0-9]{6,}|none)"?', reply)
    return m.group(1) if m else "none"


def proposed_action(case, chosen):
    if chosen == "none":
        return {"tool": "none"}
    return {"tool": case["tool"], "kwargs": {
        "order_id": case["order_id"], "item_ids": [case["old_item_id"]],
        "new_item_ids": [chosen], "payment_method_id": case["payment_method_id"]}}


def _done():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                r = json.loads(line); keys.add((r["case"], r["policy"], r["arm"], r["seed"]))
            except Exception:
                pass
    return keys


def run():
    cases = build_cases(N_CASES)
    done = _done()
    f = open(OUT, "a", encoding="utf-8")
    lock = threading.Lock()
    cnt = {"n": 0}

    def one(case, policy, arm, seed):
        if (case["name"], policy, arm, seed) in done:
            return
        llm = make_llm()
        try:
            chosen = ask(llm, case, policy, arm)
        except Exception as e:
            print(f"  skip {case['name']}/{policy}/{arm}/{seed}: {str(e)[:70]}"); return
        act = proposed_action(case, chosen)
        outcome = classify(act, setup=[], correct=case["correct"])
        with lock:
            f.write(json.dumps({"case": case["name"], "task_index": case["task_index"],
                                "policy": policy, "arm": arm, "seed": seed, "model": MODEL_NAME,
                                "chosen": chosen, "correct_new": case["correct_new"],
                                "drift_new": case["drift_new"], "outcome": outcome}) + "\n")
            f.flush(); cnt["n"] += 1
            if cnt["n"] % 25 == 0:
                print(f"  ...{cnt['n']} new", flush=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for case in cases:
            for policy in POLICIES:
                for arm in ARMS:
                    for seed in range(SEEDS):
                        jobs.append(ex.submit(one, case, policy, arm, seed))
        for j in jobs:
            j.result()
    f.close()
    print(f"done, {cnt['n']} new rows -> {OUT} ({len(cases)} cases x {len(POLICIES)} pol x {len(ARMS)} arm x {SEEDS} seed)")


def _load_all():
    import glob
    by_model = {}
    for path in sorted(glob.glob(os.path.join(HERE, "results_*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            by_model.setdefault(r["model"], []).append(r)
    return by_model


def _cell(rows, policy, arm=None):
    rs = [r for r in rows if r["policy"] == policy and (arm is None or r["arm"] == arm)]
    n = len(rs) or 1
    return {o: round(sum(r["outcome"] == o for r in rs) / n, 3)
            for o in ("reclaim", "stuck", "abstain", "other")} | {"n": len(rs)}


def analyze():
    by_model = _load_all()
    # detail for the current model, if present
    cur = MODEL_NAME
    if cur in by_model:
        rows = by_model[cur]
        print(f"DETAIL  {cur}  ({len(rows)} rows)")
        print(f"{'policy':>14} {'arm':>9} | {'reclaim':>8} {'stuck':>7} {'abstain':>8} | n")
        for policy in POLICIES:
            for arm in ARMS:
                c = _cell(rows, policy, arm)
                print(f"{policy:>14} {arm:>9} | {c['reclaim']:>8} {c['stuck']:>7} {c['abstain']:>8} | {c['n']}")
        print()
    # cross-model summary: the two load-bearing claims, one row per model
    print("GOVERNED-ACTION RECLAIM on tau-bench  -- cross-model (pooled over arm)")
    print("=" * 78)
    print(f"{'model':>22} | {'sf_recl':>8} {'lossy_recl':>10} | {'lossy_stuck':>11} {'blank_stuck':>11} | n/pol")
    for model in sorted(by_model):
        rows = by_model[model]
        sf, lo, bl = _cell(rows, "source_first"), _cell(rows, "lossy"), _cell(rows, "blank")
        print(f"{model:>22} | {sf['reclaim']:>8} {lo['reclaim']:>10} | "
              f"{lo['stuck']:>11} {bl['stuck']:>11} | {lo['n']}")
    print("\nWALL = sf_recl >> lossy_recl.  WORSE-THAN-EMPTY = lossy_stuck >> blank_stuck.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze()
    else:
        run(); print(); analyze()
