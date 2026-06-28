#!/usr/bin/env python3
"""Track 1 (source-discrimination) of the memory-correctability benchmark.

How much does a model's reclaim depend on WHAT the carried memory kept? We carry the same drifted
session forward under four policies and measure judge-free recovery of the true answer:

  blank        : neither source nor conclusion (empty-memory floor).
  lossy        : the stale CONCLUSION only, source dropped (the summary / mem0-style baseline).
  source_first : the recomputable SOURCE, conclusion dropped.
  graph        : source + an explicit derivation recipe (the claim/source/derivation structure).

Two design rules from the benchmark review, both baked in:
  - PROCEDURAL GENERATION (anti-contamination): scenarios come from gen_arith with a held-out seed,
    never the paper's published instances. Judge-free scoring makes fresh instances free.
  - THE METRIC IS THE CONTRAST, not the raw rate. A raw reclaim rate is partly a seed/mix property;
    what varies by model and actually means something is the GAP between policies:
      source_first - lossy   source-discrimination (does the model USE kept source?)
      graph - source_first   does explicit structure help beyond prose source? (the product question)
      source_first - blank   source vs nothing.

  python bench_policies.py --models dry                                  # free smoke
  python bench_policies.py --models gpt4o-mini,llama8,sonnet --n 30 --seed 9000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def load_env():
    """Populate os.environ from ROOT/.env (the reclaim package reads keys from the environment)."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


load_env()

import reclaim.problems as P  # noqa: E402
from reclaim import (  # noqa: E402
    FACTS,
    SYSTEM,
    AnthropicLLM,
    DryRunLLM,
    OpenRouterLLM,
    memory_note,
    reclaim_cross,
    score,
)
from reclaim.experiment import _logged_answer  # noqa: E402  (emitted answer; None => abstained)
from reclaim.problems_gen import gen_arith  # noqa: E402

ALIASES = {
    "gpt4o-mini": "openai/gpt-4o-mini",
    "llama8": "meta-llama/llama-3.1-8b-instruct",
    "llama70": "meta-llama/llama-3.1-70b-instruct",
    "qwen72": "qwen/qwen-2.5-72b-instruct:floor",  # bare id dropped chat-completions on OpenRouter
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
    "opus": "claude-opus-4-8",
    # frontier flagships for the abstain-vs-fabricate sweep
    "gpt5.4-mini": "gpt-5.4-mini",   # OpenAI direct API
    "gpt5.4": "gpt-5.4",             # OpenAI direct API
    "deepseek": "deepseek/deepseek-chat",   # standard chat (v4-pro returns empty content via OpenRouter)
    "glm": "z-ai/glm-4.6",
    "gemini": "google/gemini-2.5-flash",
    "grok": "x-ai/grok-4.3",
    "kimi": "moonshotai/kimi-k2",
    "qwen235": "qwen/qwen3-235b-a22b-2507",
    "mistral": "mistralai/mistral-medium-3",
}

POLICIES = ["blank", "lossy", "source_first", "graph"]


@dataclass
class OpenAILLM:
    """OpenAI direct (gpt-5.x reasoning models): max_completion_tokens + reasoning_effort, no temperature."""

    model: str
    reasoning_effort: str = "low"
    max_tokens: int = 2000
    timeout: int = 120

    def __post_init__(self):
        self.key = os.environ.get("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set (.env)")

    def chat(self, messages):
        body = {"model": self.model, "messages": messages,
                "max_completion_tokens": max(self.max_tokens, 2000),
                "reasoning_effort": self.reasoning_effort}
        for attempt in range(6):
            try:
                r = requests.post("https://api.openai.com/v1/chat/completions", json=body,
                                  timeout=self.timeout, headers={"Authorization": f"Bearer {self.key}"})
                if r.status_code == 200:
                    ch = r.json().get("choices") or []
                    if ch:
                        return (ch[0].get("message") or {}).get("content") or ""
                elif r.status_code not in (429, 500, 502, 503):
                    raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:200]}")
            except requests.RequestException:
                pass
            time.sleep(2 * (attempt + 1))
        raise RuntimeError("OpenAI failed after retries")


def make_client(name):
    if name == "dry":
        return DryRunLLM()
    model = ALIASES.get(name, name)
    if model.startswith("claude"):
        return AnthropicLLM(model=model)
    if model.startswith("gpt-5"):       # OpenAI direct (reasoning models), not OpenRouter's 'openai/...'
        return OpenAILLM(model=model)
    return OpenRouterLLM(model=model)


def graph_note(problem, integrity, facts):
    """Source kept + an explicit derivation recipe: the structured claim/source/derivation object.
    The ablation against source_first asks whether spelling out the derivation helps recompute, or
    whether the model derives it from the source anyway (then graph ties source_first)."""
    return (f"(Memory of an earlier session.) Claim: {problem.ask} (value not retained; the earlier "
            f"figure may be off). Source: {facts[problem.pid]}. Derivation: {problem.ask} = sum of "
            f"(quantity x unit price) across the line items above.")


def reclaim_breakdown(llm, compress, problems, integrity, arm="directed"):
    """Judge-free failure-mode classification of one reclaim attempt under a memory policy.
    Each reply is exactly one of:
      recovered       -- emitted the correct answer (this is reclaim_rate's numerator).
      committed_wrong -- emitted a definite WRONG answer (confident-wrong; usually the stale value).
      abstained       -- emitted no parseable answer (declined to commit).
    Recovery saturates on an easy task; the lossy-vs-blank signal lives in committed_wrong vs abstained
    (lossy commits the stale value, blank has nothing to commit, so it abstains)."""
    n = len(problems)
    rec = com = ab = 0
    for p in problems:
        note = compress(p, integrity)
        if hasattr(llm, "configure"):
            llm.configure(p.drift, p.correct, FACTS.get(p.pid), p.locus)
        reply = llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": note},
                          {"role": "user", "content": reclaim_cross(p, arm)}])
        if score(reply, p):
            rec += 1
        elif _logged_answer(reply, p) is None:
            ab += 1
        else:
            com += 1
    return {"recovered": rec / n, "committed_wrong": com / n, "abstained": ab / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="dry")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9000,
                    help="held-out seed; not the paper's published instances (anti-contamination)")
    ap.add_argument("--integrity", type=float, default=0.1)
    args = ap.parse_args()

    problems, facts = gen_arith(args.n, seed=args.seed)
    P.FACTS.update(facts)  # so memory_note (and reclaim_rate's configure) see the generated source

    models = [m.strip() for m in args.models.split(",")]
    print(f"track 1 (source-discrimination): n={args.n} procedurally generated (seed {args.seed}), "
          f"integrity={args.integrity}\n")
    print(f"{'model':12} " + " ".join(f"{p:>13}" for p in POLICIES) + "   (raw reclaim rate)")

    rows = {}
    for m in models:
        try:
            llm = make_client(m)
            bd = {}
            for pol in ("blank", "lossy", "source_first"):
                bd[pol] = reclaim_breakdown(llm, lambda p, g, _pol=pol: memory_note(p, g, _pol),
                                            problems=problems, integrity=args.integrity)
            bd["graph"] = reclaim_breakdown(llm, lambda p, g: graph_note(p, g, facts),
                                            problems=problems, integrity=args.integrity)
            rows[m] = bd
            print(f"{m:12} " + " ".join(f"{bd[p]['recovered']:>13.2f}" for p in POLICIES))
        except Exception as e:                                   # one bad model must not kill the sweep
            print(f"{m:12}  SKIPPED -- {str(e)[:64]}")

    print("\n=== FAILURE MODE on the no-recovery policies (commit-wrong | abstain) ===")
    print("  recovery ceilings out at 0 for blank and lossy; the signal is HOW each fails.")
    print(f"{'model':12} {'lossy:c-wrong':>14} {'abstain':>8}  {'blank:c-wrong':>14} {'abstain':>8}")
    for m in [m for m in models if m in rows]:
        L, B = rows[m]["lossy"], rows[m]["blank"]
        print(f"{m:12} {L['committed_wrong']:>14.2f} {L['abstained']:>8.2f}  "
              f"{B['committed_wrong']:>14.2f} {B['abstained']:>8.2f}")

    print("\n=== KEY CONTRAST: confident-wrong gap = commit_wrong(lossy) - commit_wrong(blank) ===")
    print(f"{'model':12} {'c-wrong gap':>12}")
    for m in [m for m in models if m in rows]:
        gap = rows[m]["lossy"]["committed_wrong"] - rows[m]["blank"]["committed_wrong"]
        print(f"{m:12} {gap:>12.2f}")

    print("\nreads:")
    print("  c-wrong gap > 0 : lossy commits the STALE value while blank abstains -- 'a lossy memory")
    print("                    is worse than an empty one', now visible where recovery rate is a flat 0")
    print("                    for both. Spread across models = whether capability changes the failure")
    print("                    MODE (commit-wrong vs abstain), not just the recovery rate.")


if __name__ == "__main__":
    main()
