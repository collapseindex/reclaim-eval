#!/usr/bin/env python3
"""Careful pre-flight probe: for every model in the corrected blank-vs-lossy sweep, make a few real
calls and verify three things at once before spending on the full run:
  1. the provider client works and returns parseable content (no silent empty completions);
  2. the FIXED parser + hedge classify put each raw reply in the right bucket (eyeball raw vs label);
  3. the true token usage, so we project the full-run cost per provider against the budgets.

Runs 2 problems x {lossy, blank} = 4 calls per model (cheap). Skips a model whose key is absent.
"""
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
for line in (ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1); os.environ.setdefault(k.strip(), v.strip())
os.environ.setdefault("RECLAIM_EXPAND", "1")   # 32 problems (paper set); we use the first 2

from reclaim.problems import PROBLEMS
from reclaim.experiment import SYSTEM, memory_note, reclaim_cross, _logged_answer
from bench_blank import make_llm, classify, G            # the EXACT code the full run uses

N_FULL = 192   # calls per model in the real run: n=96/cell (32 problems x 3 seeds) x 2 policies

# (provider, key_env, $/Mtok in, $/Mtok out) -- PRICES ARE MY ESTIMATES, confirm against dashboards.
SPEC = {
    "llama":                  ("openrouter", "OPENROUTER_API_KEY", 0.02, 0.03),
    "qwen/qwen-2.5-7b-instruct": ("openrouter", "OPENROUTER_API_KEY", 0.04, 0.10),
    "deepseek/deepseek-chat": ("openrouter", "OPENROUTER_API_KEY", 0.30, 1.10),
    "openai/gpt-4o-mini":     ("openrouter", "OPENROUTER_API_KEY", 0.15, 0.60),
    "claude-sonnet-4-6":      ("anthropic",  "ANTHROPIC_API_KEY",  3.00, 15.00),
    "claude-opus-4-8":        ("anthropic",  "ANTHROPIC_API_KEY", 15.00, 75.00),
    "gpt-5.4":                ("openai",     "OPENAI_API_KEY",     1.25, 10.00),
    "grok-4.3":               ("xai",        "XAI_API_KEY",        3.00, 15.00),
}

BUDGET = {"openrouter": 9.00, "anthropic": 10.13, "openai": 6.44, "xai": 9.28}


def probe_one(model, temp=0.7):
    llm = make_llm(model, temp)
    samples = []
    for prob in PROBLEMS[:2]:
        for pol in ("lossy", "blank"):
            note = memory_note(prob, G, pol)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": note},
                    {"role": "user", "content": reclaim_cross(prob, "directed")}]
            reply = llm.chat(msgs)
            bucket, attr = classify(reply, prob)
            samples.append((pol, prob, bucket, attr, _logged_answer(reply, prob), reply))
    return llm, samples


def main():
    print(f"PROBE: 2 problems x {{lossy,blank}} per model; projecting to {N_FULL} calls/model.\n"
          f"(prices are estimates -- confirm against your dashboards)\n")
    by_provider = {}
    for model, (prov, keyenv, pin, pout) in SPEC.items():
        if not os.environ.get(keyenv):
            print(f"== {model:26} [{prov}]  SKIPPED -- {keyenv} not set\n")
            continue
        try:
            t0 = time.time()
            llm, samples = probe_one(model)
            dt = time.time() - t0
        except Exception as e:
            print(f"== {model:26} [{prov}]  FAILED -- {type(e).__name__}: {str(e)[:150]}\n")
            continue
        tin, tout = llm.prompt_tokens, llm.completion_tokens
        ncalls = len(samples)
        cost_probe = tin / 1e6 * pin + tout / 1e6 * pout
        per_call = cost_probe / max(1, ncalls)
        full = per_call * N_FULL
        by_provider.setdefault(prov, 0.0)
        by_provider[prov] += full
        print(f"== {model:26} [{prov}]  {dt:.0f}s  in/out={tin}/{tout} tok  "
              f"~${per_call:.4f}/call  ->  full ~${full:.2f}")
        for pol, prob, bucket, attr, ans, reply in samples:
            tail = reply.strip().replace(chr(10), " ")[-78:]
            print(f"     {pol:6} [{bucket:7} attr={int(attr)}] ans={ans!r:>8}  correct={prob.correct} "
                  f"drift={prob.drift}")
            print(f"            raw: ...{tail}")
        print()
    print("=== projected full-run cost per provider (vs budget) ===")
    for prov, total in sorted(by_provider.items()):
        b = BUDGET.get(prov, 0)
        flag = "OK" if total <= b else "OVER BUDGET"
        print(f"  {prov:11} ~${total:6.2f}  / ${b:5.2f}   {flag}")
    print(f"\n  grand total ~${sum(by_provider.values()):.2f}")


if __name__ == "__main__":
    main()
