#!/usr/bin/env python3
"""What does a source-less MultiWOZ memory assert at 0.00 accuracy?

The non-abstention finding (Table failmode) on real dialogue. lossy carries the drift value in
the note; blank carries no value. Both score 0.00, but the failure differs: does the model
confidently assert a wrong booking time (inherit / novel) or flag that it cannot tell (abstain)?
Captures the reply and classifies.

    python scripts/bench_multiwoz_failmode.py --model llama
    python scripts/bench_multiwoz_failmode.py --model claude-opus-4-8
"""
from __future__ import annotations

import argparse
import json
import re
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

from reclaim.multiwoz import load_targets, build_note, reclaim_msg, _canon
from reclaim.experiment import SYSTEM
from reclaim.llm import OpenRouterLLM, AnthropicLLM

DATA = ROOT / "data" / "multiwoz" / "dev_001.json"
ABSTAIN = re.compile(r"(can(?:not|'t)|unable|insufficient|not enough|no record|don'?t (?:have|know)"
                     r"|not sure|cannot determine|unclear|no (?:information|time|details)|missing)", re.I)
# STRICT: only a value on the ANSWER line counts as an asserted answer (a time quoted inside a
# refusal must not count). If there is no ANSWER-line time, the model asserted no value.
_ANSLINE = re.compile(r"ANSWER:\s*\*{0,2}\s*(\d{1,2}):(\d{2})\s*(am|pm)?", re.I)


def asserted_time(reply):
    m = _ANSLINE.search(reply or "")
    if not m:
        return None
    h, mi, ap = m.groups()
    h = int(h)
    if ap and ap.lower() == "pm" and h != 12:
        h += 12
    if ap and ap.lower() == "am" and h == 12:
        h = 0
    return f"{h % 24:02d}:{int(mi):02d}"


def classify(reply, true_v, drift_v):
    got = asserted_time(reply)
    if got is None:
        return "abstain"                      # no value on the ANSWER line
    if got == _canon(true_v):
        return "correct"
    if got == _canon(drift_v):
        return "inherit"                      # ANSWER line re-asserts the wrong value from the note
    return "novel"                            # ANSWER line asserts a different wrong time


def make_llm(model):
    if model.startswith("claude"):
        return AnthropicLLM(model=model, temperature=0.0, max_tokens=1000)
    if model in ("llama", "meta-llama/llama-3.1-8b-instruct"):
        return OpenRouterLLM(model="meta-llama/llama-3.1-8b-instruct", temperature=0.7, max_tokens=1000)
    return OpenRouterLLM(model=model, temperature=0.7, max_tokens=1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama")
    ap.add_argument("--n-dialogues", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--policies", default="lossy,blank")
    ap.add_argument("--price-in", type=float, default=15.0)
    ap.add_argument("--price-out", type=float, default=75.0)
    args = ap.parse_args()

    targets = load_targets(DATA, max_n=args.n_dialogues)
    policies = tuple(p.strip() for p in args.policies.split(","))
    llm = make_llm(args.model)
    out = ROOT / "data" / "results" / f"multiwoz_failmode_{args.model.replace('/', '_')}.jsonl"
    rows = []
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as fh:
        for tgt in targets:
            for pol in policies:
                note = build_note(tgt, pol)
                for seed in range(args.seeds):
                    msgs = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": note},
                            {"role": "user", "content": reclaim_msg(tgt, "directed")}]
                    reply = llm.chat(msgs)
                    b = classify(reply, tgt.true_value, tgt.drift_value)
                    flagged = bool(ABSTAIN.search(reply))
                    row = {"dialogue_id": tgt.dialogue_id, "policy": pol, "seed": seed,
                           "bucket": b, "flagged": flagged, "reply": reply}
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    rows.append(row)
        cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out

    print(f"\nMultiWOZ failure mode at the wall, {args.model} ({time.time()-t0:.0f}s):")
    for pol in policies:
        sub = [r for r in rows if r["policy"] == pol]
        c = Counter(r["bucket"] for r in sub)
        n = len(sub)
        assertw = c["inherit"] + c["novel"]
        flagged = sum(1 for r in sub if r["flagged"])
        print(f"  {pol:6} n={n}: " + "  ".join(f"{k}:{c[k]}" for k in ("correct", "inherit", "novel", "abstain")))
        print(f"         -> asserts a WRONG time: {assertw}/{n} ({assertw/n:.0%});  "
              f"any incompleteness flag in text: {flagged}/{n}")
    if args.model.startswith("claude"):
        print(f"  cost ~${cost:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
