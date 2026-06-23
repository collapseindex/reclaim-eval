#!/usr/bin/env python3
"""Prevalence audit: how often does real assistant memory carry a COMPACT, checkable source?

The paper scopes source-first to the compact-identifiable-source regime and names, as the open
empirical question, what FRACTION of real assistant memory lives there (vs. diffuse evidence with no
isolable source). This bench estimates that fraction across three real corpora that span the
deployment spread:

  - general assistant chat      (HuggingFaceH4/no_robots)    -> expected diffuse-leaning
  - tool / function-calling     (glaiveai/glaive-function-calling-v2)
  - agentic task trajectories   (THUDM/AgentInstruct: os/db/webshop)

Each sampled conversation is classified by an LLM against a one-paragraph rubric into COMPACT /
DIFFUSE / NONE: what an assistant's memory would need to carry forward to stay correctable, and
whether that content has a small identifiable source (COMPACT), is a diffuse judgment (DIFFUSE), or
is not a correctable factual answer at all (NONE). We report the fraction per domain, so the result
is the domain SPREAD, not one misleading global number: the §7 claim is that the high-stakes/agentic
regime skews compact while open chat skews diffuse.

Honesty rails (his style, each can fail):
  - a 6-item GOLD set of unambiguous conversations the classifier must label correctly (>=5/6),
    else the labels are noise and we do not trust the run;
  - a second-model AGREEMENT pass on a subset, reported as model-vs-model Cohen's kappa (NOT a human
    gold standard -- we are an LLM labeling LLM-readable text, and we say so).

    python scripts/bench_prevalence.py --probe          # access + cost check
    python scripts/bench_prevalence.py --per-domain 100 # full run, one figure
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import warnings
warnings.filterwarnings("ignore")

from reclaim.llm import OpenRouterLLM, AnthropicLLM

CLASSES = ("COMPACT", "DIFFUSE", "NONE")
CONV_CHARS = 1600  # truncate each conversation; the task/topic is set early in all three corpora

RUBRIC = (
    "You are auditing what an AI assistant's MEMORY would carry forward from a conversation, to "
    "estimate how often that carried content has a compact, checkable source.\n\n"
    "Decide what the assistant would need to remember to stay useful and correctable later, then "
    "classify the conversation as exactly one of:\n"
    "- COMPACT: the key carried content is a checkable value or fact derivable from a small, "
    "identifiable source present in the conversation: a number, total, date, time, name, slot "
    "value, code value, computed result, or a specific looked-up fact. A later correction could "
    "recompute or re-verify it from that source.\n"
    "- DIFFUSE: the key carried content is a judgment, explanation, synthesis, recommendation, or "
    "qualitative conclusion whose support is spread across many turns or external knowledge, with "
    "no single isolable source to recompute from.\n"
    "- NONE: no factual, correctable answer is carried at all: open-ended chat, brainstorming, "
    "creative writing, roleplay, or pure stylistic help, where nothing could later be 'wrong' and "
    "need correcting.\n\n"
    "Answer with exactly one line: 'ANSWER: COMPACT' or 'ANSWER: DIFFUSE' or 'ANSWER: NONE'.\n\n"
    "Conversation:\n{conv}\n"
)

# 6 unambiguous conversations: the classifier must get >=5/6 or the run's labels are noise.
GOLD = [
    ("COMPACT", "user: I bought 7 notebooks at $4 and 9 pens at $2. assistant: Your pre-tax total "
                "is $46. user: great, remember that total for my expense report."),
    ("COMPACT", "user: Book the train leaving Cambridge at 16:15 to London. assistant: Booked, "
                "reference TR4321, departing 16:15. user: thanks, hold that departure time."),
    ("COMPACT", "user: When was the Eiffel Tower completed? assistant: It was completed in 1889. "
                "user: note that date for my quiz."),
    ("DIFFUSE", "user: Should I take the job offer or stay at my current company? assistant: It "
                "depends on growth, comp, and your risk tolerance; on balance leaning toward the "
                "offer because... [long qualitative weighing]. user: ok, remember your advice."),
    ("NONE",    "user: write me a short whimsical poem about a cat. assistant: [a whimsical poem]. "
                "user: cute, thanks."),
    ("NONE",    "user: let's roleplay, you are a pirate captain. assistant: Arr, welcome aboard! "
                "user: haha nice."),
]


def parse_class(text: str):
    if not text:
        return None
    up = text.upper()
    import re
    m = re.findall(r"ANSWER:\s*\*{0,2}(COMPACT|DIFFUSE|NONE)", up)
    if m:
        return m[-1]
    hits = [c for c in CLASSES if c in up]
    return hits[0] if len(hits) == 1 else None


def classify(llm, conv: str):
    reply = llm.chat([{"role": "user", "content": RUBRIC.format(conv=conv[:CONV_CHARS])}])
    return parse_class(reply)


# ---- corpus loaders: each yields a plain conversation string ----
def _join_turns(turns, who_key="from", val_key="value"):
    out = []
    for t in turns:
        who = (t.get(who_key) or "?")
        val = (t.get(val_key) or "")
        if val:
            out.append(f"{who}: {val}")
    return "\n".join(out)


def load_domain(domain: str, n: int, seed: int):
    from datasets import load_dataset
    rng = random.Random(seed)
    rows = []
    if domain == "chat":
        ds = load_dataset("HuggingFaceH4/no_robots", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1000)
        for r in ds:
            txt = _join_turns(r.get("messages") or [], who_key="role", val_key="content")
            if len(txt) > 80:
                rows.append(txt)
            if len(rows) >= n:
                break
    elif domain == "tool":
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1000)
        for r in ds:
            txt = (r.get("chat") or "").strip()
            if len(txt) > 80:
                rows.append(txt)
            if len(rows) >= n:
                break
    elif domain == "agentic":
        # blend three agentic task families so it is not one environment's quirk
        per = max(1, n // 3)
        for split in ("os", "db", "webshop"):
            ds = load_dataset("THUDM/AgentInstruct", split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=200)
            got = 0
            for r in ds:
                txt = _join_turns(r.get("conversations") or [])
                if len(txt) > 80:
                    rows.append(txt); got += 1
                if got >= per or len(rows) >= n:
                    break
            if len(rows) >= n:
                break
    else:
        raise ValueError(domain)
    rng.shuffle(rows)
    return rows[:n]


DOMAINS = {"chat": "general assistant (no_robots)",
           "tool": "tool / function-calling (glaive)",
           "agentic": "agentic traces (AgentInstruct)"}


def boot_ci(labels, target, iters=5000, seed=0):
    """Bootstrap 95% CI for the fraction of `labels` equal to `target`."""
    rng = random.Random(seed)
    n = len(labels)
    if n == 0:
        return (float("nan"), float("nan"))
    ind = [1 if x == target else 0 for x in labels]
    samp = []
    for _ in range(iters):
        s = sum(ind[rng.randrange(n)] for _ in range(n)) / n
        samp.append(s)
    samp.sort()
    return (samp[int(0.025 * iters)], samp[int(0.975 * iters)])


def cohen_kappa(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x and y]
    if not pairs:
        return float("nan"), 0
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in CLASSES)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else float("nan")
    return kappa, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--model2", default="x-ai/grok-4.3", help="agreement check (different vendor)")
    ap.add_argument("--per-domain", type=int, default=100)
    ap.add_argument("--agree", type=int, default=50, help="subset size for model-vs-model kappa")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=0.02)
    ap.add_argument("--price-out", type=float, default=0.05)
    ap.add_argument("--probe", action="store_true", help="gold + 3/domain, then cost estimate")
    args = ap.parse_args()

    llm = OpenRouterLLM(model=args.model, temperature=args.temp)
    out = ROOT / "data" / "results" / "prevalence_audit.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- gold validator (must pass) ----
    gold_pred = [classify(llm, conv) for _, conv in GOLD]
    gold_ok = sum(p == g for (g, _), p in zip(GOLD, gold_pred))
    print(f"\nGOLD validator: {gold_ok}/{len(GOLD)} correct "
          f"({'PASS' if gold_ok >= 5 else 'FAIL'})")
    for (g, _), p in zip(GOLD, gold_pred):
        print(f"    truth={g:<8} pred={p}")

    per = 3 if args.probe else args.per_domain
    rows = []
    for dom in DOMAINS:
        convs = load_domain(dom, per, args.seed)
        for i, conv in enumerate(convs):
            c = classify(llm, conv)
            rows.append({"domain": dom, "i": i, "label": c, "chars": len(conv)})
        labs = [r["label"] for r in rows if r["domain"] == dom]
        print(f"  loaded+classified {dom}: {len(labs)} convs")

    cost = llm.prompt_tokens / 1e6 * args.price_in + llm.completion_tokens / 1e6 * args.price_out
    dt = time.time() - t0

    if args.probe:
        full_cells = len(GOLD) + len(DOMAINS) * args.per_domain
        per_cell = cost / max(1, len(GOLD) + len(DOMAINS) * per)
        print(f"\nPROBE ok ({dt:.0f}s, ${cost:.4f}). Full ~{full_cells} classifications "
              f"+ {args.agree} agreement -> ~${per_cell*full_cells*1.4:.2f}")
        for r in rows:
            print(f"    {r['domain']:>8}: {r['label']}")
        return 0

    if gold_ok < 5:
        print("\n!! GOLD FAILED: labels are untrustworthy; not reporting fractions.")
        return 1

    # ---- second-model agreement (honest: model-vs-model, not human gold) ----
    llm2 = (AnthropicLLM(model=args.model2.split("/")[-1], temperature=args.temp)
            if args.model2.startswith("claude") else
            OpenRouterLLM(model=args.model2, temperature=args.temp))
    rng = random.Random(args.seed + 7)
    sub = rng.sample(rows, min(args.agree, len(rows)))
    # rebuild the conv text for the subset (cheap: re-sample deterministically per domain)
    conv_cache = {dom: load_domain(dom, per, args.seed) for dom in DOMAINS}
    a_lab, b_lab = [], []
    for r in sub:
        conv = conv_cache[r["domain"]][r["i"]]
        a_lab.append(r["label"])
        b_lab.append(classify(llm2, conv))
    kappa, kn = cohen_kappa(a_lab, b_lab)
    raw_agree = sum(x == y for x, y in zip(a_lab, b_lab) if x and y) / max(1, kn)
    # Binary collapse to the only load-bearing class: COMPACT vs not-COMPACT.
    bpairs = [("C" if x == "COMPACT" else "X", "C" if y == "COMPACT" else "X")
              for x, y in zip(a_lab, b_lab) if x and y]
    bn = len(bpairs)
    braw = sum(x == y for x, y in bpairs) / max(1, bn)
    pc_a = sum(x == "C" for x, _ in bpairs) / max(1, bn)
    pc_b = sum(y == "C" for _, y in bpairs) / max(1, bn)
    pe = pc_a * pc_b + (1 - pc_a) * (1 - pc_b)
    bkappa = (braw - pe) / (1 - pe) if (1 - pe) else float("nan")

    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(out.with_name("prevalence_agreement.jsonl"), "w", encoding="utf-8") as fh:
        for r, x, y in zip(sub, a_lab, b_lab):
            fh.write(json.dumps({"domain": r["domain"], "m1": x, "m2": y}) + "\n")

    cost2 = cost + (llm2.prompt_tokens / 1e6 * args.price_in +
                    llm2.completion_tokens / 1e6 * args.price_out)
    print(f"\n=== Prevalence by domain (label of carried content), n={per}/domain ===")
    print(f"  {'domain':<34} {'COMPACT':>16} {'DIFFUSE':>16} {'NONE':>16}")
    frac = {}
    for dom, name in DOMAINS.items():
        labs = [r["label"] for r in rows if r["domain"] == dom and r["label"]]
        n = len(labs); cnt = Counter(labs)
        cells = []
        for cls in CLASSES:
            f = cnt[cls] / n if n else float("nan")
            lo, hi = boot_ci(labs, cls, seed=args.seed)
            cells.append(f"{f:.2f} [{lo:.2f},{hi:.2f}]")
        frac[dom] = {cls: cnt[cls] / n if n else 0.0 for cls in CLASSES}
        print(f"  {name:<34} {cells[0]:>16} {cells[1]:>16} {cells[2]:>16}  (n={n})")

    print(f"\n  model-vs-model agreement ({args.model.split('/')[-1]} vs "
          f"{args.model2.split('/')[-1]}, n={kn}):")
    print(f"    3-way (COMPACT/DIFFUSE/NONE): raw={raw_agree:.2f}, Cohen kappa={kappa:.2f}")
    print(f"    binary (COMPACT vs rest, the load-bearing split): raw={braw:.2f}, "
          f"Cohen kappa={bkappa:.2f}")
    print(f"  (agreement is two LLMs labeling the same text, NOT a human gold standard)")
    print(f"\n  total {len(rows)} classifications, {dt:.0f}s, ${cost2:.2f}")

    # ---- figure: stacked bars per domain ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["agentic", "tool", "chat"]
        names = [DOMAINS[d].split(" (")[0] for d in order]
        comp = [frac[d]["COMPACT"] for d in order]
        diff = [frac[d]["DIFFUSE"] for d in order]
        none = [frac[d]["NONE"] for d in order]
        fig, ax = plt.subplots(figsize=(6.2, 2.6))
        y = range(len(order))
        ax.barh(y, comp, color="#2c7fb8", label="compact source")
        ax.barh(y, diff, left=comp, color="#7fcdbb", label="diffuse")
        ax.barh(y, none, left=[c + d for c, d in zip(comp, diff)], color="#d9d9d9", label="no source")
        ax.set_yticks(list(y)); ax.set_yticklabels(names)
        ax.set_xlim(0, 1); ax.set_xlabel("fraction of sampled conversations")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=8)
        for i, c in enumerate(comp):
            ax.text(c / 2, i, f"{c:.2f}", va="center", ha="center", color="white", fontsize=9)
        fig.tight_layout()
        figp = ROOT / "paper_tmlr" / "figures" / "prevalence.pdf"
        figp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figp, bbox_inches="tight")
        fig.savefig(ROOT / "prevalence.png", dpi=130, bbox_inches="tight")
        print(f"  figure -> {figp}")
    except Exception as e:
        print(f"  (figure skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
