#!/usr/bin/env python3
"""Human-validated prevalence slice: hand-label the SAME conversations the LLM labelers saw.

The prevalence audit (bench_prevalence.py) labels real conversations COMPACT/DIFFUSE/NONE with two
LLMs (llama, grok), which agree only weakly on messy text (Cohen kappa ~0.15), so the paper reports
the cross-domain ORDERING but explicitly not an absolute coverage number. This tool adds a human
anchor: a stratified slice of real conversations, labeled by YOU blind (no model labels, no domain
shown -- the same information the LLM had), giving a human point estimate per domain and
human-vs-model kappa on identical items.

Labeling and scoring need NO API keys: they reuse the llama labels already saved by the audit
(data/results/prevalence_audit.jsonl) and re-stream only the conversation TEXT from the same corpora
at the same seed (a chars check flags any upstream drift). The grok comparison is an optional cheap
re-run (--grok, ~50 calls at temperature 0) on the exact slice, since the audit only persisted a
handful of per-item grok labels.

Run the same command repeatedly; it resumes where you stopped:

    python scripts/label_prevalence.py             # prep -> label (resumable) -> score
    python scripts/label_prevalence.py --grok       # also label the slice with grok, then score
    python scripts/label_prevalence.py --score       # just recompute the score from saved labels

--seed/--per must match the bench run that wrote prevalence_audit.jsonl (defaults: seed 0, per 100).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))                      # reclaim.llm
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling bench module

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Single source of truth: reuse the audit's loader, rubric, classes, metrics.
from bench_prevalence import (
    load_domain, DOMAINS, CLASSES, RUBRIC, GOLD, CONV_CHARS, cohen_kappa, boot_ci, classify,
)

RESULTS = ROOT / "data" / "results"
KEY = {"c": "COMPACT", "d": "DIFFUSE", "n": "NONE"}
GROK = "x-ai/grok-4.3"


def binary_kappa(a, b):
    """Cohen kappa on the load-bearing split: COMPACT vs not-COMPACT."""
    pairs = [("C" if x == "COMPACT" else "X", "C" if y == "COMPACT" else "X")
             for x, y in zip(a, b) if x and y]
    n = len(pairs)
    if not n:
        return float("nan"), 0
    raw = sum(x == y for x, y in pairs) / n
    pa = sum(x == "C" for x, _ in pairs) / n
    pb = sum(y == "C" for _, y in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((raw - pe) / (1 - pe) if (1 - pe) else float("nan")), n


def slice_path(seed):   return RESULTS / f"prevalence_human_slice_s{seed}.jsonl"
def labels_path(seed):  return RESULTS / f"prevalence_human_labels_s{seed}.jsonl"
def grok_path(seed):    return RESULTS / f"prevalence_human_grok_s{seed}.jsonl"
def md_path(seed):      return RESULTS / f"prevalence_human_s{seed}.md"


RUBRIC_MD = (
    "- **C = COMPACT**: the carried content is a checkable value or fact from a small, identifiable "
    "source in the conversation (a number, total, date, time, name, slot/code value, computed or "
    "looked-up fact). A later correction could recompute or re-verify it.\n"
    "- **D = DIFFUSE**: the carried content is a judgment, synthesis, or recommendation with no "
    "single isolable source to recompute from.\n"
    "- **N = NONE**: no correctable factual answer at all (open chat, brainstorming, creative "
    "writing, roleplay, pure stylistic help).\n"
)


def make_md(items, seed):
    """Write a blind Markdown labeling sheet: fill each **Label:** with C / D / N."""
    p = md_path(seed)
    out = [
        f"# Prevalence audit: human labeling slice (seed {seed})\n\n",
        "Replace each **Label:** with `C`, `D`, or `N`. Judge blind: decide what the assistant's "
        "memory would need to carry forward to stay correctable, and whether that content has a "
        "small identifiable source. Domain and the model labels are deliberately hidden.\n\n",
        RUBRIC_MD,
        "\nWhen finished, save the file and run: `python scripts/label_prevalence.py --score-md`\n",
        "\n---\n\n## Calibration (self-check, 6 items)\n",
    ]
    for k, (_, conv) in enumerate(GOLD, 1):
        out += [f"\n### Cal {k}\n\n```text\n{conv[:CONV_CHARS].rstrip()}\n```\n\n**Label:** \n"]
    out += [f"\n---\n\n## Slice ({len(items)} items)\n"]
    for it in items:
        out += [f"\n### Item {it['pos']}\n\n```text\n{it['text'][:CONV_CHARS].rstrip()}\n```\n\n"
                "**Label:** \n"]
    p.write_text("".join(out), encoding="utf-8")
    print(f"  wrote {len(items)} items + 6 calibration -> {p}")
    print("  fill each **Label:** with C / D / N, save, then:")
    print("    python scripts/label_prevalence.py --score-md")


def _parse_label(s):
    for ch in s.strip().lower():
        if ch in "cdn":
            return {"c": "COMPACT", "d": "DIFFUSE", "n": "NONE"}[ch]
    return None


def parse_md(seed):
    """Read the filled sheet -> ({item_pos: LABEL}, gold_correct_count)."""
    p = md_path(seed)
    if not p.exists():
        sys.exit(f"missing {p}; run --make-md first.")
    human, cal, cur = {}, {}, None
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^###\s+(Item|Cal)\s+(\d+)\s*$", line)
        if m:
            cur = (m.group(1), int(m.group(2)))
            continue
        m2 = re.match(r"^\*\*Label[^:]*:\*\*\s*(.*)$", line)
        if m2 and cur:
            val = _parse_label(m2.group(1))
            kind, idx = cur
            if val is not None:
                (human if kind == "Item" else cal)[idx] = val
            cur = None
    gold = sum(cal.get(k + 1) == GOLD[k][0] for k in range(len(GOLD))) if cal else None
    return human, gold


def build_slice(seed, n, per):
    """Fresh stratified sample from the audit's already-labeled conversations."""
    audit = RESULTS / "prevalence_audit.jsonl"
    if not audit.exists():
        sys.exit(f"missing {audit}; run bench_prevalence.py first.")
    rows = [json.loads(l) for l in audit.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_dom = Counter(r["domain"] for r in rows)
    per_dom = max(1, n // len(DOMAINS))

    rng = random.Random(seed + 7)
    items, cache, drift = [], {}, 0
    for dom in DOMAINS:
        dr = [r for r in rows if r["domain"] == dom and r.get("label")]
        pick = rng.sample(dr, min(per_dom, len(dr)))
        if dom not in cache:
            print(f"  re-streaming {DOMAINS[dom]} (n={by_dom[dom]}) ...")
            cache[dom] = load_domain(dom, per, seed)
        for r in pick:
            txt = cache[dom][r["i"]] if r["i"] < len(cache[dom]) else ""
            if r.get("chars") is not None and abs(len(txt) - r["chars"]) > 2:
                drift += 1
            items.append({"domain": dom, "i": r["i"], "llama": r["label"], "text": txt})
    if drift:
        print(f"  !! WARNING: {drift}/{len(items)} conversations changed length vs the audit "
              f"(upstream dataset drift); human/model items may not be identical.")
    rng.shuffle(items)                      # interleave domains (domain stays hidden anyway)
    for pos, it in enumerate(items):
        it["pos"] = pos
    return items


def ensure_slice(seed, n, per):
    sp = slice_path(seed)
    if sp.exists():
        return [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
    RESULTS.mkdir(parents=True, exist_ok=True)
    items = build_slice(seed, n, per)
    with open(sp, "w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it) + "\n")
    print(f"  slice of {len(items)} items -> {sp}")
    return items


def _load_kv(path, key="human"):
    out = {}
    if path.exists():
        for l in path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                out[r["pos"]] = r[key]
    return out


def append_label(path, pos, val, key="human", extra=None):
    with open(path, "a", encoding="utf-8") as fh:
        rec = {"pos": pos, key: val}
        if extra:
            rec.update(extra)
        fh.write(json.dumps(rec) + "\n")


RUBRIC_SHORT = (
    "  [c] COMPACT  - carried content is a checkable value/fact from a small identifiable source\n"
    "                 (number, total, date, time, name, slot/code value, computed or looked-up fact)\n"
    "  [d] DIFFUSE  - a judgment/synthesis/recommendation with no single isolable source\n"
    "  [n] NONE     - no correctable factual answer at all (chat, brainstorm, creative, roleplay)\n"
    "  [s] skip   [b] back   [q] save & quit"
)


def prompt_one(header, conv):
    print("\n" + "=" * 78)
    print(header)
    print("-" * 78)
    print(conv[:CONV_CHARS])
    print("-" * 78)
    print(RUBRIC_SHORT)
    while True:
        ch = input("  label > ").strip().lower()
        if ch in KEY:
            return KEY[ch]
        if ch in ("s", "b", "q"):
            return ch
        print("  enter c / d / n  (or s skip, b back, q quit)")


def do_gold(seed, done):
    """6-item calibration, parallel to the LLM gold check; blind."""
    if -1 in done:
        return done[-1]
    print("\n### Calibration: 6 unambiguous conversations (parallels the LLM gold check) ###")
    correct = 0
    for k, (truth, conv) in enumerate(GOLD):
        r = prompt_one(f"calibration {k+1}/6", conv)
        if r == "q":
            return None
        r = "NONE" if r in ("s", "b") else r
        correct += (r == truth)
    print(f"  calibration: {correct}/6 match the gold labels.")
    append_label(labels_path(seed), -1, "GOLD", extra={"gold": correct})
    return correct


def label_loop(items, seed):
    lp = labels_path(seed)
    done = _load_kv(lp)
    do_gold(seed, done)
    done = {k: v for k, v in done.items() if k != -1}
    n, pos = len(items), 0
    while pos < n:
        if items[pos]["pos"] in done:
            pos += 1
            continue
        it = items[pos]
        r = prompt_one(f"item {pos+1}/{n}   ({len(done)} done)", it["text"])
        if r == "q":
            print(f"\n  saved {len(done)}/{n}. Run again to resume.")
            return False
        if r == "b":
            pos = max(0, pos - 1)
            done.pop(items[pos]["pos"], None)
            continue
        if r == "s":
            pos += 1
            continue
        append_label(lp, it["pos"], r)
        done[it["pos"]] = r
        pos += 1
    print(f"\n  all {n} items labeled.")
    return True


def run_grok(items, seed):
    """Optional: label the exact slice with grok (temperature 0), resumable."""
    from reclaim.llm import OpenRouterLLM
    gp = grok_path(seed)
    have = _load_kv(gp, "grok")
    todo = [it for it in items if it["pos"] not in have]
    if not todo:
        print(f"  grok already labeled all {len(items)} items."); return
    print(f"  labeling {len(todo)} items with grok ({GROK}) ...")
    llm = OpenRouterLLM(model=GROK, temperature=0.0)
    for k, it in enumerate(todo, 1):
        lab = classify(llm, it["text"])
        append_label(gp, it["pos"], lab, key="grok")
        if k % 10 == 0:
            print(f"    {k}/{len(todo)}")
    print(f"  grok labels -> {gp}")


def score(items, seed, human=None, gold=None):
    if human is None:                       # default: load from the interactive labels file
        human = _load_kv(labels_path(seed))
        human.pop(-1, None)
        lp = labels_path(seed)
        if lp.exists():
            for l in lp.read_text(encoding="utf-8").splitlines():
                if l.strip() and json.loads(l).get("pos") == -1:
                    gold = json.loads(l).get("gold")
    grok = _load_kv(grok_path(seed), "grok")
    by_pos = {it["pos"]: it for it in items}
    labeled = [(by_pos[p], h) for p, h in human.items() if p in by_pos]
    if not labeled:
        print("  nothing labeled yet."); return
    H = [h for _, h in labeled]
    L = [it["llama"] for it, _ in labeled]
    G = [grok.get(it["pos"]) for it, _ in labeled]

    print(f"\n=== Human-validated prevalence slice (n={len(labeled)} labeled, seed {seed}) ===")
    if gold is not None:
        print(f"  human gold calibration: {gold}/6")
    print(f"\n  per-domain COMPACT fraction (human point estimate, 95% bootstrap CI):")
    summary = {"n": len(labeled), "seed": seed, "gold": gold, "per_domain": {}, "kappa": {}}
    for dom, name in DOMAINS.items():
        dl = [h for it, h in labeled if it["domain"] == dom]
        if not dl:
            continue
        f = sum(x == "COMPACT" for x in dl) / len(dl)
        lo, hi = boot_ci(dl, "COMPACT", seed=seed)
        print(f"    {name:<34} {f:.2f} [{lo:.2f},{hi:.2f}]   (n={len(dl)})")
        summary["per_domain"][dom] = {"compact": round(f, 3), "ci": [round(lo, 3), round(hi, 3)],
                                      "n": len(dl)}

    def row(tag, a, b):
        k3, n3 = cohen_kappa(a, b)
        kb, _ = binary_kappa(a, b)
        print(f"    {tag:<26} 3-way kappa={k3:.2f}   binary(COMPACT-vs-rest) kappa={kb:.2f}   (n={n3})")
        return {"three_way": round(k3, 3), "binary": round(kb, 3), "n": n3}

    print(f"\n  agreement on the SAME {len(labeled)} items:")
    summary["kappa"]["human_vs_llama"] = row("human vs llama", H, L)
    if any(G):
        summary["kappa"]["human_vs_grok"] = row("human vs grok", H, G)
        summary["kappa"]["llama_vs_grok"] = row("llama vs grok (models)", L, G)
        print("  (the model-vs-model row is the ~0.15 the paper reports; human is the anchor)")
    else:
        print("  (run with --grok to add human-vs-grok and a fresh model-vs-model row)")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outp = RESULTS / f"{ts}_prevalence_human_summary_s{seed}.json"
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  summary -> {outp}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=51, help="slice size (split evenly across 3 domains)")
    ap.add_argument("--per", type=int, default=100, help="per-domain count the audit used")
    ap.add_argument("--grok", action="store_true", help="also label the slice with grok (needs API key)")
    ap.add_argument("--make-md", action="store_true", help="write a Markdown sheet to fill, instead of interactive")
    ap.add_argument("--score-md", action="store_true", help="score from the filled Markdown sheet")
    ap.add_argument("--score", action="store_true", help="only recompute the score from saved labels")
    args = ap.parse_args()

    items = ensure_slice(args.seed, args.n, args.per)
    if args.make_md:
        make_md(items, args.seed)
        return 0
    if args.score_md:
        if args.grok:
            run_grok(items, args.seed)
        human, gold = parse_md(args.seed)
        msg = f"  parsed {len(human)}/{len(items)} item labels from {md_path(args.seed).name}"
        print(msg + (f", calibration {gold}/6" if gold is not None else ""))
        score(items, args.seed, human, gold)
        return 0
    if args.score:
        score(items, args.seed)
        return 0
    if args.grok:
        run_grok(items, args.seed)
    try:
        finished = label_loop(items, args.seed)
    except (KeyboardInterrupt, EOFError):
        print("\n  interrupted; progress saved. Run again to resume.")
        return 0
    if finished:
        score(items, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
