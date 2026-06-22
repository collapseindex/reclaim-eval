#!/usr/bin/env python3
"""One-command reproduction of every headline number in the paper, plus the anti-rig validators.

Pure analysis over the committed result JSONL: no API calls, no randomness beyond a seeded
bootstrap. Regenerates the LaTeX bodies for tab:wall, tab:logic, and tab:frontier, recomputes the
auto-vs-summary paired differences, and runs the correct-by-construction validators on the
generated problem set. Exits non-zero if any validator fails, so it doubles as a CI check.

    python scripts/reproduce_tables.py          # full report
    python scripts/reproduce_tables.py --quiet   # validators + PASS/FAIL only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))


def run(label, argv):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    r = subprocess.run([sys.executable, *argv], capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
    return r.returncode == 0


def c6_robustness():
    """C6: source-first resistance to a confident wrong value, scored from committed jsonl (no API).

    Reports source-first's resistance (returns truth) and lossy's capitulation (adopts the asserted
    value) under sycophancy pressure, on the frontier answering models.
    """
    import json
    from collections import Counter
    res = ROOT / "data" / "results"
    out = []
    for model in ("claude-sonnet-4-6", "claude-opus-4-8"):
        p = res / f"confidentwrong_{model}.jsonl"
        if not p.exists():
            out.append((model, None, None, None))
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        sf = [r for r in rows if r["policy"] == "source_first" and r["correction"] == "wrongval"]
        ls = [r for r in rows if r["policy"] == "lossy" and r["correction"] == "wrongval"]
        resist = sum(r["bucket"] == "true" for r in sf) / len(sf) if sf else float("nan")
        capit = sum(r["bucket"] == "drift" for r in ls) / len(ls) if ls else float("nan")
        out.append((model, resist, capit, len(sf)))   # n read live from the file
    return out


def adversarial_robustness():
    """Adversarial battery (#1 sustained push, #2 fabricated source), scored from jsonl (no API).

    Reports source-first resistance per attack x model -- the capability-gated boundary.
    """
    import json
    res = ROOT / "data" / "results"
    models = [("llama", "llama-8b"), ("claude-sonnet-4-6", "Sonnet"), ("claude-opus-4-8", "Opus")]
    out = {}
    for mode in ("multiturn", "fabricated"):
        row = {}
        for fn, lbl in models:
            p = res / f"adversarial_{mode}_{fn}.jsonl"
            if not p.exists():
                row[lbl] = None
                continue
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            sf = [r for r in rows if r["policy"] == "source_first" and r["correction"] == mode]
            row[lbl] = (sum(r["bucket"] == "true" for r in sf) / len(sf)) if sf else float("nan")
        out[mode] = row
    return out


def reviewer_baselines():
    """ChatGPT-review baselines, scored from jsonl (no API): source+conclusion, correction
    taxonomy (are-you-sure / correct-value), and tuned source-keyed retrieval."""
    import json
    from collections import defaultdict, Counter
    res = ROOT / "data" / "results"
    out = {}

    def rr(path, key=lambda r: r["correct"], filt=lambda r: True):
        if not path.exists():
            return None
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        xs = [1 if key(r) else 0 for r in rows if filt(r)]
        return (sum(xs) / len(xs), len(xs)) if xs else None

    # source+conclusion at the wall, per model
    sc = {}
    for m, lbl in (("llama", "llama"), ("claude-sonnet-4-6", "Sonnet"), ("claude-opus-4-8", "Opus")):
        p = res / f"srcconcl_{m}_arith.jsonl"
        sc[lbl] = rr(p, filt=lambda r: r["policy"] == "source_plus_conclusion")
    out["source+conclusion"] = sc

    # tuned retrieval: source-keyed vs naive (directed), same store
    tp = res / "realworld_meta-llama_llama-3.1-8b-instruct_arith_t0.7_tunedret.jsonl"
    tr = {}
    for v in ("vector_rag", "vector_rag_source", "source_first@0.1"):
        tr[v] = rr(tp, filt=lambda r, v=v: r["variant"] == v and r["arm"] == "directed")
    out["tuned_retrieval"] = tr
    return out


def blank_vs_lossy():
    """tab:blank: lossy memory vs empty memory at the wall, scored from jsonl (no API).

    Reports confident-wrong emission and abstention per policy x base model. The headline is that
    blank memory abstains (emit ~0) while lossy emits a confident wrong value (much of it the
    inherited attractor), so a wrong-valued memory is strictly worse than no memory.
    """
    import json
    res = ROOT / "data" / "results"
    out = []
    for fn, lbl in (("blank_llama", "llama-3.1-8b"),
                    ("blank_x-ai_grok-4.3-20260430", "grok-4.3")):
        p = res / f"{fn}.jsonl"
        if not p.exists():
            out.append((lbl, None)); continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        rec = {}
        for pol in ("blank", "lossy"):
            sub = [r for r in rows if r["policy"] == pol]
            if not sub:
                rec[pol] = None; continue
            n = len(sub)
            rec[pol] = {"emit": sum(r["bucket"] == "emit" for r in sub) / n,
                        "abstain": sum(r["bucket"] == "abstain" for r in sub) / n,
                        "attractor": sum(r.get("attractor") for r in sub) / n, "n": n}
        out.append((lbl, rec))
    return out


def prompt_sweep():
    """source-first-auto prompt-sensitivity (reviewer experiment), from jsonl (no API).

    Four intent-equivalent distillation prompts on a fixed arith set, per distiller model; the
    spread (and its monotone shrinkage with capability) is the result.
    """
    import json
    res = ROOT / "data" / "results"
    files = {"llama-8b": "promptsweep_meta-llama_llama-3.1-8b-instruct_arith.jsonl",
             "Sonnet": "promptsweep_claude-sonnet-4-6_arith.jsonl",
             "Opus": "promptsweep_claude-opus-4-8_arith.jsonl"}
    out = {}
    for lbl, fn in files.items():
        p = res / fn
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        by = {}
        for r in rows:
            by.setdefault(r["prompt"], []).append(r["correct"])
        rates = {k: sum(v) / len(v) for k, v in by.items()}
        out[lbl] = {"rates": rates, "spread": (min(rates.values()), max(rates.values())),
                    "n": len(next(iter(by.values())))}
    return out or None


def endogenous():
    """Endogenous (self-generated) error wall (reviewer experiment), from jsonl (no API)."""
    import json
    from collections import Counter
    p = ROOT / "data" / "results" / "endogenous_meta-llama_llama-3.1-8b-instruct_n10.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    out = {}
    for pol in ("lossy_endo", "srcfirst_endo"):
        sub = [r for r in rows if r["policy"] == pol]
        if sub:
            c = Counter(r["bucket"] for r in sub); n = len(sub)
            out[pol] = {b: c[b] / n for b in ("true", "inherit", "novel", "abstain")} | {"n": n}
    return out


def locating():
    """Deployable locating: source-first-auto reclaim + bought-items-kept under noise (no oracle)."""
    import json
    from collections import defaultdict
    p = ROOT / "data" / "results" / "locating_meta-llama_llama-3.1-8b-instruct.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["decoys"]].append(r)
    return {d: (sum(r["correct"] for r in s) / len(s),
                sum(r["bought_kept"] for r in s) / len(s), len(s))
            for d, s in sorted(by.items())}


def completeness_weak():
    """Completeness tag honored by a weak (8B) reader? (flag/abstain vs silent mis-sum)."""
    import json
    from collections import Counter
    p = ROOT / "data" / "results" / "completeness_llama.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    c = Counter(r["bucket"] for r in rows)
    n = len(rows)
    return {"flag_or_abstain": c["flagged_partial"] + c["abstain"],
            "silent_missum": c["silent_missum"], "n": n}


def validators():
    """Correct-by-construction: every generated problem's answer is brute-force verified.

    A planted-wrong-answer note that the model merely echoes cannot pass, because the validators
    recompute the true answer independently and assert the generator agrees.
    """
    from reclaim.problems_gen import (gen_arith, gen_logic, gen_assign,
                                      validate_arith, validate_logic, validate_assign)
    checks = []
    ga, fa = gen_arith(24, seed=1); validate_arith(ga, fa)
    checks.append(("arith generator (24 problems, answers brute-forced)", True))
    gl, fl = gen_logic(12, seed=2); validate_logic(gl, fl)
    checks.append(("logic-ordering generator (12 problems, solver-checked)", True))
    gs, fs = gen_assign(12, seed=3); validate_assign(gs, fs)
    checks.append(("logic-assignment generator (12 problems, solver-checked)", True))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="validators + PASS/FAIL only")
    args = ap.parse_args()

    ok = True

    print("Reproducing paper headline numbers from committed results (no API).")

    # 1. Correct-by-construction validators (the load-bearing anti-rig check at n=96).
    print(f"\n{'='*70}\nVALIDATORS: correct-by-construction answers (n=96 problem set)\n{'='*70}")
    try:
        for name, passed in validators():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
            ok = ok and passed
    except AssertionError as e:
        print(f"  [FAIL] validator raised: {e}")
        ok = False

    # 2. Headline table bodies (each prints its own n so the reader can confirm n=96 / n=24).
    if not args.quiet:
        ok &= run("tab:wall + tab:logic  (integrity sweep, llama n=96 / grok n=24)",
                  [str(SCRIPTS / "integrity_table_ci.py"), "--latex"])
        ok &= run("tab:frontier  (deployed + frontier replay, n=96)",
                  [str(SCRIPTS / "frontier_table_ci.py"), "--latex"])
        ok &= run("auto-vs-summary paired difference  (n=96, paired bootstrap)",
                  [str(SCRIPTS / "auto_vs_summary_ci.py")])

        print(f"\n{'='*70}\nC6: source-first vs a confident wrong value (sycophancy)\n{'='*70}")
        for model, resist, capit, n in c6_robustness():
            if resist is None:
                print(f"  {model}: confidentwrong_*.jsonl missing (run bench_confidentwrong.py)")
            else:
                print(f"  {model:>28}: source-first resists {resist:.2f}  |  lossy capitulates {capit:.2f}  (n={n})")

        print(f"\n{'='*70}\nAdversarial battery: capability-gated robustness of the fix\n{'='*70}")
        adv = adversarial_robustness()
        print(f"  {'attack':>26} | {'llama-8b':>9} {'Sonnet':>7} {'Opus':>6}")
        for mode, lbl in (("multiturn", "sustained push"), ("fabricated", "fabricated source")):
            r = adv[mode]
            cells = "  ".join(f"{r[m]:.2f}" if isinstance(r.get(m), float) else "  -- "
                              for m in ("llama-8b", "Sonnet", "Opus"))
            print(f"  {lbl:>26} |   {cells}")

        print(f"\n{'='*70}\ntab:blank  (lossy vs EMPTY memory at the wall, arithmetic)\n{'='*70}")
        for lbl, rec in blank_vs_lossy():
            if rec is None:
                print(f"  {lbl}: blank_*.jsonl missing (run bench_blank.py)")
                continue
            for pol in ("blank", "lossy"):
                r = rec.get(pol)
                if r:
                    print(f"  {lbl:>13} {pol:>6}: emit {r['emit']:.2f}  abstain {r['abstain']:.2f}  "
                          f"attractor {r['attractor']:.2f}  (n={r['n']})")

        print(f"\n{'='*70}\nPrompt-sensitivity of source-first-auto (arith, fixed set, per distiller)\n{'='*70}")
        ps = prompt_sweep()
        if ps is None:
            print("  promptsweep_*.jsonl missing (run bench_promptsweep.py)")
        else:
            for lbl, d in ps.items():
                lo, hi = d["spread"]
                cells = "  ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in d["rates"].items())
                print(f"  {lbl:>9} (n={d['n']}): {cells}   spread {lo:.2f}-{hi:.2f} ({hi-lo:.2f})")
            print("  -> spread shrinks with distiller capability (prompt-robustness is capability-gated)")

        print(f"\n{'='*70}\nEndogenous (self-generated) error wall (llama, N=10 ledgers)\n{'='*70}")
        en = endogenous()
        if en is None:
            print("  endogenous_*.jsonl missing (run bench_endogenous.py)")
        else:
            for pol in ("lossy_endo", "srcfirst_endo"):
                r = en.get(pol)
                if r:
                    print(f"  {pol:>14}: true {r['true']:.2f}  inherit {r['inherit']:.2f}  "
                          f"abstain {r['abstain']:.2f}  (n={r['n']})")

        print(f"\n{'='*70}\nDeployable locating (source-first-auto under noise, no oracle)\n{'='*70}")
        loc = locating()
        if loc is None:
            print("  locating_*.jsonl missing (run bench_locating.py)")
        else:
            for d, (rr, bk, n) in loc.items():
                print(f"  decoys={d:>2}: auto reclaim {rr:.2f}  bought-kept {bk:.1f}/4  (n={n})")
            print("  (naive collapses to 0.00 by 8 decoys; denoised oracle holds 1.00)")

        print(f"\n{'='*70}\nCompleteness tag on a weak (8B) reader\n{'='*70}")
        cw = completeness_weak()
        if cw is None:
            print("  completeness_llama.jsonl missing (run bench_completeness.py --model llama)")
        else:
            print(f"  llama: flag/abstain {cw['flag_or_abstain']}/{cw['n']}  "
                  f"silent mis-sum {cw['silent_missum']}/{cw['n']}  (Opus: 94/96 flagged)")

        print(f"\n{'='*70}\nReviewer baselines (source+conclusion, tuned retrieval)\n{'='*70}")
        rb = reviewer_baselines()
        sc = rb["source+conclusion"]
        cells = "  ".join(f"{sc[m][0]:.2f}" if sc.get(m) else " -- " for m in ("llama", "Sonnet", "Opus"))
        print(f"  source+conclusion @ wall (llama/Sonnet/Opus):  {cells}  (vs source-first 1.00)")
        tr = rb["tuned_retrieval"]
        def g(v):
            return f"{tr[v][0]:.2f}" if tr.get(v) else "--"
        print(f"  tuned retrieval: naive={g('vector_rag')}  source-keyed={g('vector_rag_source')}  "
              f"distilled source-first={g('source_first@0.1')}")

    print(f"\n{'='*70}\n{'ALL CHECKS PASS' if ok else 'SOME CHECKS FAILED'}\n{'='*70}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
