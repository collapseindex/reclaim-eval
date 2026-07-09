"""Regenerate the capability inverted-U figure as a WITHIN-arithmetic swept curve.

Holding the carried-memory budget fixed (B=600) and growing the ledger size N drives the
source-first note from full-source (k=N items kept) through partial (k<N) to past-the-cliff, a
graded partial-source regime on a single arithmetic task. Plotting a strong (Opus) and a weak
(llama) reader's directed reclaim against N, the capability gap traces the inverted-U: ~0 at full
source (both reclaim), peaking where the source is partial (the strong reader re-derives from the
fragment, the weak one cannot), and ~0 again past the cliff (both wall). Reads the committed sweep
data; no API. Writes paper_tmlr/figures/inverted_u.pdf.
"""
from __future__ import annotations

import collections
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
FIG = ROOT / "paper_tmlr" / "figures"
BUDGET = 600


def reclaim_by_n(model: str) -> dict[int, float]:
    rows = [json.loads(l) for l in (RES / f"sizesweep_{model}.jsonl").open(encoding="utf-8") if l.strip()]
    d = collections.defaultdict(list)
    for r in rows:
        if r["policy"] == "source_first" and r["budget"] == BUDGET and r.get("arm") == "directed":
            d[r["n"]].append(r["correct"])
    return {n: sum(v) / len(v) for n, v in sorted(d.items())}


def main() -> None:
    weak = reclaim_by_n("llama")
    strong = reclaim_by_n("claude-opus-4-8")
    ns = sorted(set(weak) & set(strong))
    lw = [weak[n] for n in ns]
    st = [strong[n] for n in ns]

    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.fill_between(ns, lw, st, color="0.85", zorder=1, label="capability gap")
    ax.plot(ns, st, "-o", color="#1b6ca8", ms=4, lw=1.8, label="Opus (strong)", zorder=3)
    ax.plot(ns, lw, "-o", color="#c8541a", ms=4, lw=1.8, label="llama-3.1-8b (weak)", zorder=3)
    peak = max(ns, key=lambda n: strong[n] - weak[n])
    ax.annotate(f"gap peaks\n$+{strong[peak]-weak[peak]:.2f}$ at $N{{=}}{peak}$",
                xy=(peak, (strong[peak] + weak[peak]) / 2), xytext=(peak + 0.5, 0.45),
                fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))
    ax.text(ns[0], 1.03, "full source", fontsize=7.5, color="0.4", ha="left")
    ax.text(ns[-1], 0.04, "past the cliff", fontsize=7.5, color="0.4", ha="right")
    ax.set_xlabel("ledger size $N$ (source-first budget fixed at $B{=}600$)")
    ax.set_ylabel("directed reclaim")
    ax.set_ylim(-0.03, 1.12)
    ax.set_xticks(ns)
    ax.legend(fontsize=8, loc="center left", framealpha=0.9)
    ax.grid(axis="y", color="0.92", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "inverted_u.pdf", bbox_inches="tight")
    print("wrote", FIG / "inverted_u.pdf")
    print("gap by N:", {n: round(strong[n] - weak[n], 2) for n in ns})


if __name__ == "__main__":
    main()
