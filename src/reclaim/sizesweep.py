"""Source-size decay: the boundary of the source-first law.

The compact-source tasks (2 line items) are the favorable end for source-first: the whole
source always fits the memory budget, so "keep the source" can keep all of it. This module
scales the source. A ledger problem has N line items whose sum is the pre-tax total (still a
deterministic function, so scoring stays objective, no judge). The carried memory has a fixed
character budget B. As N grows past what B can hold, the source-first note can retain only the
first k<N items, and an exact sum needs all N, so its reclaim advantage must decay toward the
lossy floor. Sweeping B shows the crossover moves with the budget: the lever is source
recoverability (does the answer-determining source fit?), not problem size per se.

Everything is reused from the main harness: the carried memory is the only session-2 context
(no session-1 simulation needed, the drift is baked into the note), the correction is the same
directed/generic reclaim, and scoring is the same objective numeric check.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .problems import Problem

# A pool of distinct goods so N-item ledgers read naturally up to MAXN.
NOUNS = [
    "notebooks", "pens", "folders", "markers", "staplers", "erasers", "rulers",
    "binders", "clips", "tape rolls", "scissors", "glue sticks", "crayons", "chalk",
    "envelopes", "stamps", "labels", "pins", "magnets", "batteries", "bulbs", "cables",
    "chargers", "mugs", "spoons", "plates", "bowls", "napkins", "candles", "vases",
    "towels", "brushes", "sponges", "buckets", "gloves", "hooks", "nails", "screws",
    "bolts", "washers",
]
MAXN = len(NOUNS)
ASK = "the total before tax"


@dataclass
class Ledger:
    """A parametric ledger problem: a Problem plus the structured item list the note
    builder needs and the index of the planted-wrong (locus) item."""
    problem: Problem
    items: list[tuple[str, int, int]]   # (noun, price, qty)
    locus_idx: int


def make_ledger(pid_index: int, n: int) -> Ledger:
    """Deterministic N-item ledger for store `pid_index`. The total is the exact sum of
    price*qty; one item (the locus) is given a wrong subtotal to plant the drift."""
    rng = random.Random(1000 + pid_index)
    nouns = rng.sample(NOUNS, n)
    items = [(noun, rng.randint(2, 15), rng.randint(1, 9)) for noun in nouns]
    total = sum(p * q for _, p, q in items)
    locus_idx = n // 2                       # a middle item, so it is neither first nor last
    lnoun, lp, lq = items[locus_idx]
    true_sub = lp * lq
    wrong_sub = true_sub + rng.choice([-9, -7, -5, 5, 7, 9, 11])
    drift = float(total - true_sub + wrong_sub)
    pid = f"ledger{pid_index}_n{n}"
    problem = Problem(
        pid=pid,
        question=(f"A store sells {_phrase(items)}. What is the total before tax?"),
        wrong_premise=f"a note says the {lnoun} come to ${wrong_sub}",
        locus=f"the {lnoun} subtotal",
        correct=float(total),
        drift=drift,
        ask=ASK,
        kind="number",
    )
    return Ledger(problem=problem, items=items, locus_idx=locus_idx)


def _phrase(items) -> str:
    return ", ".join(f"{q} {noun} at ${p} each" for noun, p, q in items)


def source_items_str(items) -> str:
    """The recomputable source, one clause per line item (what source-first tries to keep)."""
    return "; ".join(f"{noun} at ${p} each ({q} bought)" for noun, p, q in items)


PREFIX_SF = "(Memory of an earlier session.) The line items were: "
SUFFIX_SF = f" You were determining {ASK}; the earlier answer may be off."
PAD = (" (Archived session log entry; no further working was preserved with it.)")


def build_note(led: Ledger, budget: int, policy: str):
    """Construct the carried memory at a fixed character budget `budget`.

    Returns (note_text, k_items_retained, locus_retained).

      - source_first: fill the note with as many whole line items as fit in `budget`,
        in order, then drop the (re-derivable) conclusion. Past the budget, k<N and the
        exact sum is unrecoverable.
      - lossy_padded: keep ONLY the conclusion, padded with neutral filler to the same
        length as the source_first note at this (N, budget), so the comparison is
        budget-matched and any source-first advantage is content, not text.
    """
    prob = led.problem
    if policy == "source_first":
        clauses = [f"{noun} at ${p} each ({q} bought)" for noun, p, q in led.items]
        kept, used = [], len(PREFIX_SF) + len(SUFFIX_SF)
        for i, c in enumerate(clauses):
            add = len(c) + (2 if kept else 0)        # "; " separator
            if used + add > budget:
                break
            kept.append(c)
            used += add
        k = len(kept)
        locus_kept = led.locus_idx < k
        note = PREFIX_SF + "; ".join(kept) + "." + SUFFIX_SF
        return note, k, locus_kept

    if policy == "source_first_complete":
        # identical to source_first, but adds an explicit completeness signal: the model is
        # told how many of the original items survived, so it CAN abstain instead of
        # confidently summing a truncated source.
        note, k, locus_kept = build_note(led, budget, "source_first")
        n = len(led.items)
        if k < n:
            marker = (f" [Note: this memory preserved only {k} of the original {n} line "
                      f"items; {n - k} were dropped and are not shown.]")
            note = note + marker
        return note, k, locus_kept

    if policy == "lossy_padded":
        sf_note, _, _ = build_note(led, budget, "source_first")
        target = len(sf_note)
        note = (f"(Memory of an earlier session.) You concluded {ASK} was "
                f"${prob.drift:g}.")
        while len(note) < target:
            note += PAD
        return note, 0, False

    raise ValueError(policy)


# ── noisy source: the answer-determining items are buried among plausible decoys, so the
#    budget can be spent on noise. Tests "keep the RIGHT source", not just "keep the source".
@dataclass
class NoisyLedger:
    problem: Problem
    rows: list[tuple[str, int, int, bool]]   # (noun, price, qty, bought)
    relevant_idx: list[int]                  # positions of the bought (answer-determining) items
    locus_idx: int                           # which relevant row carries the planted error


def make_noisy_ledger(pid_index: int, n_relevant: int, n_decoy: int) -> NoisyLedger:
    """n_relevant bought items determine the total; n_decoy 'considered, not bought' items
    are interleaved as noise. The total is the exact sum over bought items only."""
    rng = random.Random(7000 + pid_index * 97 + n_decoy)
    nouns = rng.sample(NOUNS, n_relevant + n_decoy)
    rel = [(nouns[i], rng.randint(2, 15), rng.randint(1, 9), True) for i in range(n_relevant)]
    dec = [(nouns[n_relevant + j], rng.randint(2, 15), 0, False) for j in range(n_decoy)]
    rows = rel + dec
    rng.shuffle(rows)                                  # interleave noise through the source
    relevant_idx = [i for i, r in enumerate(rows) if r[3]]
    total = sum(p * q for _, p, q, b in rows if b)
    locus_pos = relevant_idx[len(relevant_idx) // 2]
    lnoun, lp, lq, _ = rows[locus_pos]
    wrong = lp * lq + rng.choice([-9, -7, -5, 5, 7, 9, 11])
    drift = float(total - lp * lq + wrong)
    prob = Problem(
        pid=f"noisy{pid_index}_r{n_relevant}_d{n_decoy}",
        question=f"A shopper bought {n_relevant} items (others were considered but not bought).",
        wrong_premise=f"a note says the {lnoun} come to ${wrong}",
        locus=f"the {lnoun} subtotal",
        correct=float(total), drift=drift, ask=ASK, kind="number",
    )
    return NoisyLedger(problem=prob, rows=rows, relevant_idx=relevant_idx, locus_idx=locus_pos)


PREFIX_NOISY = "(Memory of an earlier session.) Items considered: "
SUFFIX_NOISY = f" The {ASK} sums the items marked bought; earlier answer may be off."


def _clause(row) -> str:
    noun, p, q, bought = row
    return (f"{q} {noun} at ${p} each (bought)" if bought
            else f"{noun} at ${p} each (considered, not bought)")


def build_noisy_note(nl: NoisyLedger, budget: int, policy: str):
    """Returns (note, n_relevant_kept, all_relevant_kept).

      - source_first_naive: keep items in positional order up to `budget` (decoys included),
        so noise consumes budget and can crowd the bought items out.
      - source_first_denoised: keep ONLY the bought items (drop decoys), padded to `budget`
        so the comparison is budget-matched; the oracle that identifies the source.
      - lossy_padded: conclusion only, padded to `budget`.
    """
    n_rel = len(nl.relevant_idx)
    if policy == "source_first_naive":
        kept_rows, used = [], len(PREFIX_NOISY) + len(SUFFIX_NOISY)
        for i, row in enumerate(nl.rows):
            c = _clause(row)
            add = len(c) + (2 if kept_rows else 0)
            if used + add > budget:
                break
            kept_rows.append(i)
            used += add
        rel_kept = sum(1 for i in kept_rows if nl.rows[i][3])
        note = PREFIX_NOISY + "; ".join(_clause(nl.rows[i]) for i in kept_rows) + "." + SUFFIX_NOISY
        return note, rel_kept, rel_kept == n_rel

    if policy == "source_first_denoised":
        rel_rows = [nl.rows[i] for i in nl.relevant_idx]
        note = (PREFIX_NOISY + "; ".join(_clause(r) for r in rel_rows) + "." + SUFFIX_NOISY)
        while len(note) < budget:
            note += PAD
        return note, n_rel, True

    if policy == "lossy_padded":
        naive, _, _ = build_noisy_note(nl, budget, "source_first_naive")
        target = len(naive)
        note = (f"(Memory of an earlier session.) You concluded {ASK} was "
                f"${nl.problem.drift:g}.")
        while len(note) < target:
            note += PAD
        return note, 0, False

    raise ValueError(policy)
