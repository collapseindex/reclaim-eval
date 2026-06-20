"""Brittle memory in a sequential, agentic task: Battleship.

Each turn is a fresh context that carries ONLY a compressed memory of the game so far (a clean
session boundary, as in the cross-session reclaim setup). The memory is compressed under a policy
at a fixed budget:

  - source_first : keeps the recomputable shot record (the hit/miss coordinates). The agent can see
                   what it already fired and where its hits are, so it avoids re-fires and extends hits.
  - lossy        : keeps the salient conclusion (a prose progress summary) and sheds the coordinates.
                   The agent cannot tell what it already fired, so it re-fires dead water and stalls.
  - lossy_padded : lossy padded with neutral filler to source_first's length (the budget control).

The brittle-memory prediction: at a matched budget, source_first sinks the fleet with few wasted
shots while lossy re-fires and often fails to finish, on the same model. Scoring is objective (sink
all ship cells); no judge.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

N = 8
ROWS = "ABCDEFGH"
FLEET = [("carrier", 4), ("cruiser", 3), ("destroyer", 2)]   # 9 ship cells total
TOTAL_SHIPS = len(FLEET)

SYSTEM = (
    "You are commanding a game of Battleship on an 8x8 grid. Rows are labelled A-H (top to bottom) "
    "and columns 1-8, so a cell looks like F6. The hidden enemy fleet is a carrier (4 cells), a "
    "cruiser (3 cells), and a destroyer (2 cells), each placed horizontally or vertically (9 ship "
    "cells total). You fire one shot per turn and your goal is to sink the whole fleet in as few "
    "shots as possible. Never fire a cell you have already fired. When you have an unsunk hit, fire "
    "an adjacent cell (up/down/left/right) to finish that ship. Do NOT restate the cells you have "
    "already fired. Reason in at most one short sentence, then end your reply with a line of its own: "
    "ANSWER: <cell>  (for example, ANSWER: F6)."
)

ANSWER_RE = re.compile(r"ANSWER:\s*([A-Ha-h])\s*([1-8])", re.I)
NOTE_RE = re.compile(r"\b([A-H])([1-8])\b")                   # strict: emitted coords only, no prose false-matches


def cell_name(r: int, c: int) -> str:
    return f"{ROWS[r]}{c + 1}"


def parse_answer_cell(text: str):
    """The cell on the last explicit ANSWER line, else None. No prose fallback: a reply that never
    reaches its ANSWER line (e.g. truncated mid-reasoning) is an invalid turn, not a phantom cell."""
    if not text:
        return None
    m = list(ANSWER_RE.finditer(text))
    if not m:
        return None
    g = m[-1]
    return (ROWS.index(g.group(1).upper()), int(g.group(2)) - 1)


# ── board generation ────────────────────────────────────────────────────────────
def place_fleet(seed: int):
    """Deterministic non-overlapping fleet placement for a given board seed."""
    rng = random.Random(seed)
    occupied, ships = set(), []
    for name, size in FLEET:
        while True:
            if rng.random() < 0.5:                       # horizontal
                r, c = rng.randint(0, N - 1), rng.randint(0, N - size)
                cells = [(r, c + i) for i in range(size)]
            else:                                        # vertical
                r, c = rng.randint(0, N - size), rng.randint(0, N - 1)
                cells = [(r + i, c) for i in range(size)]
            if any(cell in occupied for cell in cells):
                continue
            occupied.update(cells)
            ships.append({"name": name, "cells": cells})
            break
    return ships


def ship_cell_set(ships):
    return {tuple(c) for s in ships for c in s["cells"]}


# ── memory compression (the experiment) ─────────────────────────────────────────
PREFIX = "Memory of the game so far. "
PAD = " (Archived log fragment; no further detail was preserved.)"


def _sunk_names(history, ships):
    hit = {e["cell"] for e in history if e["result"] in ("HIT", "SUNK")}
    return [s["name"] for s in ships if all(tuple(c) in hit for c in s["cells"])]


def build_note(history, ships, policy: str, budget: int = 260) -> str:
    """Carried memory under each policy, matched to `budget` characters."""
    fired = [e["cell"] for e in history]
    sunk = _sunk_names(history, ships)
    sunk_cells = {tuple(c) for s in ships if s["name"] in sunk for c in s["cells"]}
    hit_unsunk = [e["cell"] for e in history if e["result"] in ("HIT", "SUNK")
                  and e["cell"] not in sunk_cells]
    miss = [e["cell"] for e in history if e["result"] == "MISS"]

    if policy == "lossy":
        return (PREFIX + f"You have fired {len(fired)} shots: {len(hit_unsunk)} live hits and "
                f"{len(sunk)} of {TOTAL_SHIPS} ships sunk. The remaining ships are not yet located. "
                f"Keep firing at new cells to find and sink them.")

    if policy == "lossy_padded":
        base = build_note(history, ships, "lossy", budget)
        target = len(build_note(history, ships, "source_first", budget))
        while len(base) < target:
            base += PAD
        return base

    if policy == "source_first":
        head = PREFIX
        if sunk:
            coords = " ".join(cell_name(*c) for c in sorted(sunk_cells))
            head += f"Sunk: {', '.join(sunk)} (cells {coords}). "
        if hit_unsunk:
            head += "Live hits to extend: " + " ".join(cell_name(*c) for c in hit_unsunk) + ". "
        # misses fill the remaining budget, newest first; truncation is the size cliff
        head += "Already fired (misses): "
        kept, dropped = [], 0
        for c in reversed(miss):
            tok = cell_name(*c)
            if len(head) + len(" ".join(kept + [tok])) + 1 <= budget:
                kept.append(tok)
            else:
                dropped += 1
        note = head + " ".join(reversed(kept))
        if dropped:
            note += f" (+{dropped} earlier misses not retained)"
        return note + "."

    raise ValueError(policy)


FIRE = "Fire your next shot now. End your reply with: ANSWER: <cell>."


def turn_messages(history, ships, policy, budget, feedback=None):
    """A fresh context each turn: the compressed carried memory, a one-line recency feedback of the
    last action (the same for every policy, so it does not confound the source-vs-conclusion test),
    and the fire instruction."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_note(history, ships, policy, budget)}]
    if feedback:
        msgs.append({"role": "user", "content": feedback})
    msgs.append({"role": "user", "content": FIRE})
    return msgs


# ── one game ────────────────────────────────────────────────────────────────────
@dataclass
class GameResult:
    won: bool
    turns: int            # LLM calls consumed
    distinct: int         # distinct cells fired
    redundant: int        # re-fires of an already-fired cell
    invalid: int          # unparseable / off-grid replies
    hits: int


def play_game(llm, ships, policy: str, budget: int, max_turns: int) -> GameResult:
    truth = ship_cell_set(ships)
    fired, hit, history = set(), set(), []
    redundant = invalid = turns = 0
    feedback = None
    while hit != truth and turns < max_turns:
        turns += 1
        reply = llm.chat(turn_messages(history, ships, policy, budget, feedback))
        cell = parse_answer_cell(reply)
        if cell is None:
            invalid += 1
            feedback = "Your last reply had no valid 'ANSWER: <cell>' line. Reply with exactly: ANSWER: <cell>."
            continue
        if cell in fired:
            redundant += 1
            feedback = f"{cell_name(*cell)} was already fired (a wasted shot). Pick a NEW, previously-unfired cell."
            continue
        fired.add(cell)
        if cell in truth:
            hit.add(cell)
            done = next((s for s in ships if all(tuple(c) in hit for c in s["cells"])
                         and tuple(cell) in {tuple(c) for c in s["cells"]}), None)
            result = "SUNK" if done else "HIT"
        else:
            result = "MISS"
        history.append({"cell": cell, "result": result})
        feedback = f"Your shot {cell_name(*cell)}: {result}."
    return GameResult(won=(hit == truth), turns=turns, distinct=len(fired),
                      redundant=redundant, invalid=invalid, hits=len(hit))


# ── validator: a deterministic fake that plays from the note ONLY ────────────────
class FakeCommander:
    """Stateless commander: it reconstructs what it has fired purely from the carried note, then
    fires an unfired cell (extending a live hit if the note records one). With source_first the note
    contains the coordinates, so it never re-fires and wins; with lossy the note has no coordinates,
    so it cannot tell what it fired and re-fires forever. The fake therefore PASSES only if the note
    actually carries the source, so a passing run cannot be faked by a note that dropped it."""

    def __init__(self):
        self.calls = self.prompt_tokens = self.completion_tokens = 0

    def chat(self, messages):
        self.calls += 1
        note = next(m["content"] for m in messages if m["role"] == "user")
        fired = {(ROWS.index(a), int(b) - 1) for a, b in NOTE_RE.findall(note)}
        hits_part = note.split("Live hits to extend:")[1].split(".")[0] if "Live hits to extend:" in note else ""
        live = [(ROWS.index(a), int(b) - 1) for a, b in NOTE_RE.findall(hits_part)]
        for (r, c) in live:                              # extend a live hit
            for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= nr < N and 0 <= nc < N and (nr, nc) not in fired:
                    return f"ANSWER: {cell_name(nr, nc)}"
        for r in range(N):                              # else first unfired cell
            for c in range(N):
                if (r, c) not in fired:
                    return f"ANSWER: {cell_name(r, c)}"
        return "ANSWER: A1"
