"""Reclaim on real conversational memory: MultiWOZ slot recovery.

Closes the compact-source external-validity gap. The source is now a real, chatty, multi-turn
dialogue (entangled and fuzzy), but the target is a checkable slot value (a booking/departure
time), so scoring stays objective with no judge. A user utterance states the value verbatim
(the recoverable source); a corrupted confirmation carries a wrong value (the drift). A lossy
memory keeps the wrong confirmation and walls; a source-first memory keeps the user utterance
and recovers. Same mechanism as the arithmetic ledger, on a dataset reviewers already respect.

Data: MultiWOZ 2.2 dialogues (raw JSON). Each USER turn carries span-annotated slot values.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TIME = re.compile(r"^\d{1,2}:\d{2}$")
# checkable time slots that appear verbatim in a user utterance, with a human description
SLOT_DESC = {
    "train-leaveat": "train departure time",
    "train-arriveby": "train arrival time",
    "restaurant-booktime": "restaurant booking time",
    "taxi-leaveat": "taxi pickup time",
    "taxi-arriveby": "taxi arrival time",
}


@dataclass
class WozTarget:
    dialogue_id: str
    slot: str
    desc: str
    true_value: str          # ground-truth slot value, e.g. "16:15"
    drift_value: str         # a plausible wrong value (the planted drift)
    source_utt: str          # the user utterance stating the true value (the source)


def _drift_time(hhmm: str) -> str:
    """A plausible wrong time: shift hours deterministically, keep a valid HH:MM, != true."""
    h, m = (int(x) for x in hhmm.split(":"))
    nh = (h + 3) % 24 if h < 21 else (h - 3) % 24
    if nh == h:
        nh = (h + 5) % 24
    return f"{nh:02d}:{m:02d}"


def load_targets(path, slots=tuple(SLOT_DESC), max_n=None):
    """One target per dialogue: the first checkable time slot stated verbatim by the user."""
    dialogues = json.load(open(path, encoding="utf-8"))
    out = []
    for dlg in dialogues:
        chosen = None
        for t in dlg["turns"]:
            if t["speaker"] != "USER":
                continue
            for fr in t.get("frames", []):
                for s in fr.get("slots", []):
                    slot, val = s.get("slot"), s.get("value", "")
                    # require the value verbatim in the utterance: the source must actually
                    # contain the recoverable answer (drops "noon"/"4pm"-style normalizations)
                    if (slot in slots and isinstance(val, str) and TIME.match(val)
                            and 0 <= int(val.split(":")[0]) <= 23 and val in t["utterance"]):
                        chosen = WozTarget(
                            dialogue_id=dlg["dialogue_id"], slot=slot, desc=SLOT_DESC[slot],
                            true_value=val, drift_value=_drift_time(val),
                            source_utt=t["utterance"].strip())
                        break
                if chosen:
                    break
            if chosen:
                break
        if chosen:
            out.append(chosen)
        if max_n and len(out) >= max_n:
            break
    return out


PREFIX = "(Memory of an earlier session, summarizing a booking conversation.) "
PAD = (" (Archived dialogue log; no further turns were preserved.)")


def build_note(tgt: WozTarget, policy: str, budget: int = 320) -> str:
    """Carried memory under each policy, budget-matched.

      - lossy: keep the (corrupted) confirmation, drop the source utterance.
      - source_first: keep the user's verbatim utterance (the source), drop the conclusion.
      - lossy_padded: lossy padded with filler to source_first's length.
      - blank: neither source nor value (free-guess floor; times are ~unguessable).
    """
    if policy == "source_first":
        return (PREFIX + f"The user said, verbatim: \"{tgt.source_utt}\" "
                f"A {tgt.desc} was being arranged; the earlier confirmation may be off.")
    if policy == "lossy":
        return PREFIX + f"You confirmed a {tgt.desc} of {tgt.drift_value}."
    if policy == "lossy_padded":
        base = PREFIX + f"You confirmed a {tgt.desc} of {tgt.drift_value}."
        target_len = len(build_note(tgt, "source_first", budget))
        while len(base) < target_len:
            base += PAD
        return base
    if policy == "blank":
        return PREFIX + f"A {tgt.desc} was being arranged."
    raise ValueError(policy)


def reclaim_msg(tgt: WozTarget, arm: str) -> str:
    base = (f"Looking back at that earlier conversation, I think the {tgt.desc} is wrong. "
            f"Please recheck and give the correct {tgt.desc} as 'ANSWER: <HH:MM>' "
            f"in 24-hour time.")
    if arm == "generic":
        base = (f"Looking back at that earlier conversation, I think there is a mistake. "
                f"Please recheck and give the correct {tgt.desc} as 'ANSWER: <HH:MM>' "
                f"in 24-hour time.")
    return base


_ANS = re.compile(r"ANSWER:\s*\*?\*?\s*(\d{1,2}):(\d{2})\s*(am|pm)?", re.I)
_ANY = re.compile(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?", re.I)


def _norm(h, m, ap):
    h, m = int(h), int(m)
    if ap:
        ap = ap.lower()
        if ap == "pm" and h != 12:
            h += 12
        if ap == "am" and h == 12:
            h = 0
    return f"{h % 24:02d}:{m:02d}"


def extract_time(reply: str):
    m = _ANS.search(reply or "")
    if not m:
        cand = list(_ANY.finditer(reply or ""))
        m = cand[-1] if cand else None
    return _norm(*m.groups()) if m else None


def _canon(hhmm: str) -> str:
    h, m = hhmm.split(":")
    return f"{int(h) % 24:02d}:{int(m):02d}"


def score(reply: str, tgt: WozTarget) -> bool:
    got = extract_time(reply)
    return got is not None and got == _canon(tgt.true_value)
