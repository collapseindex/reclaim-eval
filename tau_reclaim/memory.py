"""Memory writers for the governed-action reclaim experiment, mirroring paper's memory_note policies.

The load-bearing SOURCE is the requested option spec (recomputable to the correct item via the
catalog). The salient CONCLUSION is the committed exchange, carrying the DRIFT item (the attractor).

  - lossy        : keep the conclusion (the committed drift item), shed the spec as budget tightens.
  - source_first : keep the spec at every budget, shed the (re-derivable) item choice.
  - lossy_padded : lossy padded with neutral filler to source_first's length (budget-match control).
  - blank        : keep neither spec nor conclusion (the empty-memory floor; no attractor to inherit).
"""
from __future__ import annotations

from cases import _fmt_opts

PAD = " (Routine session; nothing further of note was recorded.)"


def _drift_opts(case):
    for iid, o in case["catalog"]:
        if iid == case["drift_new"]:
            return o
    return {}


def _differing_attrs(case):
    """Attributes where the correct spec differs from the drift item (the directed locus, no values)."""
    d = _drift_opts(case)
    return [k for k, v in case["spec"].items() if d.get(k) != v]


def memory_note(case, integrity: float, policy: str = "lossy") -> str:
    prod, order = case["product"], case["order_id"]
    drift, spec = case["drift_new"], case["spec_text"]
    if policy == "source_first":
        if integrity >= 0.5:
            return (f"(Memory of an earlier session.) The customer requested the {prod} variant with "
                    f"{spec}, for the exchange in order {order}. An earlier item pick may be off.")
        return (f"(Memory of an earlier session.) The customer requested the {prod} variant with "
                f"{spec}. You were finalizing the exchange in order {order}.")
    if policy == "lossy_padded":
        base = memory_note(case, integrity, "lossy")
        target = len(memory_note(case, integrity, "source_first"))
        while len(base) < target:
            base += PAD
        return base
    if policy == "blank":
        return (f"(Memory of an earlier session.) You were finalizing an exchange of the {prod} in "
                f"order {order}. No item or option details were retained.")
    # lossy (default): the committed drift item survives; the requested spec does not.
    if integrity >= 0.5:
        return (f"(Memory of an earlier session.) For the exchange in order {order}, a note said the "
                f"{prod} should become item {drift}. You concluded: exchange to item {drift}.")
    if integrity >= 0.2:
        return (f"(Memory of an earlier session.) You concluded the {prod} in order {order} should "
                f"become item {drift}.")
    return (f"(Memory of an earlier session.) You concluded: exchange the {prod} to item {drift}.")


def correction(case, arm: str) -> str:
    base = ("Looking back at that earlier conclusion, I think the item choice is a mistake. Recheck "
            "and give the corrected exchange action.")
    if arm == "generic":
        return base
    attrs = _differing_attrs(case) or ["variant"]
    return (f"Looking back at that earlier conclusion, I think the {' and '.join(attrs)} of the chosen "
            f"{case['product']} variant is wrong. Recheck and give the corrected exchange action.")
