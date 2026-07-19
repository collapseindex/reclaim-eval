"""Build governed-action reclaim cases from real tau-bench single-item exchange/modify tasks.

Each case has the reclaim structure, with every piece sourced from tau-bench:
  - catalog   : the product's available variants (item_id -> options) = the shared, recomputable context
  - spec      : the option set of the CORRECT variant = the load-bearing SOURCE a lossy writer drops
  - correct   : the annotated write action (tool + kwargs)  -> gt_hash
  - drift     : the SAME action to a different available variant (wrong option) -> a real wrong mutation

The correct item_id is recomputable from (spec + catalog), so source-first enables genuine
re-derivation, not id-lookup: lossy keeps "you exchanged the keyboard to <drift id>" and sheds the
spec, so the reader re-asserts the drift; source-first keeps the spec, so the reader re-picks correctly.
"""
from __future__ import annotations

from scorer import TASKS_TEST, load_data, WRITE_TOOLS

_DATA = load_data()
_PRODUCTS = _DATA["products"]
EXCHANGE_TOOLS = {"exchange_delivered_order_items", "modify_pending_order_items"}


def _product_of(item_id):
    for p in _PRODUCTS.values():
        if item_id in p.get("variants", {}):
            return p
    return None


def _fmt_opts(opts):
    return ", ".join(f"{k}={v}" for k, v in sorted(opts.items()))


def build_cases(limit=12):
    """One case per single-item exchange/modify task that has a clean wrong-variant drift."""
    cases = []
    for i, t in enumerate(TASKS_TEST):
        writes = [a for a in t.actions if a.name in WRITE_TOOLS]
        if len(writes) != 1:
            continue
        w = writes[0]
        if w.name not in EXCHANGE_TOOLS:
            continue
        kw = w.kwargs
        if len(kw.get("item_ids", [])) != 1 or len(kw.get("new_item_ids", [])) != 1:
            continue
        old_id, correct_new = kw["item_ids"][0], kw["new_item_ids"][0]
        prod = _product_of(correct_new)
        if not prod or correct_new not in prod["variants"]:
            continue
        if not prod["variants"][correct_new]["available"]:
            continue
        # candidate drifts: other AVAILABLE variants, not the correct one, not the current item
        alts = [iid for iid, v in prod["variants"].items()
                if v["available"] and iid not in (correct_new, old_id)]
        if not alts:
            continue
        spec = prod["variants"][correct_new]["options"]
        # pick the drift whose options differ most from the spec (an unambiguous wrong pick)
        def ndiff(iid):
            o = prod["variants"][iid]["options"]
            return sum(o.get(k) != v for k, v in spec.items())
        drift_new = max(alts, key=ndiff)
        catalog = [(iid, v["options"]) for iid, v in prod["variants"].items() if v["available"]]
        cases.append({
            "name": f"t{i}_{prod['name'].replace(' ', '')}",
            "task_index": i,
            "tool": w.name,
            "product": prod["name"],
            "order_id": kw["order_id"],
            "old_item_id": old_id,
            "payment_method_id": kw["payment_method_id"],
            "correct_new": correct_new,
            "drift_new": drift_new,
            "spec": spec,
            "spec_text": _fmt_opts(spec),
            "catalog": catalog,
            "correct": {"tool": w.name, "kwargs": dict(kw)},
        })
        if len(cases) >= limit:
            break
    return cases


def catalog_text(case):
    lines = [f"  - item {iid}: {_fmt_opts(o)}" for iid, o in case["catalog"]]
    return (f"Available variants of the {case['product']} (product catalog you can consult):\n"
            + "\n".join(lines))


if __name__ == "__main__":
    cs = build_cases()
    print(f"built {len(cs)} cases")
    for c in cs[:4]:
        print(f"\n--- {c['name']}  ({c['tool']}) ---")
        print(f"  order {c['order_id']}  old {c['old_item_id']}  pay {c['payment_method_id']}")
        print(f"  SPEC (source): {c['spec_text']}")
        print(f"  correct_new {c['correct_new']}  drift_new {c['drift_new']}")
        print("  " + catalog_text(c).replace("\n", "\n  "))
