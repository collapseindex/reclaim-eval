#!/usr/bin/env python3
"""Multi-rater agreement: pairwise Cohen kappa, Fleiss kappa, and majority-vote consensus.
Reads the author labels (key.csv) plus every rater*.csv present (item,label).
    python score_multi.py
"""
import csv, glob, itertools
LABS = ["COMPACT", "DIFFUSE", "NONE"]

def load(path, col):
    return {int(r["item"]): (r.get(col) or "").strip().upper()
            for r in csv.DictReader(open(path, encoding="utf-8")) if (r.get(col) or "").strip()}

def cohen(a, b):
    items = sorted(set(a) & set(b)); n = len(items)
    obs = sum(a[i] == b[i] for i in items) / n
    exp = sum((sum(a[i] == l for i in items) / n) * (sum(b[i] == l for i in items) / n) for l in LABS)
    return n, obs, (obs - exp) / (1 - exp) if exp < 1 else 1.0

def fleiss(raters):
    items = sorted(set.intersection(*[set(r) for r in raters]))
    k = len(raters)
    P = []
    for i in items:
        c = {l: sum(r[i] == l for r in raters) for l in LABS}
        P.append((sum(v * v for v in c.values()) - k) / (k * (k - 1)))
    Pbar = sum(P) / len(P)
    pj = {l: sum(r[i] == l for r in raters for i in items) / (len(items) * k) for l in LABS}
    Pe = sum(v * v for v in pj.values())
    return len(items), k, (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0

raters = {"author": load("key.csv", "author_label")}
for p in sorted(glob.glob("rater*.csv")):
    raters[p.replace(".csv", "")] = load(p, "label")
names = list(raters)
print(f"raters: {names}\n")
print("pairwise Cohen kappa:")
for a, b in itertools.combinations(names, 2):
    n, obs, k = cohen(raters[a], raters[b])
    print(f"  {a:8s} vs {b:8s}: kappa={k:.2f}  agreement={obs:.2f}  (n={n})")
if len(names) >= 3:
    n, k, fk = fleiss(list(raters.values()))
    print(f"\nFleiss kappa ({k} raters, n={n}): {fk:.2f}")
    items = sorted(set.intersection(*[set(r) for r in raters.values()]))
    from collections import Counter
    maj = {i: Counter(r[i] for r in raters.values()).most_common(1)[0][0] for i in items}
    for nm, r in raters.items():
        print(f"  {nm} matches majority: {sum(r[i]==maj[i] for i in items)}/{len(items)}")
