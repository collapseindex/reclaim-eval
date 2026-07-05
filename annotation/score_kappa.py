#!/usr/bin/env python3
"""Inter-rater agreement between an annotator and the author key.
    python score_kappa.py annotator.csv           # annotator.csv has columns: item,label (COMPACT/DIFFUSE/NONE)
"""
import csv, sys

def load(path, col):
    d = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        v = (r.get(col) or "").strip().upper()
        if v:
            d[int(r["item"])] = v
    return d

def cohen_kappa(a, b):
    items = sorted(set(a) & set(b))
    n = len(items)
    labs = ["COMPACT", "DIFFUSE", "NONE"]
    obs = sum(a[i] == b[i] for i in items) / n
    pa = {l: sum(a[i] == l for i in items) / n for l in labs}
    pb = {l: sum(b[i] == l for i in items) / n for l in labs}
    exp = sum(pa[l] * pb[l] for l in labs)
    k = (obs - exp) / (1 - exp) if exp < 1 else 1.0
    return n, obs, k

if __name__ == "__main__":
    ann_path = sys.argv[1] if len(sys.argv) > 1 else "annotator.csv"
    key = load("key.csv", "author_label")
    ann = load(ann_path, "label")
    n, obs, k = cohen_kappa(key, ann)
    print(f"labelled items compared: {n}/51")
    print(f"raw agreement: {obs:.2f}")
    print(f"Cohen's kappa (author vs annotator): {k:.2f}")
