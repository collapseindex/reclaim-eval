#!/usr/bin/env python3
"""Pull the ticked checkbox per item from a filled annotation_form.pdf into annotator.csv.
    pip install pypdf ; python extract_pdf.py annotation_form_filled.pdf
Falls back cleanly if pypdf is absent (just have the annotator fill annotator_template.csv instead)."""
import csv, sys
try:
    from pypdf import PdfReader
except ImportError:
    sys.exit("pypdf not installed; have the annotator fill annotator_template.csv instead.")
path = sys.argv[1] if len(sys.argv) > 1 else "annotation_form_filled.pdf"
fields = PdfReader(path).get_fields() or {}
def on(name):
    v = fields.get(name, {}).get("/V")
    return v not in (None, "/Off", "Off", "")
rows = []
for i in range(51):
    picks = [lab for lab, sfx in (("COMPACT","c"),("DIFFUSE","d"),("NONE","n")) if on(f"q{i}{sfx}")]
    rows.append((i, picks[0] if len(picks) == 1 else ""))  # blank if none or multiple ticked
with open("annotator.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["item", "label"]); w.writerows(rows)
bad = [i for i, l in rows if not l]
print(f"wrote annotator.csv; {51-len(bad)}/51 single-ticked" + (f"; check items {bad}" if bad else ""))
