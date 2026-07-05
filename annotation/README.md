# Independent annotation set: compact-source prevalence (51 traces)

A second, independent human rater converts the audit's single-author human anchor into a real
**inter-rater** agreement number, the one thing the prevalence audit's human anchor currently lacks.
Same 51 blind traces, same rubric as the audit (App.\ "Prevalence audit").

## Give this to the annotator
- **`annotation_form.pdf`** — the fillable form: rubric, 6 calibration examples (answers shown), then the
  51 items, each with COMPACT / DIFFUSE / NONE checkboxes. They tick one box per item, save, and send it
  back. (Or they fill **`annotator_template.csv`** by hand instead.)
- Do **not** send them `key.csv` — that's the author's labels, kept private so their read is independent.

## Score it when it comes back
```
python extract_pdf.py annotation_form_filled.pdf   # filled PDF -> annotator.csv  (needs: pip install pypdf)
python score_kappa.py annotator.csv                # Cohen's kappa, author vs annotator
```
If they filled the CSV instead, skip the extract step and run `score_kappa.py annotator_template.csv`.

## Files
- `form.tex` — LaTeX source of the form.
- `annotation_form.pdf` — the form to send.
- `annotator_template.csv` — blank CSV alternative for recording labels.
- `key.csv` — author labels (**private**; scoring only).
- `extract_pdf.py`, `score_kappa.py` — tooling.

The traces are drawn verbatim from public HF corpora (no_robots, glaive-function-calling-v2,
AgentInstruct); no private data.
