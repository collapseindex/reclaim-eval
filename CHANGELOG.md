# Changelog

All notable changes to the reclaim-eval harness and paper. Dates are absolute.

## arXiv v2 (2026-06-29)

### Fixed
- Answer-parser bug that scraped a non-answer (a stray number or filler word) as a committed answer,
  inflating the *confident-wrong emission* tables. Re-scored under **strict scoring** (the value
  committed on the `ANSWER` line). Affected: `tab:disposition`, `tab:blank`, `tab:failmode`,
  `tab:corrtax` (Opus cell), the sycophancy capitulation cells, and the endogenous self-error numbers.
  Source-first and all recovery-rate results are unchanged; MultiWOZ `tab:attractor` was already strict
  and is unchanged. Full autopsy: `NOTE_parser_fix.md`.

### Changed
- Reframed "worse than empty" as **disposition-contingent**: strong on models disposed to answer
  (deepseek, grok, qwen, llama), absent on the four frontier OpenAI/Anthropic models that abstain. The
  split is disposition, not capability.
- Tightened the abstract; restored the compact-identifiable-source scope clause and the writer
  conditioning on the deployable number; folded in reviewer scope-honesty edits (locating-vs-reading,
  cascade-as-corollary, prevalence as the keystone uncertainty).
- Split Results into navigable top-level sections (Generality, Boundary, Real Conversational Memory,
  Cascade).

### Added
- `gpt-5.4` (OpenAI API) and `grok-4.3` via the **official xAI API** to the disposition breadth sweep
  (now 8 models across 4 vendors). New harness scripts: `run_sweep.py`, `logic_failmode.py`,
  `probe_models.py`, `verify_multiwoz.py`.

### Removed
- Unused `pennington2014glove` bib entry; verified all model-card references.

## arXiv v1 — 2606.25449 (2026)

- Initial release: reclaim evaluation, the source-first remedy, boundary mapping, deployed-system and
  MultiWOZ replication, and the cascade.
