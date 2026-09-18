# Phase 3 — documentation (2026-09-18)

**Outcome: done.** Drafted by a Sonnet subagent from the phase 2 report, reviewed and committed here.

| File | Change |
|---|---|
| `timesfm/README.md` | new: intro, three result tables, plot, reading, how to run, license, files |
| `README.md` | TimesFM row (bold) in honest table, row in teacher-forced table, Models paragraph, Discussion subsection "Foundation model vs trained-from-scratch", complexity row, project-structure entry, `timesfm[torch]` in setup, license note in intro |
| `CLAUDE.md` | now tracked; both tables get a TimesFM row, ranking sentence updated, pointer to `timesfm/reports/` |

Review fix: complexity-table row rewritten to match the Time / Space / Parallelisable columns.
All numbers cross-checked against `timesfm/results.csv` (481.81 / 3.09%, 94.03 / 0.61%).

Not touched: `energy_forecasting.ipynb` (TimesFM lives in its own folder by request),
`app.py` (no dashboard integration; note its `'LSTM (best)'` label is stale either way).
