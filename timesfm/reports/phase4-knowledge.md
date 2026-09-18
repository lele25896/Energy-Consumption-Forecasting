# Phase 4 — knowledge base + closing (2026-09-18)

**Outcome: done.** Drafted by a Sonnet subagent, reviewed, committed and pushed here.

| Where | Change |
|---|---|
| Obsidian vault `wiki/entities/projects/energy-forecasting.md` | new section "2026-09-18: TimesFM 3.0 zero-shot" (tables, ablation, calibration, caveats), `last_updated` bumped; committed and pushed to `origin main` of the second-brain repo |
| Interview deep-dive `..\Colloqui\Knowledge\Project\energy-forecasting-deep.md` | new §9 (Italian, matches file) with talking points: no exposure bias by construction, why zero-shot parity with the LSTM matters, "why still train custom models" answer, weak spot |
| `CLAUDE.md` | deep-dive path corrected (old path did not exist) |
| Claude Code memory | project memory `timesfm-zero-shot` |

## Open follow-ups (not done, by scope)
- Rolling-origin backtest over several 4-week windows across seasons; single window is the main weakness.
- Covariates: `predict_batch(..., past_future_covariates=[calendar feats])` to test whether TimesFM gains from what XGBoost already uses.
- `app.py`: no TimesFM option (user decision); sidebar label `'LSTM (best)'` is stale regardless.
- Repo not pushed to GitHub; four local commits (phase 1-4) ready.
