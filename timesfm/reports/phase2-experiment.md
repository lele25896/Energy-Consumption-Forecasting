# Phase 2 — zero-shot experiment (2026-09-18)

**Outcome: TimesFM 3.0 zero-shot beats every trained model on the honest 672 h protocol and ties the
LSTM on teacher-forced 1-step.** Script: `timesfm/run_timesfm.py`, artefacts `results.csv`,
`timesfm_pred.csv`, `forecast_timesfm.png`. Total runtime ≈ 30 s on RTX 4070 Laptop (no training).

Setup identical to the notebook: AEP hourly, train 120 601 h, test = last 672 h
(2018-07-06 01:00 → 2018-08-03 00:00). Context = last 2048 h of train (~12 weeks).
`use_symmetric_averaging=False`, defaults otherwise.

## Honest protocol — single direct 672 h forecast, no test data seen

| Model | MAE (MW) | MAPE | Note |
|---|---|---|---|
| **TimesFM 3.0 zero-shot** | **481.81** | **3.09%** | one call, 0.9 s, context 2048 h |
| XGBoost + lag (rollout) | 983.25 | 6.40% | best trained model |
| Prophet | 1535.93 | 10.33% | |
| LSTM + scheduled sampling | 2154.65 | 14.02% | |
| PatchTST | 2399.23 | 15.95% | |

10–90 % quantile band: empirical coverage 88.5 % over the 672 test hours (nominal 80 %), slightly
conservative rather than over-confident.

## Teacher-forced 1-step (real history up to t, horizon 1, 672 calls batched)

| Model | MAE (MW) | MAPE |
|---|---|---|
| LSTM + cyclic | 94.35 | 0.61% |
| **TimesFM 3.0 zero-shot** | **94.03** | **0.61%** |
| XGBoost + lag | 133.58 | 0.85% |
| PatchTST | 141.22 | 0.94% |
| Prophet | 1535.93 | 10.33% |

## Context-length ablation (honest protocol)

| Context (h) | MAE | MAPE |
|---|---|---|
| 512 | 812.41 | 5.14% |
| 1024 | 518.00 | 3.28% |
| **2048** | **481.81** | **3.09%** |
| 4096 | 487.78 | 3.14% |
| 8192 | 524.73 | 3.37% |
| 16384 | 535.78 | 3.45% |

Sweet spot ~2–4k h (3–6 months). Below 1k the model lacks a full weekly/seasonal picture; above
8k the extra history is mostly other seasons and mildly hurts a July–August forecast.

## Reading

- The honest gap (482 vs 983) is the exposure-bias story from the main README, seen from the other
  side: TimesFM emits the whole horizon in patches of 64 and never feeds its own point estimate
  back as a lag feature, so it has no compounding error to fight.
- Teacher-forced parity with the LSTM (94.0 vs 94.4) says the pretrained model already matches a
  model trained for 40 epochs on this exact series, without seeing it.
- Caveat: one 4-week window in summer 2018. A rolling-origin backtest over several windows is the
  obvious next step before calling the ranking robust.
- Weights are under the TimesFM Non-Commercial License v1.0; fine for a portfolio, not for
  production.
