# Energy Consumption Forecasting — project context

Working clone of `github.com/lele25896/Energy-Consumption-Forecasting`. Code is edited here.
Owner: Gabriele Giacometti (PhD Physics → ML/DS). Environment: conda env `torch_env`
(`C:\Users\gabri\miniconda3\envs\torch_env\python.exe` — call with the full path, `conda activate` is broken in PowerShell).

## What it is
Hourly electricity demand forecasting, AEP dataset (~121k hours). 5 models compared:

Zero-shot TimesFM 3.0 baseline: `timesfm/run_timesfm.py`. Weights under the TimesFM
Non-Commercial License v1.0, portfolio use only.

Teacher-forced metrics (historical reference):

| Model | MAE (MW) | MAPE |
|---|---|---|
| Prophet | 1535.93 | 10.33% |
| XGBoost + lag | 133.58 | 0.85% |
| LSTM + cyclic | 94.35 | 0.61% |
| TimesFM 3.0 zero-shot | 94.03 | 0.61% |
| PatchTST | 141.22 | 0.94% |

Honest metrics — 672h autoregressive rollout (no re-anchor):

| Model | MAE (MW) | MAPE | Note |
|---|---|---|---|
| **TimesFM 3.0 zero-shot** | **481.81** | **3.09%** | **best overall, no training** |
| Prophet | 1535.93 | 10.33% | immune, global model |
| XGBoost + lag | 983.25 | 6.40% | best trained model |
| LSTM + scheduled sampling | 2154.65 | 14.02% | -48% vs pure TF |
| PatchTST | 2399.23 | 15.95% | |

The ranking flips: XGBoost wins among trained-from-scratch models, but zero-shot TimesFM
wins overall; LSTM with scheduled sampling cuts the error in half (4132→2154).

`AEP_hourly.csv` is gitignored → download from Kaggle before running `energy_forecasting.ipynb`.

## Evaluation
Implemented a **multi-step autoregressive** rollout: single 672h horizon, no re-anchor
on real data. Each prediction is fed back as input for the next step.
Valid both for XGBoost (lag features) and for LSTM/PatchTST (168h window).

## References
- Full deep-dive (9 sections, weak spots): `..\Colloqui\Knowledge\Project\energy-forecasting-deep.md`
- Obsidian entity: `brain\wiki\entities\projects\energy-forecasting.md`
- TimesFM phase 1/2 write-ups: `timesfm\reports\`
