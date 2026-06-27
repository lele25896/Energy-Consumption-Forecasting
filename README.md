# Energy Consumption Forecasting

Hourly electricity demand forecasting on the AEP dataset (~121k hours, 2004–2018).
Compares four approaches — from classical decomposition to a Transformer — using a
fully **autoregressive 672-hour rollout** evaluation and an interactive Streamlit dashboard.

## Results

Evaluation uses a **single-pass autoregressive rollout** on the held-out last 4 weeks (672 h):
each model predicts one step ahead using only its own prior predictions, no re-anchoring on real data.

| Model | MAE (MW) | MAPE |
|-------|----------|------|
| Prophet | 1535.93 | 10.33% |
| **XGBoost + lag features** | **983.25** | **6.40%** |
| LSTM + cyclic + scheduled sampling | 2154.65 | 14.02% |
| PatchTST | 2399.23 | 15.95% |

<details>
<summary>Teacher-forced reference numbers (one-step-ahead, non-production)</summary>

| Model | MAE (MW) | MAPE |
|-------|----------|------|
| Prophet | 1535.93 | 10.33% |
| XGBoost | 133.58 | 0.85% |
| LSTM | 94.35 | 0.61% |
| PatchTST | 141.22 | 0.94% |

Teacher-forced evaluation uses the real energy history at every step — it measures
one-step-ahead nowcasting accuracy, not multi-step forecasting ability.

</details>

![Forecast comparison — last 2 weeks of test set](forecast_comparison.png)

## Models

**Prophet** — additive decomposition (trend + daily/weekly/yearly seasonality).
No memory of the immediate past; treats the series as a sum of smooth components.
Provides 80% confidence intervals out of the box.
Immune to autoregressive error accumulation because it predicts the full horizon at once.

**XGBoost** — gradient boosted trees on handcrafted lag features (1 h, 24 h, 48 h, 168 h),
rolling statistics, and calendar features. Early stopping on an inner validation split
(last 2 weeks of train) — the test set is never seen during fitting.

![XGBoost feature importance](feature_importance.png)

**LSTM** — two-layer LSTM (hidden=64, ~80k params) with 8 input features: scaled energy +
cyclic sin/cos encodings of hour, day-of-week, and month + is_weekend flag.
Trained with **scheduled sampling** (ss_prob 0 → 0.5 over 40 epochs) to reduce exposure bias.

**PatchTST** — Transformer encoder on overlapping patches (patch_len=16, stride=8 → 20 tokens,
~419k params). Pre-LayerNorm, learnable positional embeddings.
Reduces attention complexity from O(L²) to O(P²) — 70× fewer ops vs raw attention at L=168.

## Discussion

### The ranking inverts under autoregressive evaluation

Teacher-forced evaluation flatters sequence models: every prediction sees the real energy history,
so even a model that accumulates errors under rollout will score well. Under true multi-step
rollout the picture changes completely.

**XGBoost wins the autoregressive evaluation** because its explicit lag_24 and lag_168 features
use real historical values for the first 24–168 steps, anchoring the rollout before the window
fills with predictions. GBTs are also less sensitive to small input perturbations by construction.

**LSTM collapses without scheduled sampling** (MAE 4132 MW, MAPE 27%) due to **exposure bias**:
trained exclusively on real inputs (teacher forcing), it was never hardened against its own
prediction errors. After 168 autoregressive steps the entire lookback window is synthetic —
the model is in full distribution shift.

**Scheduled sampling halves LSTM error** (4132 → 2154 MW): during training, inputs are
gradually replaced with the model's own predictions (ss_prob 0 → 0.5 after a 5-epoch warmup),
forcing the model to learn to recover from its own mistakes. LSTM with SS now matches PatchTST.

**PatchTST degrades but survives** without any explicit fix, likely because attention diffuses
accumulated errors across the patch window rather than piping them directly through hidden state.

### Why LSTM beats PatchTST in teacher-forced evaluation

The AEP series is dominated by **short-term autocorrelation**: the single best predictor
is the value one hour ago (lag_1), followed by lag_24 (same hour yesterday).

LSTM's sequential inductive bias is well-suited: its hidden state implicitly tracks the
immediate past, giving lag_1/lag_24 strong weight at every step.

PatchTST groups 16 consecutive hours into one patch token, **diluting the lag_1 signal**.
The most recent hour is merged with the preceding 15; the model must reconstruct fine-grained
recency from a coarser representation. Echoes **Zeng et al. 2022** (*Are Transformers
Effective for Time Series Forecasting?*).

### Computational complexity

| Model | Time | Space | Parallelisable |
|-------|------|-------|----------------|
| LSTM | O(L · d²) | O(L · d) | ❌ Sequential |
| Transformer (raw) | O(L² · d) | O(L²) | ✅ Parallel |
| PatchTST | O(P² · d) | O(P²) | ✅ Parallel |

L = sequence length, d = hidden size, P = number of patches (P ≪ L).

### When Transformers are the right choice

1. **Long-range dependencies** — forecasting 30 days ahead, where today relates to the
   same weekday last month. Self-attention connects tokens 720 steps apart directly; LSTM
   gradients vanish over that distance.
2. **Multivariate cross-series interactions** — 50 substations with load-shifting; attention
   captures pairwise relationships across all series in one operation.
3. **Large-scale pretraining** — foundation models (TimesFM, Chronos) pretrained on millions
   of diverse series outperform task-specific LSTMs when labeled data is scarce.

**Rule of thumb**: if lag_1 and lag_24 explain most variance → XGBoost or LSTM with
scheduled sampling. Reach for Transformers when the predictive signal is long or diffuse.

## Project structure

```
energy_forecasting.ipynb   # training, evaluation, and model comparison
app.py                     # Streamlit dashboard (autoregressive inference)
lstm_forecaster.pt         # LSTM weights (trained with scheduled sampling)
patchtst_forecaster.pt     # PatchTST weights
xgb_forecaster.pkl         # XGBoost model
energy_scaler.pkl          # StandardScaler for the energy column
AEP_hourly.csv             # raw data (download from Kaggle, not in repo)
```

## Dataset

[Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
from Kaggle. Use `AEP_hourly.csv` (largest region, ~121k rows).
Place it in the project root before running the notebook.

## Setup

```bash
conda create -n torch_env python=3.11
conda activate torch_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn xgboost prophet matplotlib tqdm joblib streamlit
```

## Run

**Notebook** — open `energy_forecasting.ipynb` and run all cells in order.
Trained models are saved automatically at the end.

**Dashboard** — after running the notebook at least once:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).
