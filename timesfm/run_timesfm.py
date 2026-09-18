"""TimesFM 3.0 zero-shot on AEP hourly load. Same split as ../energy_forecasting.ipynb
(last 4 weeks = 672 h held out). Three protocols:
  1. honest   : one direct 672h forecast from the last CONTEXT hours of train (no test data used)
  2. teacher  : 672 one-step forecasts, each context = real history up to t (matches the TF table)
  3. ablation : honest protocol vs context length
Writes results.csv, timesfm_pred.csv, forecast_timesfm.png next to this file.
Run: <torch_env>/python.exe timesfm/run_timesfm.py   (from repo root)
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from timesfm3 import ModelConfig, TimesFM3Evaluator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEST_WEEKS = 4
CONTEXT = 2048                      # ~12 weeks, picked from the ablation below
ABLATION = [512, 1024, 2048, 4096, 8192, 16384]

def mape(y, p):
    return np.mean(np.abs((y - p) / y)) * 100

# ── data, identical to the notebook ──────────────────────────────────────────
df = pd.read_csv(ROOT / 'AEP_hourly.csv', parse_dates=['Datetime'], index_col='Datetime').sort_index()
df.columns = ['energy']
test_size = TEST_WEEKS * 7 * 24
train, test = df.iloc[:-test_size], df.iloc[-test_size:]
y_train = train['energy'].values.astype(np.float32)
y_test = test['energy'].values
print(f'Train: {len(train)} | Test: {len(test)} ({test.index[0]} -> {test.index[-1]})')

t0 = time.time()
tfm = TimesFM3Evaluator(ModelConfig(checkpoint_path='google/timesfm-3.0-pytorch',
                                    per_core_batch_size=32, device='cuda'))
print(f'TimesFM loaded in {time.time() - t0:.1f}s')

def direct(ctx_len, horizon=test_size):
    """Honest: forecast the whole horizon from train only. Returns ForecastOutput."""
    ctx = y_train[-ctx_len:]
    return next(iter(tfm.predict_batch([ctx], horizon=horizon, return_quantiles=True,
                                       use_symmetric_averaging=False)))

rows = []

# ── 1. honest 672h direct forecast ───────────────────────────────────────────
t0 = time.time()
out = direct(CONTEXT)
pred, q10, q90 = out.forecast, out.quantiles[:, 0], out.quantiles[:, -1]
assert pred.shape == (test_size,) and np.isfinite(pred).all()
rows.append(dict(protocol='honest_672h', context=CONTEXT, MAE=mean_absolute_error(y_test, pred), MAPE=mape(y_test, pred)))
print(f'honest 672h  ctx={CONTEXT}: MAE {rows[-1]["MAE"]:.2f} | MAPE {rows[-1]["MAPE"]:.2f}%  ({time.time()-t0:.2f}s)')
coverage = np.mean((y_test >= q10) & (y_test <= q90)) * 100
print(f'  10-90% quantile band empirical coverage: {coverage:.1f}% (nominal 80%)')

# ── 2. teacher-forced 1-step (real history up to t, one batched call) ────────
full = np.concatenate([y_train, y_test.astype(np.float32)])
o = len(train)
t0 = time.time()
ctxs = [full[t - CONTEXT:t] for t in range(o, o + test_size)]
tf_pred = np.array([r.forecast[0] for r in tfm.predict_batch(ctxs, horizon=1, return_quantiles=False,
                                                              use_symmetric_averaging=False)])
assert tf_pred.shape == (test_size,)
rows.append(dict(protocol='teacher_forced_1step', context=CONTEXT, MAE=mean_absolute_error(y_test, tf_pred), MAPE=mape(y_test, tf_pred)))
print(f'teacher 1-step ctx={CONTEXT}: MAE {rows[-1]["MAE"]:.2f} | MAPE {rows[-1]["MAPE"]:.2f}%  ({time.time()-t0:.1f}s)')

# ── 3. context-length ablation (honest protocol) ─────────────────────────────
for c in ABLATION:
    p = direct(c).forecast
    rows.append(dict(protocol='ablation_honest_672h', context=c, MAE=mean_absolute_error(y_test, p), MAPE=mape(y_test, p)))
    print(f'ablation ctx={c:>5}: MAE {rows[-1]["MAE"]:.2f} | MAPE {rows[-1]["MAPE"]:.2f}%')

results = pd.DataFrame(rows)
results.to_csv(HERE / 'results.csv', index=False)
pd.DataFrame({'pred': pred, 'q10': q10, 'q90': q90, 'tf_pred': tf_pred, 'actual': y_test},
             index=test.index).to_csv(HERE / 'timesfm_pred.csv')

# ── XGBoost overlay: best trained model, same 672h rollout as the notebook ───
# ponytail: copied from ../app.py run_xgb; 672 single-row predicts, fast enough
xgb = joblib.load(ROOT / 'xgb_forecaster.pkl')
feat_cols = ['hour', 'dayofweek', 'month', 'quarter', 'is_weekend',
             'lag_1', 'lag_24', 'lag_48', 'lag_168',
             'rolling_mean_24', 'rolling_std_24', 'rolling_mean_168']
hist = train['energy'].copy()
xgb_pred = []
for ts in test.index:
    row = {'hour': ts.hour, 'dayofweek': ts.dayofweek, 'month': ts.month, 'quarter': ts.quarter,
           'is_weekend': int(ts.dayofweek >= 5),
           'lag_1': hist.iloc[-1], 'lag_24': hist.iloc[-24], 'lag_48': hist.iloc[-48], 'lag_168': hist.iloc[-168],
           'rolling_mean_24': hist.iloc[-24:].mean(), 'rolling_std_24': hist.iloc[-24:].std(),
           'rolling_mean_168': hist.iloc[-168:].mean()}
    yhat = float(xgb.predict(pd.DataFrame([row])[feat_cols])[0])
    xgb_pred.append(yhat)
    hist.loc[ts] = yhat
xgb_pred = np.array(xgb_pred)
print(f'XGBoost rollout (reference): MAE {mean_absolute_error(y_test, xgb_pred):.2f} | MAPE {mape(y_test, xgb_pred):.2f}%')

# ── plot: full 4 weeks, honest protocol ──────────────────────────────────────
idx = test.index
plt.figure(figsize=(14, 5))
plt.plot(idx, y_test, label='Actual', linewidth=1.5, color='k')
plt.fill_between(idx, q10, q90, alpha=0.2, color='C2', label='TimesFM 10-90% quantiles')
plt.plot(idx, pred, label=f'TimesFM 3.0 zero-shot (MAE {rows[0]["MAE"]:.0f})', linestyle='--', color='C2')
plt.plot(idx, xgb_pred, label=f'XGBoost rollout (MAE {mean_absolute_error(y_test, xgb_pred):.0f})', linestyle='--', color='C0', alpha=0.8)
plt.legend()
plt.ylabel('MW')
plt.title(f'AEP load, 672 h held out, no re-anchoring — TimesFM context {CONTEXT} h')
plt.tight_layout()
plt.savefig(HERE / 'forecast_timesfm.png', dpi=150)
print(results.to_string(index=False))
