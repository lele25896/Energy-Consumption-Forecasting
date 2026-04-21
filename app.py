import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

# ── Config ───────────────────────────────────────────────────────────────────
DATA_PATH  = 'AEP_hourly.csv'
TEST_WEEKS = 4
SEQ_LEN    = 168
N_FEATURES = 8
DEVICE     = torch.device('cpu')

# ── Model definitions (must match the notebook) ───────────────────────────────
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class PatchTST(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, patch_len=16, stride=8,
                 n_features=N_FEATURES, d_model=128, nhead=8,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.patch_len   = patch_len
        self.stride      = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_emb    = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout     = nn.Dropout(dropout)
        self.head        = nn.Linear(self.num_patches * d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        patches = x.unfold(1, self.patch_len, self.stride).permute(0, 1, 3, 2)
        patches = patches.reshape(B, self.num_patches, -1)
        tokens  = self.dropout(self.patch_proj(patches) + self.pos_emb)
        out     = self.transformer(tokens)
        return self.head(out.reshape(B, -1))

# ── Feature engineering ───────────────────────────────────────────────────────
def build_cyclic_features(df):
    feats = pd.DataFrame(index=df.index)
    feats['energy']     = df['energy']
    feats['sin_hour']   = np.sin(2 * np.pi * df.index.hour / 24)
    feats['cos_hour']   = np.cos(2 * np.pi * df.index.hour / 24)
    feats['sin_dow']    = np.sin(2 * np.pi * df.index.dayofweek / 7)
    feats['cos_dow']    = np.cos(2 * np.pi * df.index.dayofweek / 7)
    feats['sin_month']  = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    feats['cos_month']  = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    feats['is_weekend'] = (df.index.dayofweek >= 5).astype(float)
    return feats

def make_xgb_features(df):
    d = df.copy()
    d['hour']       = d.index.hour
    d['dayofweek']  = d.index.dayofweek
    d['month']      = d.index.month
    d['quarter']    = d.index.quarter
    d['is_weekend'] = (d.index.dayofweek >= 5).astype(int)
    for lag in [1, 24, 48, 168]:
        d[f'lag_{lag}'] = d['energy'].shift(lag)
    d['rolling_mean_24']  = d['energy'].shift(1).rolling(24).mean()
    d['rolling_std_24']   = d['energy'].shift(1).rolling(24).std()
    d['rolling_mean_168'] = d['energy'].shift(1).rolling(168).mean()
    return d

# ── Data & model loading (cached) ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'], index_col='Datetime')
    df = df.sort_index()
    df.columns = ['energy']
    test_size = TEST_WEEKS * 7 * 24
    return df.iloc[:-test_size], df.iloc[-test_size:]

@st.cache_resource
def load_models():
    scaler  = joblib.load('energy_scaler.pkl')
    xgb     = joblib.load('xgb_forecaster.pkl')

    lstm = LSTMForecaster().to(DEVICE)
    lstm.load_state_dict(torch.load('lstm_forecaster.pt', map_location=DEVICE))
    lstm.eval()

    # Infer seq_len from checkpoint so it works regardless of which run was saved
    pt_sd        = torch.load('patchtst_forecaster.pt', map_location=DEVICE)
    num_patches  = pt_sd['pos_emb'].shape[1]
    pt_seq_len   = (num_patches - 1) * 8 + 16  # stride=8, patch_len=16
    patchtst     = PatchTST(seq_len=pt_seq_len).to(DEVICE)
    patchtst.load_state_dict(pt_sd)
    patchtst.eval()
    patchtst._seq_len = pt_seq_len  # store for inference

    return scaler, xgb, lstm, patchtst

# ── Inference helpers ─────────────────────────────────────────────────────────
def run_seq_model(model, scaler, train, test):
    seq_len = getattr(model, '_seq_len', SEQ_LEN)

    train_f = build_cyclic_features(train).copy()
    test_f  = build_cyclic_features(test).copy()
    train_f['energy'] = scaler.transform(train_f[['energy']])
    test_f['energy']  = scaler.transform(test_f[['energy']])

    train_arr = train_f.values.astype(np.float32)
    test_arr  = test_f.values.astype(np.float32)
    full      = np.concatenate([train_arr[-seq_len:], test_arr])

    n       = len(test_arr)
    windows = np.stack([full[i:i + seq_len] for i in range(n)])
    tensor  = torch.tensor(windows, dtype=torch.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, n, 128):
            preds.append(model(tensor[i:i + 128].to(DEVICE)).cpu().numpy())

    return scaler.inverse_transform(np.concatenate(preds).reshape(-1, 1)).flatten()

def run_xgb(model, train, test):
    df_all  = pd.concat([train, test])
    df_feat = make_xgb_features(df_all).dropna()
    feat_cols = [c for c in df_feat.columns if c != 'energy']
    return model.predict(df_feat[feat_cols].iloc[-len(test):])

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Energy Forecast', layout='wide')
st.title('Energy Consumption Forecasting')
st.caption('AEP hourly dataset — comparison of Prophet, XGBoost, LSTM, and PatchTST')

with st.sidebar:
    st.header('Settings')
    model_choice = st.selectbox(
        'Model',
        ['LSTM (best)', 'XGBoost', 'PatchTST'],
    )
    show_days = st.slider('Days to display', min_value=1, max_value=28, value=14)
    st.markdown('---')
    st.markdown(
        '**Models trained on** ~120k hours of AEP data (2004–2018).  \n'
        'Test set: last 4 weeks (672 h).'
    )

train, test = load_data()

with st.spinner('Loading models…'):
    scaler, xgb, lstm, patchtst = load_models()

with st.spinner('Running inference…'):
    if model_choice == 'LSTM (best)':
        pred = run_seq_model(lstm, scaler, train, test)
        label = 'LSTM'
    elif model_choice == 'XGBoost':
        pred = run_xgb(xgb, train, test)
        label = 'XGBoost'
    else:
        pred = run_seq_model(patchtst, scaler, train, test)
        label = 'PatchTST'

actual = test['energy'].values
mae    = mean_absolute_error(actual, pred)
mape   = np.mean(np.abs((actual - pred) / actual)) * 100

# ── Metrics ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric('Model', label)
col2.metric('MAE (MW)', f'{mae:,.1f}')
col3.metric('MAPE', f'{mape:.2f}%')

# ── Chart ─────────────────────────────────────────────────────────────────────
n_hours = show_days * 24
idx     = test.index[-n_hours:]
chart_df = pd.DataFrame({
    'Actual': actual[-n_hours:],
    label:    pred[-n_hours:],
}, index=idx)

st.subheader(f'Forecast vs Actual — last {show_days} days of test set')
st.line_chart(chart_df, height=400)

# ── Error distribution ────────────────────────────────────────────────────────
with st.expander('Error distribution'):
    errors = pred[-n_hours:] - actual[-n_hours:]
    err_df = pd.DataFrame({'Error (MW)': errors}, index=idx)
    st.bar_chart(err_df, height=250)
    st.caption(
        f'Mean error: {errors.mean():+.1f} MW  |  '
        f'Std: {errors.std():.1f} MW  |  '
        f'Max abs: {np.abs(errors).max():.1f} MW'
    )
