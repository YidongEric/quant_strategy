import os
import json
import warnings

# ─── Suppress TensorFlow INFO logs and oneDNN optimizations messages ─────────────────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ─── Config & Constants ─────────────────────────────────────────────────────────────────────────
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
sns.set_style("whitegrid")

ETFS       = ['DIA','IWM','QQQ','SPY','VXX','XLB','XLC','XLE','XLF','XLI','XLK','XLP','XLRE','XLU','XLV','XLY']
LOOKBACK   = 5
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Utility Functions ────────────────────────────────────────────────────────────────────────────
def calculate_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute rank-based Information Coefficient between two vectors."""
    mask = (~np.isnan(pred)) & (~np.isnan(actual))
    if mask.sum() == 0:
        return np.nan
    pr = pd.Series(pred[mask]).rank()
    ar = pd.Series(actual[mask]).rank()
    return np.corrcoef(pr, ar)[0,1]


def daily_cross_sectional_ic(pred_df: pd.DataFrame, true_df: pd.DataFrame) -> pd.Series:
    """For each date, compute cross-sectional IC across tickers."""
    dates = pred_df.index.intersection(true_df.index)
    ics = []
    for d in dates:
        p = pred_df.loc[d].values
        t = true_df.loc[d].values
        ics.append(calculate_ic(p, t))
    return pd.Series(ics, index=dates)

# ─── Data Loading & Feature Engineering ───────────────────────────────────────────────────────────
def load_and_preprocess_data() -> dict:
    all_data = {}
    for etf in ETFS:
        path = f'etf_data/{etf}_data.csv'
        if not os.path.exists(path):
            print(f"Warning: {path} not found; skipping {etf}.")
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        df['close_to_close'] = df['Close'].pct_change()
        df['returns']        = df['Close'].pct_change()
        df['log_returns']    = np.log(df['Close']).diff()
        df['volume_change']  = df['Volume'].pct_change()
        df['volatility']     = (df['High'] - df['Low']) / df['Open']
        df['MA5']            = df['Close'].rolling(window=5).mean()
        df['MA20']           = df['Close'].rolling(window=20).mean()
        delta = df['Close'].diff()
        gain  =  delta.clip(lower=0).rolling(14).mean()
        loss  = -delta.clip(upper=0).rolling(14).mean()
        df['RSI']            = 100 - (100/(1 + gain/loss))
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)
        all_data[etf] = df
        print(f"Loaded {etf}: {len(df)} rows")
    return all_data


def prepare_data(all_data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = None
    for df in all_data.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    features = pd.DataFrame(index=common_idx)
    target   = pd.DataFrame(index=common_idx)
    for etf, df in all_data.items():
        d = df.loc[common_idx]
        for col in ['returns','log_returns','volume_change','volatility','MA5','MA20','RSI']:
            features[f"{etf}_{col}"] = d[col]
        target[etf] = d['close_to_close'].shift(-1)
    features.fillna(0, inplace=True)
    target.fillna(0, inplace=True)
    return features, target

# ─── LSTM Helpers ─────────────────────────────────────────────────────────────────────────────────
def create_lstm_sequences(X: np.ndarray, y: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - lookback):
        Xs.append(X[i:i+lookback])
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys)


def build_lstm_model(input_shape: tuple[int,int], output_dim: int) -> Sequential:
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ─── Main Pipeline ──────────────────────────────────────────────────────────────────────────────────
def main():
    # 1. Load & preprocess data
    data = load_and_preprocess_data()
    features, target = prepare_data(data)

    # 2. Train-test split
    n     = len(features)
    split = int(0.8 * n)
    X_train, X_test = features.iloc[:split], features.iloc[split:]
    y_train, y_test = target.iloc[:split],   target.iloc[split:]

    # 3. Scale features
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── VAR ─────────────────────────────────────────────────────────────────────
    var_model = VAR(endog=y_train)
    var_res   = var_model.fit(maxlags=1)
    var_in    = var_res.fittedvalues
    var_out   = var_res.forecast(y_train.values[-var_res.k_ar:], steps=len(X_test))
    var_out_df = pd.DataFrame(var_out, index=y_test.index, columns=y_test.columns)

    # ── XGBoost ─────────────────────────────────────────────────────────────────
    xgb_in_df  = pd.DataFrame(index=y_train.index, columns=y_train.columns)
    xgb_out_df = pd.DataFrame(index=y_test.index,  columns=y_test.columns)
    for col in y_train.columns:
        model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05)
        model.fit(X_train_s, y_train[col])
        xgb_in_df[col]  = model.predict(X_train_s)
        xgb_out_df[col] = model.predict(X_test_s)

    # ── LSTM ────────────────────────────────────────────────────────────────────
    Xtr_seq, ytr_seq = create_lstm_sequences(X_train_s, y_train.values, LOOKBACK)
    Xte_seq, yte_seq = create_lstm_sequences(X_test_s,  y_test.values,  LOOKBACK)
    lstm_model       = build_lstm_model((LOOKBACK, X_train_s.shape[1]), y_train.shape[1])
    lstm_model.fit(Xtr_seq, ytr_seq, epochs=20, batch_size=32, verbose=0)
    lstm_in_preds    = lstm_model.predict(Xtr_seq)
    lstm_out_preds   = lstm_model.predict(Xte_seq)
    lstm_in_df       = pd.DataFrame(lstm_in_preds,  index=y_train.index[LOOKBACK:], columns=y_train.columns)
    lstm_out_df      = pd.DataFrame(lstm_out_preds, index=y_test.index[LOOKBACK:],  columns=y_test.columns)

    # ── Evaluation ───────────────────────────────────────────────────────────────
    summaries = {}
    cs_series = {}

    def evaluate(pred_df, true_df, name, period):
        idx = pred_df.index.intersection(true_df.index)
        y_p = pred_df.loc[idx].values
        y_t = true_df.loc[idx].values
        rmse = np.sqrt(mean_squared_error(y_t.flatten(), y_p.flatten()))
        r2   = r2_score(y_t.flatten(), y_p.flatten())
        ic   = calculate_ic(y_p.flatten(), y_t.flatten())
        cs   = daily_cross_sectional_ic(pred_df, true_df)
        # store summary and series
        summaries.setdefault(name, {})[period] = {
            'RMSE':   rmse,
            'R2':     r2,
            'IC':     ic,
            'CS-IC':  cs.mean()
        }
        cs_series[f"{name}_{period}"] = cs

    # Evaluate each model
    evaluate(var_in,   y_train, 'VAR',     'in_sample')
    evaluate(var_out_df,y_test,  'VAR',     'out_sample')
    evaluate(xgb_in_df, y_train, 'XGBoost', 'in_sample')
    evaluate(xgb_out_df,y_test,  'XGBoost', 'out_sample')
    evaluate(lstm_in_df,y_train, 'LSTM',    'in_sample')
    evaluate(lstm_out_df,y_test, 'LSTM',    'out_sample')

    # 4. Print summary table
    print("\nModel     Period       RMSE    R2     IC     CS-IC")
    print("--------------------------------------------------")
    for model, periods in summaries.items():
        for period, stats in periods.items():
            print(f"{model:8} {period:11} {stats['RMSE']:.4f}  {stats['R2']:.4f}  {stats['IC']:.4f}  {stats['CS-IC']:.4f}")

    # 5. Plot cumulative CS-IC
    plt.figure(figsize=(10,6))
        # 5. Plot cumulative CS-IC (fixed key splitting)
    plt.figure(figsize=(10,6))
    for key, series in cs_series.items():
        # split only on last underscore to separate model from period
        model, period = key.rsplit('_', 1)
        style = '-' if period == 'in_sample' else '--'
        plt.plot(series.cumsum(), linestyle=style, linewidth=2, label=f"{model} {period.replace('_', ' ')}")
    plt.axhline(0, color='black', lw=0.5)
    plt.title("Cumulative Cross-Sectional IC")
    plt.xlabel("Date")
    plt.ylabel("Cumulative CS-IC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/cumulative_cs_ic.png")
    plt.show()

    # 6. Save metrics JSON (no Timestamp keys)
    with open(f"{RESULTS_DIR}/metrics_summary.json", 'w') as f:
        json.dump(summaries, f, indent=4)

    # Save CS-IC series to CSV
    for name, series in cs_series.items():
        series.to_csv(f"{RESULTS_DIR}/{name}_cs_ic.csv", header=['CS-IC'])


    # 6. Save metrics JSON (no Timestamp keys)
    with open(f"{RESULTS_DIR}/metrics_summary.json", 'w') as f:
        json.dump(summaries, f, indent=4)

    # Save CS-IC series to CSV
    for name, series in cs_series.items():
        series.to_csv(f"{RESULTS_DIR}/{name}_cs_ic.csv", header=['CS-IC'])

if __name__ == '__main__':
    main()
