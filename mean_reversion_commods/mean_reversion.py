import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Mean Reversion Backtest Script
# -----------------------------

# 1) Load back-adjusted futures data, parse dates with day-first format
df = pd.read_csv(
    'rolled_continuous.csv',
    parse_dates=['Date'],
    dayfirst=True
).sort_values('Date')
df.set_index('Date', inplace=True)

# 2) Strategy parameters
months = [1, 2, 3, 4]              # M1 to M4 spreads
entry_z = 2.0                      # entry when z > ±2
exit_z = 0.0                       # exit at z = 0
window = 20                        # lookback for mean/std
slippage_per_side = 0.05           # points per trade side
commission_per_side = 0.02         # points per trade side
cost_per_side = slippage_per_side + commission_per_side

# 3) Container for results
results = []

# 4) Loop through each M1–M4 spread
for m in months:
    bcol = f'Brent M{m}'
    wcol = f'WTI M{m}'
    
    # 4a) Align on first non-NaN date for both series
    sub = df[[bcol, wcol]].dropna()
    if sub.empty:
        continue
    start_date = sub.index.min()
    data = sub.copy()
    
    # 4b) Compute spread and rolling z-score
    data['spread'] = data[bcol] - data[wcol]
    data['mean']   = data['spread'].rolling(window).mean()
    data['std']    = data['spread'].rolling(window).std()
    data['z']      = (data['spread'] - data['mean']) / data['std']
    
    # 4c) Generate positions
    data['pos'] = 0
    pos = 0
    entries, exits = [], []
    for i in range(window, len(data)):
        z = data['z'].iat[i]
        dt = data.index[i]
        if pos == 0 and z >  entry_z:
            pos = -1
            entries.append(dt)
        elif pos == 0 and z < -entry_z:
            pos = 1
            entries.append(dt)
        elif pos ==  1 and z >= exit_z:
            pos = 0
            exits.append(dt)
        elif pos == -1 and z <= exit_z:
            pos = 0
            exits.append(dt)
        data.at[dt, 'pos'] = pos
    
    # 4d) Calculate P&L and equity
    data['chg']    = data['spread'].diff()
    data['trades'] = data['pos'].diff().abs()
    data['costs']  = data['trades'] * cost_per_side
    data['pnl']    = data['pos'].shift(1) * data['chg'] - data['costs']
    data['equity'] = data['pnl'].cumsum().fillna(0)
    
    # 4e) Compute performance metrics
    total_ret = data['equity'].iat[-1]
    ann_vol   = data['pnl'].std() * np.sqrt(252)
    sharpe    = (data['pnl'].mean() * 252) / ann_vol if ann_vol else np.nan
    max_dd    = (data['equity'] - data['equity'].cummax()).min()
    
    results.append({
        'Month':         f'M{m}',
        'Start Date':    start_date.strftime('%Y-%m-%d'),
        'Total Return':  total_ret,
        'Ann. Volatility': ann_vol,
        'Sharpe Ratio':  sharpe,
        'Max Drawdown':  max_dd
    })
    
    # 4f) Plot equity curve
    plt.figure(figsize=(8,4))
    data['equity'].plot(title=f'Equity Curve: Brent M{m} – WTI M{m}')
    plt.xlabel('Date'); plt.ylabel('Cumulative P&L'); plt.grid(True)
    plt.show()
    
    # 4g) Plot spread with entry/exit markers
    plt.figure(figsize=(8,4))
    data['spread'].plot(label='Spread')
    plt.scatter(entries, data.loc[entries, 'spread'], marker='^', color='green', label='Entry')
    plt.scatter(exits,   data.loc[exits,   'spread'], marker='v', color='red',   label='Exit')
    plt.title(f'Spread & Signals: Brent M{m} – WTI M{m}')
    plt.xlabel('Date'); plt.legend(); plt.grid(True)
    plt.show()

# 5) Print summary table
perf = pd.DataFrame(results)
print(perf.to_string(index=False))
