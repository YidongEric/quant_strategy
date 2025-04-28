import pandas as pd
import numpy as np
from collections import defaultdict

"""
Notes on methodology:

1. Calendar-day roll on or before the 25th:
   • For each year-month, find the 25th trading day; if the 25th is a holiday or weekend,
     roll on the last available date before the 25th.
   • This ensures consistency even when the 25th doesn’t exist in the data.

2. Panama (difference) back-adjustment:
   • On each roll date, compute the price gap:
       diff = next_month_price − current_month_price
   • Accumulate a running cum_diff and add it to all earlier raw prices.
   • This “removes” the artificial gap, preserving point-based P&L calculations.

3. Interior-only interpolation:
   • Before rolling, we linearly interpolate internal NaNs in each contract series.
   • We intentionally do NOT forward- or backward-fill leading/trailing NaNs,
     so “pre-launch” or post-roll tails remain blank.

4. Rolling each contract leg M0→M1, M1→M2, …, M11→M12:
   • We keep all M0…M12 through the roll process so that M11 can roll into M12.
   • After completing all back-adjustments, we drop the M12 columns
     (they are only used as roll targets).

5. Output:
   • The final CSV has Date + M0…M11 columns, each now containing a
     back-adjusted continuous series per contract leg.
"""

# 1) Load & sort
df = (
    pd.read_csv('output_continuous.csv', parse_dates=['Date'])
      .sort_values('Date')
      .reset_index(drop=True)
)
dates = df['Date']

# 2) Identify all futures‐contract columns (M0…M12)
contract_cols = [c for c in df.columns if ' M' in c]

# 3) Interior-only interpolation
df[contract_cols] = df[contract_cols].astype(float).interpolate(method='linear')

# 4) Group by underlying
groups = defaultdict(list)
for col in contract_cols:
    und, _ = col.rsplit(' ', 1)
    groups[und].append(col)
for und in groups:
    groups[und].sort(key=lambda c: int(c.split('M')[-1]))  # M0…M12

# 5) Build effective roll‐dates lookup
#    For each year-month, pick the 25th if present; otherwise the last prior date.
roll_dates = {}
for (y, m), sub in df.groupby([df['Date'].dt.year, df['Date'].dt.month]):
    month_dates = sub['Date']
    target = pd.Timestamp(year=y, month=m, day=25)
    if target in set(month_dates):
        roll_dates[(y, m)] = target
    else:
        # max date strictly before the 25th
        prior = month_dates[month_dates < target]
        if len(prior):
            roll_dates[(y, m)] = prior.max()
        # else: no roll for that month (e.g. first-ever month), skip it

# 6) Panama back-adjust M0…M11, in-place
output = df.copy()
for und, months in groups.items():
    # allow M11→M12 rolls
    for start_idx in range(len(months)-1):
        col = months[start_idx]
        idx = start_idx
        cum_diff = 0.0
        cont = []

        for t, date in enumerate(dates):
            ym = (date.year, date.month)
            # if this date is our effective roll date for its month, and next exists
            if roll_dates.get(ym, None) == date and (idx+1) < len(months):
                old_p = output.at[t, months[idx]]
                new_p = output.at[t, months[idx+1]]
                if pd.notna(old_p) and pd.notna(new_p):
                    cum_diff += (new_p - old_p)
                    idx += 1
                else:
                    break  # can't roll, stop leg here

            raw = output.at[t, months[idx]]
            cont.append(raw + cum_diff if pd.notna(raw) else np.nan)

        # fill only interior NaNs
        cont_s = pd.Series(cont, index=dates[:len(cont)]).interpolate(method='linear')

        # overwrite that column for its valid range
        output.loc[:len(cont_s)-1, col] = cont_s.values
        # beyond that, leave NaN

# 7) Drop M12 columns now that every M0–M11 has rolled into its M12
m12 = [c for c in contract_cols if c.endswith(' M12')]
output.drop(columns=m12, inplace=True)

# 8) Save
output.to_csv('rolled_continuous.csv', index=False)
