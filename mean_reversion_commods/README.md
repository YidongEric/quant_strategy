# Mean Reversion Strategy on Brent/WTI Futures Spreads

## Overview
This project develops a simple **mean reversion** trading strategy on the spread between **Brent** and **WTI** futures (M1–M4 contracts).  
It includes **data cleaning**, **rolling**, **strategy backtesting**, and **performance evaluation**.

## How to Use

1. **Clone or download** this repository.
2. Place the following files under the **same folder**:
   - `data_rolling.py`
   - `mean_reversion.py`
   - `output_continuous.csv`
3. Run `data_rolling.py` first:
   ```bash
   python data_rolling.py
   ```
   This will generate a new file called `rolled_continuous.csv` in the same folder.

4. Then run `mean_reversion.py`:
   ```bash
   python mean_reversion.py
   ```
   This will backtest the mean reversion strategy and output performance metrics and charts.

## Notes

- `output_continuous.csv` is **raw unrolled futures data** downloaded from a **paid data provider**.
- `data_rolling.py` handles **data cleaning**, **rolling logic**, and **holiday adjustments** before strategy application.
- The strategy evaluates the spread between:
  - **Brent M1 – WTI M1**
  - **Brent M2 – WTI M2**
  - **Brent M3 – WTI M3**
  - **Brent M4 – WTI M4**

## Results

Example backtest results:
- **Total Return**: 15%–17% across different contract months
- **Sharpe Ratio**: ~0.61–0.70
- **Annualized Volatility**: ~3.3%
- **Max Drawdown**: ~6%–7%

