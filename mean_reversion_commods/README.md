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
### Performance Metrics
The strategy performance across different contract months:

| Month | Start Date  | Total Return | Ann. Volatility | Sharpe Ratio | Sortino Ratio | Calmar Ratio | Win Rate (%) | Max Cons. Losses | Max Drawdown |
|-------|-------------|--------------|----------------|--------------|---------------|--------------|--------------|------------------|--------------|
| M1    | 2018-01-02  | 16.95%       | 3.35%          | 0.68         | 0.94          | 2.58         | 37.76        | 9                | -6.58%       |
| M2    | 2018-01-02  | 17.08%       | 3.29%          | 0.70         | 1.03          | 2.59         | 37.59        | 9                | -6.60%       |
| M3    | 2018-01-02  | 15.22%       | 3.31%          | 0.62         | 0.87          | 2.22         | 36.69        | 9                | -6.85%       |
| M4    | 2018-01-02  | 15.11%       | 3.31%          | 0.61         | 0.87          | 2.17         | 37.41        | 9                | -6.96%       |

### Results Analysis

#### Key Findings
- **Consistent Performance**: All contract months show similar performance patterns, indicating strategy robustness across different time horizons.
- **Returns**: Total returns range from 15.11% to 17.08%, with shorter-term contracts (M1 and M2) slightly outperforming longer-term contracts.
- **Risk-Adjusted Metrics**: The strategy shows moderate Sharpe ratios (0.61-0.70) and stronger Sortino ratios (0.87-1.03), suggesting better downside risk control.

#### Risk Profile
- **Low Volatility**: Annual volatility remains consistent at ~3.3% across all contract months, indicating stable strategy behavior.
- **Limited Drawdowns**: Maximum drawdowns are contained between -6.58% and -6.96%, demonstrating effective risk management.
- **Strong Calmar Ratios**: Values above 2.0 across all contracts show good returns relative to maximum risk taken.

#### Trading Characteristics
- **Moderate Win Rate**: ~37% win rate is common for mean reversion strategies, which typically rely on fewer large winning trades to offset more frequent small losses.
- **Psychological Challenge**: Maximum consecutive losses of 9 across all contracts indicates periods of sustained underperformance that would require disciplined execution.

#### Strategic Implications
- The M1 and M2 contracts slightly outperform longer-dated contracts, suggesting that near-term contracts may respond more efficiently to mean reversion signals.
- The consistent performance across all contract months validates the underlying mean reversion hypothesis in the Brent-WTI spread relationship.
- While the strategy generates positive returns with controlled risk, there remains room for optimization to push Sharpe ratios above 1.0.
