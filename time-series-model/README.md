# Advanced ETF Return Prediction

This project demonstrates a complete pipeline for downloading ETF data, engineering features, training multiple models (VAR, XGBoost, LSTM), and evaluating performance using cross-sectional Information Coefficient (IC).

## Repository Structure
```
├── download_etf_data.py         # Script to fetch historical ETF data via yfinance
├── advanced_etf_analysis.py     # Main analysis: feature engineering, modeling, evaluation
├── etf_data/                    # Directory to store downloaded CSV price data
├── results/                     # Outputs: metrics JSON, CSV of daily CS-IC, plots
└── requirements.txt             # Python dependencies
```

## Setup
1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd <repo_folder>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Download ETF Data
Run the downloader to fetch last 10 years of daily ETF prices.
```bash
python download_etf_data.py
```
This will create the `etf_data/` folder and save one CSV per ticker.


### 2. Run Analysis
Once data is available, run the main analysis script:
```bash
python advanced_etf_analysis.py
```
This will:
- Load and preprocess price data
- Compute features (returns, log returns, volume change, volatility, MA5/MA20, RSI)
- Prepare training and test sets
- Train three models:
  - **VAR**: Vector Autoregression on returns
  - **XGBoost**: Gradient-boosted regression per ETF
  - **LSTM**: Deep recurrent network with 2 LSTM layers
- Evaluate each model’s performance:
  - **RMSE** (error magnitude)
  - **R²** (explained variance)
  - **IC** (rank correlation)
  - **Cross-Sectional IC** (average daily cross-sectional correlation)
- Print a summary table to console
- Save detailed metrics in `results/metrics_summary.json`
- Save daily CS-IC series in `results/*_cs_ic.csv`
- Generate a cumulative CS-IC plot at `results/cumulative_cs_ic.png`

## Interpretation
- **RMSE**: Lower is better (forecast error)
- **R²**: Higher is better (variance explained)
- **IC**: Higher correlation implies stronger predictive rank ordering
- **Cross-Sectional IC**: How well model predicts relative performance across ETFs each day; cumulative plot tracks consistency over time.

## Notes
- Adjust model hyperparameters in `advanced_etf_analysis.py` as needed.

---

**Author:** Eric Wang
**Date:** 2025-04-28
