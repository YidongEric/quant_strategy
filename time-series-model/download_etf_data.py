# First, update yfinance to the latest version

# Then modify your code to use the improved ticker object approach
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# List of ETFs/ETNs
etfs = ['DIA', 'IWM', 'QQQ', 'SPY', 'VXX', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

# Set date range (last 5 years)
end_date = datetime.now()
start_date = end_date - timedelta(days=10*365)

# Create directory to store data
if not os.path.exists('etf_data'):
    os.makedirs('etf_data')

# Download data for each ETF
for etf in etfs:
    try:
        print(f"Downloading data for {etf}...")
        
        # Create a Ticker object (more robust approach)
        ticker = yf.Ticker(etf)
        
        # Download the data using the history method
        data = ticker.history(start=start_date, end=end_date)
        
        # Check if we got data
        if len(data) > 0:
            # Save to CSV
            filename = f'etf_data/{etf}_data.csv'
            data.to_csv(filename)
            print(f"Successfully saved {etf} data with {len(data)} rows to {filename}")
        else:
            print(f"No data returned for {etf}.")
            
    except Exception as e:
        print(f"Error downloading {etf}: {str(e)}")

print("\nDownload complete!")