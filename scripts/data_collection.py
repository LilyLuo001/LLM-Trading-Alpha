import yfinance as yf
import pandas as pd
import numpy as np
import os

def download_stock_data(tickers, start_date="2010-01-01", end_date="2024-01-01", save_path="data/stock_returns.csv"):
    """Downloads stock price data, calculates log returns, and saves to CSV."""

    # Ensure the data folder exists
    if not os.path.exists("data"):
        os.makedirs("data")

    try:
        print("Downloading stock data...")
        data = yf.download(tickers, start=start_date, end=end_date)

        # Print available columns for debugging
        print("Available columns in data:", data.columns)

        # Fix MultiIndex issue: Select only 'Close' prices
        if ('Close', tickers[0]) in data.columns:  # Check if multi-index
            close_prices = data['Close']  # Extract 'Close' price level only
        else:
            print("Error: 'Close' price data not found. Check data format.")
            return

        # Compute log returns
        returns = close_prices.pct_change().apply(lambda x: np.log(1 + x))

        # Save dataset
        returns.to_csv(save_path)
        print(f"✅ Stock return data saved successfully to {save_path}")

    except Exception as e:
        print(f"❌ Error during data collection: {e}")

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    download_stock_data(tickers)

