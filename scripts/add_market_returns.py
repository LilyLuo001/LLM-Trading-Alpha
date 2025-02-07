import yfinance as yf
import pandas as pd
import numpy as np  # 🔹 Fix: Import NumPy

# 📥 Download S&P 500 returns
print("📥 Downloading S&P 500 returns...")
sp500 = yf.download("^GSPC", start="2010-01-01", end="2024-01-01")

# 🛠️ Debug: Check available columns
print("Available columns in downloaded S&P 500 data:", sp500.columns)

# 🔹 Extract the closing price properly
if ("Adj Close" in sp500.columns):  
    sp500 = sp500["Adj Close"]
elif ("Close", "^GSPC") in sp500.columns:  
    sp500 = sp500[("Close", "^GSPC")]
else:
    raise ValueError("❌ Error: 'Adj Close' or 'Close' column not found in S&P 500 data.")

# ✅ Compute market returns
sp500_returns = sp500.pct_change().apply(lambda x: np.log(1 + x))
sp500_returns.name = "SPY"

# 📂 Load stock return data
file_path = "data/stock_returns.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 🔄 Merge S&P 500 returns into the stock return dataset
df = df.join(sp500_returns, how="left")

# 💾 Save updated dataset
df.to_csv(file_path)
print("✅ S&P 500 returns added successfully to stock_returns.csv")

