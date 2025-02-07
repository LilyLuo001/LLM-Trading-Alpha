import numpy as np
import matplotlib.pyplot as plt

# Load data
returns = pd.read_csv("data/stock_returns.csv", index_col=0)
factors = pd.read_csv("data/fama_french_factors.csv")

# Merge datasets
df = returns.merge(factors, on="Date", how="left")

# Compute rolling Sharpe Ratio
df["Sharpe_Ratio"] = df["AI_Strategy_Returns"].rolling(252).mean() / df["AI_Strategy_Returns"].rolling(252).std()

# Save processed dataset
df.to_csv("data/processed_returns.csv")
