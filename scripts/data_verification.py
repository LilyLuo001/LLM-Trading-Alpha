import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = "data/stock_returns.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Summary statistics
print(df.describe())

# Plot stock return trends
df.plot(figsize=(12, 6), title="Log Returns of Selected Stocks")
plt.xlabel("Date")
plt.ylabel("Log Returns")
plt.legend(title="Stock")
plt.grid()
plt.show()
