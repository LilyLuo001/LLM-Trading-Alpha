import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# Load the data
df = pd.read_csv("data/stock_returns.csv", index_col=0, parse_dates=True)

# Compute skewness for each stock
skewness = df.apply(skew)
print("Skewness of Returns:")
print(skewness)

# Plot distribution for each stock
plt.figure(figsize=(12, 8))
for i, col in enumerate(df.columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[col], bins=50, kde=True, color='blue')
    plt.title(f"{col} Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

