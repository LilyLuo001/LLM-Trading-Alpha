import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew, kurtosis

# Load data
file_path = "data/stock_returns.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# Compute descriptive statistics
stats = df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
stats['skewness'] = df.apply(lambda x: skew(x.dropna()), axis=0)
stats['kurtosis'] = df.apply(lambda x: kurtosis(x.dropna()), axis=0)

print("\n🔹 Descriptive Statistics:")
print(stats)

# Save statistics to CSV
stats.to_csv("data/descriptive_statistics.csv")
print("\n✅ Descriptive statistics saved to 'data/descriptive_statistics.csv'")

# Boxplot Visualization
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, orient="h", palette="coolwarm")
plt.title("Boxplot of Stock Returns")
plt.xlabel("Daily Return")
plt.show()

import pandas as pd
import statsmodels.api as sm

# 📂 Load data
file_path = "data/stock_returns.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 🔍 Check if SPY exists
if "SPY" not in df.columns:
    print("⚠️ Warning: 'SPY' (market return proxy) not found in dataset. Skipping CAPM regression.")
else:
    # 🛠️ Handle missing or infinite values in SPY
    df["SPY"] = df["SPY"].replace([float("inf"), float("-inf")], None)  # Convert infinite values to NaN
    df = df.dropna(subset=["SPY"])  # Drop rows with missing market returns

    # 📊 Run CAPM Regression for Each Stock
    for stock in df.columns:
        if stock == "SPY":
            continue  # Skip SPY itself in regression

        y = df[stock]  # Stock returns
        X = sm.add_constant(df["SPY"])  # Add constant for the intercept
        
        # 🛠️ Drop rows with missing data
        valid_data = pd.concat([y, X], axis=1).dropna()

        if valid_data.empty:
            print(f"⚠️ Warning: No valid data for {stock}. Skipping regression.")
            continue

        # 📈 Run regression
        model = sm.OLS(valid_data[stock], valid_data[["const", "SPY"]]).fit()

        # 📜 Print results
        print(f"📊 CAPM Regression Results for {stock}:")
        print(model.summary())
        print("\n")
