import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 📂 Load data
file_path = "data/stock_returns.csv"
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

# 🛠️ Ensure SPY exists and handle missing data
if "SPY" not in df.columns:
    print("⚠️ Warning: 'SPY' (market return) not found in dataset. Skipping CAPM regression.")
else:
    df["SPY"] = df["SPY"].replace([float("inf"), float("-inf")], None)  # Convert infinite values to NaN
    df = df.dropna(subset=["SPY"])  # Drop rows with missing market returns

    beta_values = {}  # Store Beta coefficients

    for stock in df.columns:
        if stock == "SPY":
            continue

        y = df[stock]  
        X = sm.add_constant(df["SPY"])
        valid_data = pd.concat([y, X], axis=1).dropna()

        if valid_data.empty:
            print(f"⚠️ Warning: No valid data for {stock}. Skipping regression.")
            continue

        model = sm.OLS(valid_data[stock], valid_data[["const", "SPY"]]).fit()

        # Store Beta
        beta_values[stock] = model.params["SPY"]

        # 📜 Print results
        print(f"📊 CAPM Regression Results for {stock}:")
        print(model.summary())
        print("\n")

    # 📊 Visualizing Beta Coefficients
    plt.figure(figsize=(10, 5))
    plt.bar(beta_values.keys(), beta_values.values(), color="blue")
    plt.axhline(y=1, color="red", linestyle="dashed", label="Market Beta = 1")
    plt.xlabel("Stocks")
    plt.ylabel("Beta (Market Sensitivity)")
    plt.title("Market Beta for Each Stock")
    plt.legend()
    plt.show()

    # 💾 Save Beta values
    beta_df = pd.DataFrame.from_dict(beta_values, orient="index", columns=["Beta"])
    beta_df.to_csv("data/beta_values.csv")
    print("✅ Beta values saved to 'data/beta_values.csv'")
