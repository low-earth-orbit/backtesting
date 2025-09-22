# Drawdown and recovery analysis for Canadian equities
import pandas as pd
import numpy as np

canada_path = "dataset/canada-price-cad.xls"


def load_and_clean_data(file_path):
    df = pd.read_excel(file_path)
    # Find the row with 'Date' header
    date_row_idx = None
    for idx, row in df.iterrows():
        if isinstance(row["Unnamed: 0"], str) and "Date" in row["Unnamed: 0"]:
            date_row_idx = idx
            break
    if date_row_idx is None:
        raise ValueError("Could not find 'Date' header in the file")
    df = df.iloc[date_row_idx:].reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    # Find where the data ends
    data_end_idx = None
    for idx, row in df.iterrows():
        date_val = str(row["Date"]).lower()
        if "copyright" in date_val or date_val == "nan" or date_val == "":
            data_end_idx = idx
            break
    if data_end_idx is not None:
        df = df.iloc[:data_end_idx]
    df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y")
    df = df.set_index("Date")
    price_col = df.columns[0]
    df[price_col] = pd.to_numeric(
        df[price_col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
        errors="coerce",
    )
    df = df.dropna()
    df = df.sort_index()
    return df[price_col]


prices = load_and_clean_data(canada_path)

# Calculate drawdown series
cummax = prices.cummax()
drawdown = (prices - cummax) / cummax


# Find drawdown events and recovery

# Analyze 12m return after first dip below each threshold
thresholds = [0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50]
results = []
# All time high (drawdown == 0)
ath_indices = np.where(drawdown == 0)[0]
for idx in ath_indices:
    dip_date = drawdown.index[idx]
    dd_val = drawdown.iloc[idx]
    ret_6m = (
        (prices.iloc[idx + 6] / prices.iloc[idx]) - 1
        if idx + 6 < len(prices)
        else np.nan
    )
    ret_12m = (
        (prices.iloc[idx + 12] / prices.iloc[idx]) - 1
        if idx + 12 < len(prices)
        else np.nan
    )
    ret_24m = (
        (prices.iloc[idx + 24] / prices.iloc[idx]) - 1
        if idx + 24 < len(prices)
        else np.nan
    )
    ret_36m = (
        (prices.iloc[idx + 36] / prices.iloc[idx]) - 1
        if idx + 36 < len(prices)
        else np.nan
    )
    results.append(
        {
            "threshold": 0.0,
            "dip_date": dip_date,
            "drawdown": dd_val,
            "6m_return": ret_6m,
            "12m_return": ret_12m,
            "24m_return": ret_24m,
            "36m_return": ret_36m,
        }
    )
# All other thresholds
for threshold in thresholds[1:]:
    dips = (drawdown.shift(1, fill_value=0) > threshold) & (drawdown <= threshold)
    dip_indices = np.where(dips)[0]
    for idx in dip_indices:
        dip_date = drawdown.index[idx]
        dd_val = drawdown.iloc[idx]
        ret_6m = (
            (prices.iloc[idx + 6] / prices.iloc[idx]) - 1
            if idx + 6 < len(prices)
            else np.nan
        )
        ret_12m = (
            (prices.iloc[idx + 12] / prices.iloc[idx]) - 1
            if idx + 12 < len(prices)
            else np.nan
        )
        ret_24m = (
            (prices.iloc[idx + 24] / prices.iloc[idx]) - 1
            if idx + 24 < len(prices)
            else np.nan
        )
        ret_36m = (
            (prices.iloc[idx + 36] / prices.iloc[idx]) - 1
            if idx + 36 < len(prices)
            else np.nan
        )
        results.append(
            {
                "threshold": threshold,
                "dip_date": dip_date,
                "drawdown": dd_val,
                "6m_return": ret_6m,
                "12m_return": ret_12m,
                "24m_return": ret_24m,
                "36m_return": ret_36m,
            }
        )

# Print results
print("Return After Drawdown Dips for Canada Equities:")
for r in results:
    print(
        f"Threshold: {r['threshold']*100:.0f}% | Dip Date: {r['dip_date'].date()} | Drawdown: {r['drawdown']:.2%} | 6m Return: {r['6m_return']:.2%} | 12m Return: {r['12m_return']:.2%} | 24m Return: {r['24m_return']:.2%} | 36m Return: {r['36m_return']:.2%}"
    )

# Visualization and statistics
import matplotlib.pyplot as plt
import seaborn as sns

# Organize results by threshold
import pandas as pd

df_results = pd.DataFrame(results)

# Boxplot of 12m returns by threshold

# Boxplots for 6m, 12m, 24m returns by threshold
fig, axes = plt.subplots(1, 4, figsize=(24, 6))
for ax, period, label in zip(
    axes,
    ["6m_return", "12m_return", "24m_return", "36m_return"],
    ["6-Month", "12-Month", "24-Month", "36-Month"],
):
    sns.boxplot(
        x=df_results["threshold"] * 100, y=df_results[period], showmeans=True, ax=ax
    )
    ax.set_xlabel("Drawdown Threshold (%)")
    ax.set_ylabel(f"{label} Return After Dip")
    ax.set_title(f"Distribution of {label} Returns After Dips")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
plt.show()

# Print summary statistics for each threshold
print("\nSummary statistics for returns after drawdown dips:")
for threshold in thresholds:
    print(f"Threshold {threshold*100:.0f}%:")
    for period, label in zip(
        ["6m_return", "12m_return", "24m_return", "36m_return"],
        ["6m", "12m", "24m", "36m"],
    ):
        vals = df_results[df_results["threshold"] == threshold][period].dropna()
        if len(vals) > 0:
            print(
                f"  {label}: count={len(vals)}, mean={vals.mean():.2%}, median={vals.median():.2%}, std={vals.std():.2%}"
            )
        else:
            print(f"  {label}: No events found.")
