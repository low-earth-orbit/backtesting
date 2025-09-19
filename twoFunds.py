import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Configuration
LOOKBACK_YEARS = 20  # Number of years of historical data to use for optimization


def calculate_portfolio_vol(weights, cov_matrix):
    """
    Calculate portfolio volatility (standard deviation)
    """
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))


def optimize_min_variance(cov_matrix):
    """
    Optimize for minimum variance portfolio
    """
    n = cov_matrix.shape[0]

    # Initial guess (equal weights)
    init_guess = np.array([1 / n] * n)

    # Constraints
    bounds = tuple((0, 1) for _ in range(n))  # Each weight between 0 and 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}  # Weights sum to 1

    # Minimize volatility
    result = minimize(
        calculate_portfolio_vol,
        init_guess,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def load_and_clean_data(file_path):
    """
    Load Excel file and clean the data to ensure we have proper price series
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)
        print(f"\nExamining structure of {file_path}:")
        print("Initial rows:")
        print(df.head(10))

        # Find the row with 'Date' header
        date_row_idx = None
        for idx, row in df.iterrows():
            if isinstance(row["Unnamed: 0"], str) and "Date" in row["Unnamed: 0"]:
                date_row_idx = idx
                break

        if date_row_idx is None:
            raise ValueError("Could not find 'Date' header in the file")

        # Extract data starting from the header row
        df = df.iloc[date_row_idx:].reset_index(drop=True)

        # Set the first row as headers
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)

        # Find where the data ends (usually when we hit copyright or empty rows)
        data_end_idx = None
        for idx, row in df.iterrows():
            # Check if the date column contains copyright text or is empty
            date_val = str(row["Date"]).lower()
            if "copyright" in date_val or date_val == "nan" or date_val == "":
                data_end_idx = idx
                break

        if data_end_idx is not None:
            df = df.iloc[:data_end_idx]

        # Convert dates to datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%b %d, %Y")

        # Set date as index
        df = df.set_index("Date")

        # Get the price column (second column)
        price_col = df.columns[0]

        # Convert prices to numeric, removing any non-numeric characters
        df[price_col] = pd.to_numeric(
            df[price_col].astype(str).str.replace(r"[^\d.-]", "", regex=True),
            errors="coerce",
        )

        # Remove any rows with invalid prices
        df = df.dropna()

        print(f"\nSuccessfully loaded {len(df)} rows of clean data")
        print("Sample of cleaned data:")
        print(df.head())

        # Sort index to ensure chronological order
        df = df.sort_index()
        return df[price_col]
    except Exception as e:
        print(f"\nError loading {file_path}:")
        print(f"Detailed error: {str(e)}")
        print("\nPlease check if:")
        print("1. The file exists and is readable")
        print("2. The file contains both date and price columns")
        print("3. There are valid numeric prices in the data")
        raise


# File paths
canada_path = "dataset/canada-price-cad.xls"
acwi_path = "dataset/acwi-price-cad.xls"

# Read and clean price data
print("Loading and cleaning data...")
canada_prices = load_and_clean_data(canada_path)
acwi_prices = load_and_clean_data(acwi_path)

# Combine into a single DataFrame
prices = pd.concat(
    [
        canada_prices.rename("Canada"),
        acwi_prices.rename("ACWI"),
    ],
    axis=1,
)

# Drop rows with missing values
prices = prices.dropna()

# Filter to use only the last LOOKBACK_YEARS years of data
end_date = prices.index.max()
start_date = end_date - pd.DateOffset(years=LOOKBACK_YEARS)
prices = prices[prices.index >= start_date]

print(f"\nData Summary (using last {LOOKBACK_YEARS} years of data):")
print("Date Range:", prices.index.min(), "to", prices.index.max())
print("\nNumber of observations:", len(prices))
print("\nPrice Statistics:")
print(prices.describe())

# Calculate log returns instead of simple returns for better statistical properties
returns = np.log(prices / prices.shift(1))

# Print detailed returns information for debugging
print("\nMonthly Returns Statistics (in %):")
monthly_returns_pct = returns * 100
print(monthly_returns_pct.describe())

# Print the number of extreme returns (more than 3 standard deviations)
std_dev = returns.std()
extreme_returns = (returns.abs() > 3 * std_dev).sum()
print("\nNumber of extreme monthly returns (>3 std dev):")
print(extreme_returns)

# Calculate covariance matrix
print("\nCalculating portfolio optimization parameters...")
cov_matrix = returns.cov()  # Monthly covariance matrix

# Print the correlation matrix
print("\nCorrelation Matrix:")
print(returns.corr())

# Calculate minimum variance portfolio weights
weights = optimize_min_variance(cov_matrix)

# Calculate portfolio metrics
portfolio_monthly_vol = calculate_portfolio_vol(weights, cov_matrix)
annualized_vol = portfolio_monthly_vol * np.sqrt(12)  # Annualize monthly volatility

print("\nMinimum Variance Portfolio Allocation:")
for asset, weight in zip(returns.columns, weights):
    print(f"{asset}: {weight:.2%}")

print(f"\nPortfolio Metrics:")
print(f"Monthly Volatility: {portfolio_monthly_vol:.2%}")
print(f"Annual Volatility: {annualized_vol:.2%}")

# Calculate portfolio returns
portfolio_returns = returns.dot(weights)
monthly_return = portfolio_returns.mean()
annual_return = np.exp(monthly_return * 12) - 1  # Properly annualize log returns
print(f"Expected Annual Return: {annual_return:.2%}")
print(f"Sharpe Ratio (assuming 0% risk-free rate): {annual_return/annualized_vol:.2f}")


# Plot efficient frontier and allocation analysis
def get_portfolio_metrics(weights):
    portfolio_ret = np.sum(weights * returns.mean() * 12)  # Annualized return
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 12, weights)))
    return portfolio_vol, portfolio_ret


# Generate random portfolios
n_portfolios = 1000
all_weights = []
ret_arr = []
vol_arr = []
sharpe_arr = []

for i in range(n_portfolios):
    weights = np.random.random(2)
    weights = weights / np.sum(weights)
    all_weights.append(weights)

    vol, ret = get_portfolio_metrics(weights)
    ret_arr.append(ret)
    vol_arr.append(vol)
    sharpe_arr.append(ret / vol)  # Assuming 0% risk-free rate

# Convert to numpy arrays
returns_arr = np.array(ret_arr)
volatility_arr = np.array(vol_arr)
sharpe_arr = np.array(sharpe_arr)

# Calculate metrics for the true minimum variance portfolio weights
min_var_weights = optimize_min_variance(
    returns.cov() * 12
)  # Using annualized covariance
min_var_vol, min_var_ret = get_portfolio_metrics(min_var_weights)

# Get metrics for individual assets
asset_rets = returns.mean() * 12
asset_vols = returns.std() * np.sqrt(12)

# Create subplots
fig, (ax2) = plt.subplots(1, 1, figsize=(9, 6))

# Plot 2: Canada Allocation vs Volatility
canada_weights = [w[0] for w in all_weights]  # First weight is for Canada
scatter2 = ax2.scatter(
    volatility_arr,
    canada_weights,
    c=returns_arr,
    cmap="viridis",
    marker="o",
    s=10,
    alpha=0.3,
)

# Plot minimum variance portfolio point
ax2.scatter(
    min_var_vol,
    min_var_weights[0],  # Canada weight in minimum variance portfolio
    color="red",
    marker="*",
    s=200,
    label="Minimum Variance Portfolio",
)

ax2.set_xlabel("Historical Annual Price Volatility")
ax2.set_ylabel("Allocation to Canada")
ax2.set_title("Canada/ACWI Allocation vs Portfolio Volatility")
ax2.legend()
ax2.grid(True, alpha=0.2)
ax2.set_ylim([0, 1])

# Adjust layout and display
plt.tight_layout()
plt.show()

# Print optimal allocation analysis
print("\nOptimal Allocation Analysis:")
print("\nMinimum Variance Portfolio Composition:")
print(f"Canada Weight: {min_var_weights[0]:.2%}")
print(f"ACWI Weight: {min_var_weights[1]:.2%}")
print(f"Total Weight: {sum(min_var_weights):.2%}")
print(f"\nRisk-Return Metrics:")
print(f"Portfolio Volatility (Annual): {min_var_vol:.2%}")
print(f"Expected Return (Annual): {min_var_ret:.2%}")
print(f"Sharpe Ratio: {min_var_ret/min_var_vol:.2f}")
