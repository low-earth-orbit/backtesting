import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configuration
LOOKBACK_YEARS = 40  # Number of years of historical data to use for optimization
ROLLING_WINDOW_YEARS = 12  # Length of rolling window for stability analysis
WINDOW_STEP_MONTHS = 12  # How often to calculate new weights (annual rebalancing)
RESAMPLE_ITERATIONS = 1000  # Number of bootstrap iterations for resampled efficiency


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


def bootstrap_returns(returns, size=None):
    """
    Generate bootstrapped returns by sampling with replacement
    """
    if size is None:
        size = len(returns)
    indices = np.random.randint(0, len(returns), size=size)
    return returns.iloc[indices]


def optimize_resampled_portfolio(returns, n_iterations=RESAMPLE_ITERATIONS):
    """
    Perform resampled efficiency optimization
    Returns both the resampled weights and the distribution of weights
    """
    n_assets = returns.shape[1]
    all_weights = np.zeros((n_iterations, n_assets))

    for i in range(n_iterations):
        # Generate bootstrapped sample
        boot_returns = bootstrap_returns(returns)

        # Calculate covariance matrix for this sample
        boot_cov = boot_returns.cov()

        # Optimize for this sample
        weights = optimize_min_variance(boot_cov)
        all_weights[i] = weights

    # Calculate mean weights across all iterations
    mean_weights = np.mean(all_weights, axis=0)

    # Calculate confidence intervals and statistics
    weight_stats = {
        "mean": mean_weights,
        "std": np.std(all_weights, axis=0),
        "percentile_5": np.percentile(all_weights, 5, axis=0),
        "percentile_95": np.percentile(all_weights, 95, axis=0),
    }

    return mean_weights, weight_stats


def analyze_rolling_weights(returns, window_years, step_months):
    """
    Perform rolling window analysis of minimum variance portfolio weights
    """
    window_size = window_years * 12  # Convert years to months
    weights_over_time = []
    dates = []

    # Calculate weights for each window
    for start_idx in range(0, len(returns) - window_size, step_months):
        end_idx = start_idx + window_size
        window_returns = returns.iloc[start_idx:end_idx]
        window_cov = window_returns.cov()
        weights = optimize_min_variance(window_cov)
        weights_over_time.append(weights)
        dates.append(returns.index[end_idx - 1])  # Use end of window date

    # Convert to DataFrame for easier analysis
    weights_df = pd.DataFrame(weights_over_time, columns=returns.columns, index=dates)

    return weights_df


def plot_resampled_weights(weight_stats, asset_names):
    """
    Plot the distribution of resampled portfolio weights with confidence intervals
    """
    plt.figure(figsize=(12, 6))
    x = np.arange(len(asset_names))
    width = 0.35

    # Plot mean weights as bars
    plt.bar(x, weight_stats["mean"], width, label="Mean Weight")

    # Add error bars for 90% confidence interval
    plt.errorbar(
        x,
        weight_stats["mean"],
        yerr=[
            weight_stats["mean"] - weight_stats["percentile_5"],
            weight_stats["percentile_95"] - weight_stats["mean"],
        ],
        fmt="none",
        color="black",
        capsize=5,
        label="90% Confidence Interval",
    )

    plt.xlabel("Assets")
    plt.ylabel("Portfolio Weight")
    plt.title("Resampled Portfolio Weights with 90% Confidence Intervals")
    plt.xticks(x, asset_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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
canada_path = "dataset/canada-cad-price.xls"
us_path = "dataset/usa-cad-price.xls"
eafe_path = "dataset/eafe-cad-price.xls"
em_path = "dataset/em-cad-price.xls"


# Read and clean price data
print("Loading and cleaning data...")
canada_prices = load_and_clean_data(canada_path)
eafe_prices = load_and_clean_data(eafe_path)
em_prices = load_and_clean_data(em_path)
us_prices = load_and_clean_data(us_path)

# Combine into a single DataFrame
prices = pd.concat(
    [
        canada_prices.rename("Canada"),
        eafe_prices.rename("EAFE"),
        em_prices.rename("EM"),
        us_prices.rename("US"),
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
print(returns.corr().round(3))  # Rounded to 3 decimal places for readability

# Calculate traditional minimum variance portfolio weights
print("\nCalculating traditional minimum variance portfolio...")
traditional_weights = optimize_min_variance(cov_matrix)

# Calculate resampled efficient portfolio
print(
    f"\nCalculating resampled efficient portfolio (using {RESAMPLE_ITERATIONS} iterations)..."
)
resampled_weights, weight_stats = optimize_resampled_portfolio(returns)

# Compare traditional vs resampled weights
print("\nPortfolio Weights Comparison:")
print("\nTraditional Minimum Variance Weights:")
for asset, weight in zip(returns.columns, traditional_weights):
    print(f"{asset}: {weight:.2%}")

print("\nResampled Efficient Weights:")
for asset, weight in zip(returns.columns, resampled_weights):
    print(f"{asset}: {weight:.2%}")

print("\nResampled Weight Statistics:")
for i, asset in enumerate(returns.columns):
    print(f"\n{asset}:")
    print(f"  Mean: {weight_stats['mean'][i]:.2%}")
    print(f"  Std Dev: {weight_stats['std'][i]:.2%}")
    print(
        f"  90% CI: [{weight_stats['percentile_5'][i]:.2%}, {weight_stats['percentile_95'][i]:.2%}]"
    )

# Plot resampled weight distribution
plot_resampled_weights(weight_stats, returns.columns)

# Calculate portfolio metrics using resampled weights
weights = resampled_weights  # Use resampled weights for subsequent analysis
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

# Perform rolling window analysis
print("\nPerforming rolling window analysis...")
rolling_weights = analyze_rolling_weights(
    returns, ROLLING_WINDOW_YEARS, WINDOW_STEP_MONTHS
)

# Calculate weight stability metrics
print(f"\nWeight Stability Analysis ({ROLLING_WINDOW_YEARS}-year rolling windows):")
print("\nWeight Statistics:")
print(rolling_weights.describe().round(3))

print("\nWeight Ranges (Min-Max):")
for asset in rolling_weights.columns:
    weight_range = rolling_weights[asset].max() - rolling_weights[asset].min()
    print(f"{asset}:")
    print(f"  Range: {weight_range:.2%}")
    print(f"  Standard Deviation: {rolling_weights[asset].std():.2%}")

# Additional analysis: Individual market statistics
print("\nIndividual Market Statistics (Annualized):")
for market in returns.columns:
    market_monthly_return = returns[market].mean()
    market_annual_return = np.exp(market_monthly_return * 12) - 1
    market_annual_vol = returns[market].std() * np.sqrt(12)
    market_sharpe = market_annual_return / market_annual_vol
    print(f"\n{market}:")
    print(f"  Annual Return: {market_annual_return:.2%}")
    print(f"  Annual Volatility: {market_annual_vol:.2%}")
    print(f"  Sharpe Ratio: {market_sharpe:.2f}")
