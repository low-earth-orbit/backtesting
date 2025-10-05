import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_clean_data(file_path):
    """
    Load Excel file and clean the data to ensure we have proper price series
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

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

        # Sort index to ensure chronological order
        df = df.sort_index()
        return df[price_col]
    except Exception as e:
        print(f"\nError loading {file_path}:")
        print(f"Detailed error: {str(e)}")
        raise


def calculate_portfolio_metrics(returns_df, weights):
    """Calculate portfolio returns and risk metrics"""
    # Input `returns_df` are log returns. Convert to arithmetic returns for
    # correct weighted portfolio aggregation: a = exp(l) - 1
    arith_returns = np.exp(returns_df) - 1

    if isinstance(weights, pd.DataFrame):
        # For dynamic weights, align dates then compute weighted arithmetic returns
        common_dates = arith_returns.index.intersection(weights.index)
        arith_returns = arith_returns.loc[common_dates]
        weights = weights.loc[common_dates]
        portfolio_arith = (arith_returns * weights[arith_returns.columns]).sum(axis=1)
    else:
        # For fixed weights, elementwise dot with arithmetic returns
        portfolio_arith = arith_returns.dot(weights)

    # Convert portfolio arithmetic returns to portfolio log returns for
    # annualization and cum_wealth: ln(1 + r_portfolio)
    portfolio_returns = np.log1p(portfolio_arith)

    monthly_return = portfolio_returns.mean()
    monthly_vol = portfolio_returns.std()
    annual_return = np.exp(monthly_return * 12) - 1
    annual_vol = monthly_vol * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    return {
        "monthly_returns": portfolio_returns,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        # cumulative wealth from portfolio log returns
        "cum_wealth": np.exp(portfolio_returns.cumsum()),
    }


def calculate_fixed_rebalanced_metrics(
    returns_df, fixed_weights_series, rebalance_months=12
):
    """Compute portfolio metrics for a fixed allocation that is only rebalanced every `rebalance_months`.

    fixed_weights_series: pd.Series with index as asset names and sum=1
    rebalance_months: int number of months between rebalances (12 = annual)
    """
    # Convert log returns to arithmetic
    arith = np.exp(returns_df) - 1

    # Prepare a float-typed DataFrame of applied weights per month
    n = len(returns_df)
    applied_weights = pd.DataFrame(
        0.0, index=returns_df.index, columns=returns_df.columns, dtype=float
    )

    # Assign target weights for each rebalance block
    for start in range(0, n, rebalance_months):
        end = min(start + rebalance_months, n)
        applied_weights.iloc[start:end] = fixed_weights_series.values

    # compute portfolio arithmetic returns (ensure float dtype)
    port_arith = (arith.astype(float) * applied_weights.astype(float)).sum(axis=1)
    port_log = np.log1p(port_arith)
    monthly_return = port_log.mean()
    monthly_vol = port_log.std()
    annual_return = np.exp(monthly_return * 12) - 1
    annual_vol = monthly_vol * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    return {
        "monthly_returns": port_log,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "cum_wealth": np.exp(port_log.cumsum()),
    }


def adjust_returns_for_equal_performance(returns_df):
    """
    Adjust EAFE and US returns to have equal long-term performance while
    preserving their correlation structure and volatility characteristics
    """
    # Calculate the average return for EAFE and US
    eafe_mean = returns_df["EAFE"].mean()
    us_mean = returns_df["US"].mean()
    target_mean = (eafe_mean + us_mean) / 2

    # Adjust the returns to have the same mean while preserving volatility
    adjusted_returns = returns_df.copy()
    adjusted_returns["EAFE"] = returns_df["EAFE"] - eafe_mean + target_mean
    adjusted_returns["US"] = returns_df["US"] - us_mean + target_mean

    return adjusted_returns


# File paths
canada_path = "dataset/canada-cad-net.xls"
eafe_path = "dataset/eafe-cad-net.xls"
us_path = "dataset/usa-cad-net.xls"

# Load and clean price data
print("Loading price data...")
canada_prices = load_and_clean_data(canada_path)
eafe_prices = load_and_clean_data(eafe_path)
us_prices = load_and_clean_data(us_path)

# Combine into a single DataFrame
prices = pd.concat(
    [
        canada_prices.rename("Canada"),
        eafe_prices.rename("EAFE"),
        us_prices.rename("US"),
    ],
    axis=1,
)

# Drop rows with missing values
prices = prices.dropna()

# Calculate log returns
returns = np.log(prices / prices.shift(1)).dropna()

# Adjust EAFE and US returns to have equal long-term performance
adjusted_returns = adjust_returns_for_equal_performance(returns)
# adjusted_returns = returns

# Print original vs adjusted annualized returns
print("\nAnnualized Returns (Original vs Adjusted):")
print("Original:")
for market in ["EAFE", "US"]:
    ann_ret = np.exp(returns[market].mean() * 12) - 1
    print(f"{market}: {ann_ret:.2%}")

print("\nAdjusted (Equal Long-term Returns):")
for market in ["EAFE", "US"]:
    ann_ret = np.exp(adjusted_returns[market].mean() * 12) - 1
    print(f"{market}: {ann_ret:.2%}")

# Strategy 1: Fixed weights
CANADA_WEIGHT = 0.30  # Fixed Canada allocation
# Split remaining 70% between EAFE and US (50/50 of the remainder)
FIXED_EAFE = (1 - CANADA_WEIGHT) * 0.5
FIXED_US = (1 - CANADA_WEIGHT) * 0.5

fixed_weights = pd.DataFrame(
    {
        "Canada": [CANADA_WEIGHT] * len(adjusted_returns),
        "EAFE": [FIXED_EAFE] * len(adjusted_returns),
        "US": [FIXED_US] * len(adjusted_returns),
    },
    index=adjusted_returns.index,
)


# Strategy 2: Market cap-weighted for EAFE/US portion
def calculate_market_cap_weights(prices_df):
    """Calculate dynamic weights based on relative market values"""
    # Normalize prices to start at 100 to track relative market cap changes
    norm_prices = prices_df / prices_df.iloc[0] * 100

    # Calculate EAFE and US weights maintaining their sum at (1 - CANADA_WEIGHT)
    total_ex_canada = norm_prices["EAFE"] + norm_prices["US"]
    eafe_weight = (1 - CANADA_WEIGHT) * (norm_prices["EAFE"] / total_ex_canada)
    us_weight = (1 - CANADA_WEIGHT) * (norm_prices["US"] / total_ex_canada)

    return pd.DataFrame(
        {
            "Canada": [CANADA_WEIGHT] * len(prices_df),
            "EAFE": eafe_weight,
            "US": us_weight,
        },
        index=prices_df.index,
    )


dynamic_weights = calculate_market_cap_weights(prices[["EAFE", "US"]])

# Calculate fixed weights strategy performance
fixed_metrics = calculate_portfolio_metrics(adjusted_returns, fixed_weights.iloc[0])

# Calculate dynamic weights strategy performance (vectorized via helper)
# Align dynamic weights so they are based on previous-period prices (no look-ahead)
initial_weights = pd.Series(
    {"Canada": CANADA_WEIGHT, "EAFE": FIXED_EAFE, "US": FIXED_US}
)
# Single source of truth for lag used both historically and in simulations
LAG_MONTHS = 1

# Compute dynamic weights from lagged prices (weights at t use prices at t-LAG_MONTHS)
# Build dynamic weights from lagged prices robustly: drop initial NaNs before normalization
shifted_prices = prices[["EAFE", "US"]].shift(LAG_MONTHS).dropna()
if len(shifted_prices) > 0:
    dyn_from_shifted = calculate_market_cap_weights(shifted_prices)
    # reindex to full timeline, head will be NaN for the first LAG_MONTHS rows
    dynamic_weights = dyn_from_shifted.reindex(prices.index)
else:
    # fallback: if all NaNs (very short series), set dynamic_weights to initial weights
    dynamic_weights = pd.DataFrame(
        [initial_weights.values] * len(prices),
        index=prices.index,
        columns=initial_weights.index,
    )
# fill the first `LAG_MONTHS` rows with the initial_weights to avoid NaNs
if LAG_MONTHS > 0:
    dynamic_weights.iloc[:LAG_MONTHS] = initial_weights
dynamic_metrics = calculate_portfolio_metrics(adjusted_returns, dynamic_weights)
dynamic_portfolio_returns = dynamic_metrics["monthly_returns"]

# Calculate cumulative wealth paths and exclude the first month from final-wealth/drawdown
# Use cum_wealth from metric dict (already uses log-return compounding)
cum_returns_fixed = fixed_metrics["cum_wealth"].iloc[1:]
cum_returns_dynamic = dynamic_metrics["cum_wealth"].iloc[1:]


# Bootstrap simulation function
def simulate_portfolio_returns(
    returns_data,
    weights_fixed,
    weights_dynamic,
    num_simulations=1000,
    block_size=12,
    seed=None,
    lag_months=None,
):
    """
    Perform block bootstrap simulation of portfolio returns.
    Uses block bootstrapping to preserve autocorrelation structure.
    """
    # Initialize results storage
    fixed_metrics = []
    dynamic_metrics = []

    n_periods = len(returns_data)
    original_index = returns_data.index.copy()

    # Guard block size
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size > n_periods:
        block_size = n_periods

    # Seed RNG for reproducibility when requested
    rng = np.random.default_rng(seed)

    for _ in range(num_simulations):
        # Generate random blocks
        n_blocks = n_periods // block_size + 1
        # Use rng to draw block starts
        block_starts = rng.integers(
            0, len(returns_data) - block_size + 1, size=n_blocks
        )

        # Create simulated returns by concatenating blocks
        sim_blocks = []
        for start in block_starts:
            block = returns_data.iloc[start : start + block_size].copy()
            sim_blocks.append(block)

        sim_returns = pd.concat(sim_blocks, axis=0)

        # Trim to original length and realign index
        sim_returns = sim_returns.iloc[:n_periods]
        sim_returns.index = original_index

        # Recompute dynamic weights from simulated returns (convert back to prices)
        if isinstance(weights_dynamic, pd.DataFrame):
            # Use neutral starting prices so simulated weights follow neutral scaling
            start_prices = pd.Series({"EAFE": 100.0, "US": 100.0})

            sim_prices = pd.DataFrame(index=sim_returns.index)
            for col in ["EAFE", "US"]:
                sim_prices[col] = start_prices[col] * np.exp(sim_returns[col].cumsum())

            # determine lag to use (default to global LAG_MONTHS)
            if lag_months is None:
                use_lag = LAG_MONTHS
            else:
                use_lag = int(lag_months)

            # compute weights from prices available `use_lag` months earlier
            shifted_sim_prices = sim_prices.shift(use_lag).dropna()
            if len(shifted_sim_prices) > 0:
                recomputed_dynamic_weights = calculate_market_cap_weights(
                    shifted_sim_prices
                )
                recomputed_dynamic_weights = recomputed_dynamic_weights.reindex(
                    sim_returns.index
                )
            else:
                recomputed_dynamic_weights = pd.DataFrame(
                    [initial_weights.values] * len(sim_returns),
                    index=sim_returns.index,
                    columns=initial_weights.index,
                )
            # fill the first `use_lag` rows with initial_weights to avoid NaNs
            if use_lag > 0:
                recomputed_dynamic_weights.iloc[:use_lag] = initial_weights.values
            aligned_dynamic_weights = recomputed_dynamic_weights.reindex(
                index=sim_returns.index
            )
        else:
            aligned_dynamic_weights = weights_dynamic

        # Calculate metrics for both strategies using vectorized metric function
        # For fixed weights we may want to enforce periodic rebalancing (e.g., annual)
        if isinstance(weights_fixed, pd.Series):
            fixed_portfolio_metrics = calculate_fixed_rebalanced_metrics(
                sim_returns, weights_fixed, rebalance_months=12
            )
        else:
            fixed_portfolio_metrics = calculate_portfolio_metrics(
                sim_returns, weights_fixed
            )
        dynamic_portfolio_metrics = calculate_portfolio_metrics(
            sim_returns, aligned_dynamic_weights
        )

        # Store results
        fixed_metrics.append(fixed_portfolio_metrics)
        dynamic_metrics.append(dynamic_portfolio_metrics)

    return pd.DataFrame(
        {
            "fixed_returns": [m["annual_return"] for m in fixed_metrics],
            "fixed_sharpe": [m["sharpe"] for m in fixed_metrics],
            "dynamic_returns": [m["annual_return"] for m in dynamic_metrics],
            "dynamic_sharpe": [m["sharpe"] for m in dynamic_metrics],
            "outperformance": [
                d["annual_return"] - f["annual_return"]
                for d, f in zip(dynamic_metrics, fixed_metrics)
            ],
        }
    )


# Print performance comparison
print("\nStrategy Comparison (historical return series):")
print("\n1. Fixed Equal Weight Strategy:")
print(f"Annual Return: {fixed_metrics['annual_return']:.2%}")
print(f"Annual Volatility: {fixed_metrics['annual_vol']:.2%}")
print(f"Sharpe Ratio: {fixed_metrics['sharpe']:.2f}")
print(f"Final Wealth (starting at 1): {cum_returns_fixed.iloc[-1]:.2f}")

print("\n2. Market Cap Weight Strategy:")
print(f"Annual Return: {dynamic_metrics['annual_return']:.2%}")
print(f"Annual Volatility: {dynamic_metrics['annual_vol']:.2%}")
print(f"Sharpe Ratio: {dynamic_metrics['sharpe']:.2f}")
print(f"Final Wealth (starting at 1): {cum_returns_dynamic.iloc[-1]:.2f}")

# Run bootstrap simulation
print("\nRunning Bootstrap Simulation...")
sim_results = simulate_portfolio_returns(
    adjusted_returns,
    fixed_weights.iloc[0],
    dynamic_weights,
    num_simulations=1000,
    block_size=12,
    seed=42,
)

# Calculate probability of dynamic strategy outperformance
prob_outperform = (sim_results["outperformance"] > 0).mean()
median_outperform = sim_results["outperformance"].median()

print("\nBootstrap Simulation Results (1000 simulations):")
print(f"Probability of Market Cap Strategy Outperformance: {prob_outperform:.1%}")
print(f"Median Outperformance: {median_outperform:.2%}")

print("\nMarket Cap Weight Strategy Statistics:")
print(f"Average Annual Return: {sim_results['dynamic_returns'].mean():.2%}")
print(f"Return 5th Percentile: {np.percentile(sim_results['dynamic_returns'], 5):.2%}")
print(
    f"Return 95th Percentile: {np.percentile(sim_results['dynamic_returns'], 95):.2%}"
)
print(f"Average Sharpe Ratio: {sim_results['dynamic_sharpe'].mean():.2f}")

print("\nFixed Weight Strategy Statistics:")
print(f"Average Annual Return: {sim_results['fixed_returns'].mean():.2%}")
print(f"Return 5th Percentile: {np.percentile(sim_results['fixed_returns'], 5):.2%}")
print(f"Return 95th Percentile: {np.percentile(sim_results['fixed_returns'], 95):.2%}")
print(f"Average Sharpe Ratio: {sim_results['fixed_sharpe'].mean():.2f}")

# Calculate and print tracking error between strategies
tracking_error = (
    fixed_metrics["monthly_returns"] - dynamic_metrics["monthly_returns"]
).std() * np.sqrt(12)
print(f"\nTracking Error between strategies: {tracking_error:.2%}")

# Additional risk metrics
max_drawdown_fixed = (1 - cum_returns_fixed / cum_returns_fixed.expanding().max()).max()
max_drawdown_dynamic = (
    1 - cum_returns_dynamic / cum_returns_dynamic.expanding().max()
).max()

print("\nRisk Metrics:")
print(f"Maximum Drawdown (Fixed): {max_drawdown_fixed:.2%}")
print(f"Maximum Drawdown (Dynamic): {max_drawdown_dynamic:.2%}")

# Visualize the distribution of simulation outcomes
plt.figure(figsize=(12, 6))
plt.hist(sim_results["outperformance"], bins=50, alpha=0.5, color="blue")
plt.axvline(x=0, color="red", linestyle="--", label="Zero Outperformance")
plt.axvline(
    x=median_outperform, color="green", linestyle="-", label="Median Outperformance"
)
plt.title(
    "Distribution of Market Cap Strategy Outperformance\n(Market Cap - Fixed Returns)"
)
plt.xlabel("Annual Outperformance")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


def _compute_lag_diagnostics(prices, adjusted_returns, dynamic_weights, LAG_MONTHS):
    """Compute and print diagnostics that explain the lag effect.

    Shows annualized returns for portfolios that use weights lagged by 0, 1, and 12 months
    (applying the same fill rule) and prints the covariance between weight changes
    and same-period returns to reveal whether the weight update is correlated with
    contemporaneous returns (i.e. a source of look-ahead or momentum capture).
    """
    arith = np.exp(adjusted_returns) - 1
    initial_weights = pd.Series(
        {"Canada": CANADA_WEIGHT, "EAFE": FIXED_EAFE, "US": FIXED_US}
    )

    def portfolio_annual_return_for_lag(lag):
        w = dynamic_weights.shift(lag)
        if lag > 0:
            w.iloc[:lag] = initial_weights
        # Align
        common = arith.index.intersection(w.index)
        port_arith = (arith.loc[common] * w.loc[common][arith.columns]).sum(axis=1)
        port_log = np.log1p(port_arith)
        ann = np.exp(port_log.mean() * 12) - 1
        return ann, port_log, w

    lags = [0, 1, 12]
    results = {}
    print("\nLag diagnostics (annualized returns):")
    for lag in lags:
        ann, port_log, w = portfolio_annual_return_for_lag(lag)
        results[lag] = (ann, port_log, w)
        print(f"lag={lag}: annual return = {ann:.4%}")

    # Print differences
    print("\nDifferences (no-lag minus lagged):")
    for lag in [1, 12]:
        diff = results[0][0] - results[lag][0]
        print(f"no-lag - lag={lag}: {diff:.4%}")

    # Compute simple diagnostic: correlation between weight changes (w_t - w_{t-1})
    # and same-period returns r_t for EAFE and US when using no-lag weights.
    w0 = results[0][2]
    w1 = results[1][2]
    # weight change from t-1 to t for no-lag (this shows responsiveness)
    dw = w0[["EAFE", "US"]].diff()
    # compute covariances between dw and same-period arithmetic returns
    covs = {}
    for col in ["EAFE", "US"]:
        cov = np.cov(dw[col].dropna(), arith.loc[dw[col].dropna().index, col].values)[
            0, 1
        ]
        covs[col] = cov
        print(f"Cov(weight_change_{col}, return_{col}) = {cov:.6f}")

    print(
        "\nInterpretation: positive covariance indicates the weighting rule tends to "
        "increase exposure to assets in months when they also have positive returns (i.e. "
        "it captures contemporaneous momentum / look-ahead). That explains why lag=0 "
        "can look better than lag>=1."
    )


# Run lag diagnostics to help explain the flipping result
_compute_lag_diagnostics(prices, adjusted_returns, dynamic_weights, LAG_MONTHS)
