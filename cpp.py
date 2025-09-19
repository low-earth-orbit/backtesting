# Chinese Couch Potato

import numpy as np
from scipy.optimize import minimize

# Asset names


# Asset names (stocks only)
assets = ["A Shares", "HK-listed Stocks", "US Stocks", "Europe Stocks"]

# Nominal returns (before fees)
nominal_returns = np.array([0.08, 0.09, 0.06, 0.075])
# Fees
fees = np.array([0.005, 0.012, 0.018, 0.018])
# Net returns
net_returns = nominal_returns - fees

# Volatilities (annualized, as decimals)
vols = np.array([0.2176, 0.1930, 0.1446, 0.16])

# Correlation matrix (4x4)
corr = np.array(
    [
        [1.00, 0.68, 0.44, 0.40],  # A Shares
        [0.68, 1.00, 0.49, 0.50],  # HK Stocks
        [0.44, 0.49, 1.00, 0.70],  # US Stocks
        [0.40, 0.50, 0.70, 1.00],  # Europe Stocks
    ]
)

# Covariance matrix
cov = np.outer(vols, vols) * corr


# Min variance optimization


def portfolio_vol(w, cov):
    return np.sqrt(np.dot(w.T, np.dot(cov, w)))


n = len(assets)
init_guess = np.array([1 / n] * n)
bounds = tuple((0, 1) for _ in range(n))
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

result = minimize(
    portfolio_vol, init_guess, args=(cov,), bounds=bounds, constraints=constraints
)
weights = result.x


# --- Common Allocation Strategies ---


def print_allocation(title, weights, net_returns, cov):
    port_vol = portfolio_vol(weights, cov)
    port_return = np.dot(weights, net_returns)
    print(f"\n{title}")
    for asset, weight in zip(assets, weights):
        print(f"{asset}: {weight:.2%}")
    print(f"Portfolio Volatility: {port_vol:.2%}")
    print(f"Expected Net Return: {port_return:.2%}")


# 1. Minimum Variance
print_allocation(
    "Minimum Variance Portfolio Allocation (Net of Fees):", weights, net_returns, cov
)

# 2. Equal Weight
equal_weights = np.array([1 / n] * n)
print_allocation("Equal Weight Portfolio:", equal_weights, net_returns, cov)


# 3. Risk Parity
def risk_contribution(weights, cov):
    port_vol = portfolio_vol(weights, cov)
    mrc = np.dot(cov, weights)
    rc = weights * mrc / port_vol
    return rc


def risk_parity_objective(weights, cov):
    rc = risk_contribution(weights, cov)
    return np.sum((rc - np.mean(rc)) ** 2)


rp_result = minimize(
    risk_parity_objective,
    init_guess,
    args=(cov,),
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)
risk_parity_weights = rp_result.x
print_allocation("Risk Parity Portfolio:", risk_parity_weights, net_returns, cov)


# 4. Max Sharpe Ratio
def neg_sharpe(weights, net_returns, cov):
    port_vol = portfolio_vol(weights, cov)
    port_return = np.dot(weights, net_returns)
    return -port_return / port_vol


sharpe_result = minimize(
    neg_sharpe,
    init_guess,
    args=(net_returns, cov),
    bounds=bounds,
    constraints=constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)
sharpe_weights = sharpe_result.x
print_allocation("Max Sharpe Ratio Portfolio:", sharpe_weights, net_returns, cov)
