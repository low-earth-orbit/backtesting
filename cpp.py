# Chinese Couch Potato

import numpy as np
from scipy.optimize import minimize

# Asset names


# --- Asset Classes ---
# Stocks only (original logic remains unchanged)
stock_assets = ["A Shares", "HK-listed Stocks", "US Stocks", "Europe Stocks"]
stock_nominal_returns = np.array([0.08, 0.085, 0.07, 0.08])
stock_fees = np.array([0.005, 0.012, 0.012, 0.02])
stock_net_returns = stock_nominal_returns - stock_fees
stock_vols = np.array([0.2176, 0.1930, 0.1446, 0.16])
stock_corr = np.array(
    [
        [1.00, 0.68, 0.44, 0.40],
        [0.68, 1.00, 0.49, 0.50],
        [0.44, 0.49, 1.00, 0.70],
        [0.40, 0.50, 0.70, 1.00],
    ]
)
stock_cov = np.outer(stock_vols, stock_vols) * stock_corr

# --- Add China Bond ---
assets = ["A Shares", "HK-listed Stocks", "US Stocks", "Europe Stocks", "China Bond"]

nominal_returns = np.array([0.08, 0.085, 0.07, 0.08, 0.03])
fees = np.array([0.005, 0.012, 0.012, 0.02, 0.005])
net_returns = nominal_returns - fees
vols = np.array([0.2176, 0.1930, 0.1446, 0.16, 0.02])

# Correlation matrix (5x5)
corr = np.array(
    [
        [1.00, 0.68, 0.44, 0.40, 0.30],  # A Shares
        [0.68, 1.00, 0.49, 0.50, 0.10],  # HK Stocks
        [0.44, 0.49, 1.00, 0.70, 0.10],  # US Stocks
        [0.40, 0.50, 0.70, 1.00, 0.10],  # Europe Stocks
        [0.30, 0.10, 0.10, 0.10, 1.00],  # China Bond
    ]
)
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


# --- Common Allocation Strategies (stocks only) ---
def print_allocation(title, weights, net_returns, cov, assets):
    port_vol = portfolio_vol(weights, cov)
    port_return = np.dot(weights, net_returns)
    print(f"\n{title}")
    for asset, weight in zip(assets, weights):
        print(f"{asset}: {weight:.2%}")
    print(f"Portfolio Volatility: {port_vol:.2%}")
    print(f"Expected Net Return: {port_return:.2%}")


# 1. Minimum Variance (stocks only)
stock_n = len(stock_assets)
stock_init_guess = np.array([1 / stock_n] * stock_n)
stock_bounds = tuple((0, 1) for _ in range(stock_n))
stock_constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
stock_result = minimize(
    portfolio_vol,
    stock_init_guess,
    args=(stock_cov,),
    bounds=stock_bounds,
    constraints=stock_constraints,
)
stock_weights = stock_result.x
print_allocation(
    "Minimum Variance Portfolio Allocation (Stocks Only):",
    stock_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)

# 2. Equal Weight (stocks only)
stock_equal_weights = np.array([1 / stock_n] * stock_n)
print_allocation(
    "Equal Weight Portfolio (Stocks Only):",
    stock_equal_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)

# 3. Risk Parity (stocks only)
def risk_contribution(weights, cov):
    port_vol = portfolio_vol(weights, cov)
    mrc = np.dot(cov, weights)
    rc = weights * mrc / port_vol
    return rc

def risk_parity_objective(weights, cov):
    rc = risk_contribution(weights, cov)
    return np.sum((rc - np.mean(rc)) ** 2)

stock_rp_result = minimize(
    risk_parity_objective,
    stock_init_guess,
    args=(stock_cov,),
    bounds=stock_bounds,
    constraints=stock_constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)
stock_risk_parity_weights = stock_rp_result.x
print_allocation(
    "Risk Parity Portfolio (Stocks Only):",
    stock_risk_parity_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)

# 4. Max Sharpe Ratio (stocks only)
def neg_sharpe(weights, net_returns, cov):
    port_vol = portfolio_vol(weights, cov)
    port_return = np.dot(weights, net_returns)
    return -port_return / port_vol

stock_sharpe_result = minimize(
    neg_sharpe,
    stock_init_guess,
    args=(stock_net_returns, stock_cov),
    bounds=stock_bounds,
    constraints=stock_constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)
stock_sharpe_weights = stock_sharpe_result.x
print_allocation(
    "Max Sharpe Ratio Portfolio (Stocks Only):",
    stock_sharpe_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)

# --- Efficient Frontier with China Bond ---
import matplotlib.pyplot as plt


def efficient_frontier(cov, net_returns, bounds, constraints, assets, n_points=50):
    # Volatility targets
    min_var_result = minimize(
        portfolio_vol, init_guess, args=(cov,), bounds=bounds, constraints=constraints
    )
    min_vol = portfolio_vol(min_var_result.x, cov)
    max_vol = portfolio_vol(np.eye(len(assets))[np.argmax(net_returns)], cov)
    vol_targets = np.linspace(min_vol, max_vol, n_points)

    allocations = []
    achieved_vols = []
    achieved_returns = []
    for target_vol in vol_targets:

        def constraint_vol(w):
            return portfolio_vol(w, cov) - target_vol

        cons = [constraints, {"type": "eq", "fun": constraint_vol}]
        res = minimize(
            lambda w: -np.dot(w, net_returns),
            init_guess,
            bounds=bounds,
            constraints=cons,
        )
        if res.success:
            allocations.append(res.x)
            achieved_vols.append(portfolio_vol(res.x, cov))
            achieved_returns.append(np.dot(res.x, net_returns))
        else:
            allocations.append(np.full(len(assets), np.nan))
            achieved_vols.append(np.nan)
            achieved_returns.append(np.nan)
    allocations = np.array(allocations)
    achieved_vols = np.array(achieved_vols)
    achieved_returns = np.array(achieved_returns)
    return allocations, achieved_vols, achieved_returns


# Run efficient frontier
allocations, achieved_vols, achieved_returns = efficient_frontier(
    cov, net_returns, bounds, constraints, assets
)

# Stack plot: allocations vs volatility
plt.figure(figsize=(12, 6))
plt.stackplot(achieved_vols, allocations.T, labels=assets)
plt.xlabel("Portfolio Volatility")
plt.ylabel("Allocation")
plt.title("Efficient Frontier: Allocations vs Portfolio Volatility")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Stack plot: allocations vs expected net return
plt.figure(figsize=(12, 6))
plt.stackplot(achieved_returns, allocations.T, labels=assets)
plt.xlabel("Expected Net Return")
plt.ylabel("Allocation")
plt.title("Efficient Frontier: Allocations vs Expected Net Return")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
