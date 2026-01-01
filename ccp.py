# Chinese Couch Potato

import numpy as np

# Optional import: SciPy's optimizer may not be available in every environment.
try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

# --- Asset Classes ---
stock_assets = ["A Shares", "HK-listed Stocks", "US Stocks", "Europe Stocks"]
stock_nominal_returns = np.array([0.055, 0.057, 0.045, 0.049])
stock_fees = np.array([0.0029, 0.0135, 0.0101, 0.0197])
stock_net_returns = stock_nominal_returns - stock_fees
stock_vols = np.array([0.2727, 0.2293, 0.1594, 0.1822])
stock_corr = np.array(
    [
        [1.00, 0.68, 0.27, 0.32],
        [0.68, 1.00, 0.41, 0.56],
        [0.27, 0.41, 1.00, 0.85],
        [0.32, 0.56, 0.85, 1.00],
    ]
)
stock_cov = np.outer(stock_vols, stock_vols) * stock_corr

# --- Bond ---
# Bond: nominal return 2.70%, volatility 2.75%, fees 0.30%, zero correlation with stocks
bond_name = "China Bond"
bond_nominal = 0.027
bond_fee = 0.002
bond_net = bond_nominal - bond_fee
bond_vol = 0.0275

# --- Gold ---
# Gold: nominal return 3.40%, fees 0.20%, volatility 15.88%, zero correlation with stocks/bond
gold_name = "Gold"
gold_nominal = 0.034
gold_fee = 0.0023
gold_net = gold_nominal - gold_fee
gold_vol = 0.1588

# --- Use stocks only ---
# We only need the 4 stock asset classes. Use the stock returns/covariance directly.
assets = stock_assets
net_returns = stock_net_returns
cov = stock_cov


# Min variance optimization
EPS = 1e-12
RISK_FREE = 0.019  # risk-free rate (1.9%)


def portfolio_vol(w, cov):
    """Portfolio volatility (sqrt of quadratic form). Guard against tiny negative round-off."""
    w = np.asarray(w)
    val = np.dot(w.T, np.dot(cov, w))
    return np.sqrt(max(val, 0.0))


n = len(assets)
init_guess = np.array([1 / n] * n)
bounds = tuple((0, 1) for _ in range(n))
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

if minimize is None:
    raise ImportError(
        "scipy.optimize.minimize is required to run this script. Install SciPy (e.g. `pip install scipy`)."
    )

result = minimize(
    portfolio_vol, init_guess, args=(cov,), bounds=bounds, constraints=constraints
)
if not getattr(result, "success", False):
    raise RuntimeError(
        f"Global optimization failed: {getattr(result, 'message', None)}"
    )
weights = result.x


# --- Common Allocation Strategies  ---
def print_allocation(title, weights, net_returns, cov, assets):
    port_vol = portfolio_vol(weights, cov)
    port_return = np.dot(weights, net_returns)
    print(f"\n{title}")
    for asset, weight in zip(assets, weights):
        print(f"{asset}: {weight:.2%}")
    print(f"Portfolio Volatility: {port_vol:.2%}")
    print(f"Expected Net Return: {port_return:.2%}")
    # Print Sharpe ratio using the assumed risk-free rate
    try:
        if port_vol > EPS:
            sharpe = (port_return - RISK_FREE) / port_vol
            print(f"Sharpe Ratio (rf={RISK_FREE:.2%}): {sharpe:.3f}")
        else:
            print("Sharpe Ratio: N/A (zero volatility)")
    except Exception:
        pass


# 1. Minimum Variance
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
    "Minimum Variance Portfolio Allocation:",
    stock_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)

# 2. Equal Weight
stock_equal_weights = np.array([1 / stock_n] * stock_n)
print_allocation(
    "Equal Weight Portfolio:",
    stock_equal_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)


# 3. Risk Parity
def risk_contribution(weights, cov):
    port_vol = portfolio_vol(weights, cov)
    if port_vol <= EPS:
        return np.zeros_like(weights)
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
if not getattr(stock_rp_result, "success", False):
    raise RuntimeError(
        f"Risk-parity optimization failed: {getattr(stock_rp_result, 'message', None)}"
    )
stock_risk_parity_weights = stock_rp_result.x
print_allocation(
    "Risk Parity Portfolio:",
    stock_risk_parity_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)


# 4. Max Sharpe Ratio
def neg_sharpe(weights, net_returns, cov):
    port_vol = portfolio_vol(weights, cov)
    if port_vol <= EPS:
        return 1e6
    port_return = np.dot(weights, net_returns)
    # Use excess return over the risk-free rate
    return -(port_return - RISK_FREE) / port_vol


stock_sharpe_result = minimize(
    neg_sharpe,
    stock_init_guess,
    args=(stock_net_returns, stock_cov),
    bounds=stock_bounds,
    constraints=stock_constraints,
    options={"ftol": 1e-12, "maxiter": 1000},
)
if not getattr(stock_sharpe_result, "success", False):
    raise RuntimeError(
        f"Sharpe optimization failed: {getattr(stock_sharpe_result, 'message', None)}"
    )
stock_sharpe_weights = stock_sharpe_result.x
print_allocation(
    "Max Sharpe Ratio Portfolio:",
    stock_sharpe_weights,
    stock_net_returns,
    stock_cov,
    stock_assets,
)

# --- Efficient Frontier with China Bond ---
import matplotlib.pyplot as plt


def efficient_frontier(cov, net_returns, bounds, constraints, assets, n_points=50):
    # Volatility targets
    # local initial guess for this asset universe
    local_init = np.array([1.0 / len(assets)] * len(assets))

    min_var_result = minimize(
        portfolio_vol, local_init, args=(cov,), bounds=bounds, constraints=constraints
    )
    if not getattr(min_var_result, "success", False):
        raise RuntimeError(
            f"Min-variance optimization failed: {getattr(min_var_result, 'message', None)}"
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
            local_init,
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
# Compute frontiers for stocks-only and stocks+bond
# Stocks-only
stock_assets_list = stock_assets
stock_net = stock_net_returns
stock_cov_local = stock_cov
stock_n = len(stock_assets_list)
stock_bounds_local = tuple((0, 1) for _ in range(stock_n))
stock_constraints_local = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
stock_allocs, stock_vols_f, stock_rets_f = efficient_frontier(
    stock_cov_local,
    stock_net,
    stock_bounds_local,
    stock_constraints_local,
    stock_assets_list,
)

# All assets
assets_with_bond = stock_assets + [bond_name, gold_name]
net_with_bond = np.concatenate([stock_net_returns, np.array([bond_net, gold_net])])
vols_with_bond = np.concatenate([stock_vols, np.array([bond_vol, gold_vol])])
# correlation: stock_corr in top-left, bond/gold zero-corr with others (diagonal 1)
nb = len(assets_with_bond)
corr_with_bond = np.zeros((nb, nb), dtype=float)
corr_with_bond[:4, :4] = stock_corr
corr_with_bond[4, 4] = 1.0
corr_with_bond[5, 5] = 1.0
cov_with_bond = np.outer(vols_with_bond, vols_with_bond) * corr_with_bond

bounds_with_bond = tuple((0, 1) for _ in range(nb))
constraints_with_bond = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
allocations, achieved_vols, achieved_returns = efficient_frontier(
    cov_with_bond,
    net_with_bond,
    bounds_with_bond,
    constraints_with_bond,
    assets_with_bond,
)

# --- Allocation vs Volatility ---
plt.figure(figsize=(12, 6))
plt.stackplot(achieved_vols, allocations.T, labels=assets_with_bond)
plt.xlabel("Portfolio Volatility")
plt.ylabel("Allocation")
plt.title("Optimal Allocations Along Efficient Frontier")
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
ax = plt.gca()
ax.set_yticks(np.arange(0.0, 1.01, 0.1))
plt.tight_layout()
plt.show()
