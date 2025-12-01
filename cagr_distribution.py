import numpy as np

# from math import exp, sqrt
import matplotlib.pyplot as plt


# def cagr_bounds(mu=0.07, sigma=0.125, n=30, conf=0.90):
#     # Get z-value for confidence interval
#     z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[conf]

#     mu_l = mu - sigma**2 / 2  # lognormal drift
#     s = sigma / sqrt(n)  # annualized uncertainty term

#     cagr_lower = exp(mu_l - z * s) - 1
#     cagr_upper = exp(mu_l + z * s) - 1
#     cagr_median = exp(mu_l) - 1
#     cagr_efv = exp(mu) - 1  # implied CAGR of expected FV

#     return {
#         "CAGR Lower": cagr_lower,
#         "CAGR Median": cagr_median,
#         "CAGR Upper": cagr_upper,
#         "CAGR from Expected FV": cagr_efv,
#     }

# # Example:
# results = cagr_bounds(mu=0.07, sigma=0.125, n=30, conf=0.90)
# for k, v in results.items():
#     print(f"{k:25s}: {v*100:8.4f}%")


# Inputs
mu = 0.07  # arithmetic mean return
sigma = 0.125  # annual volatility
n = 30  # years
paths = 200_000  # Monte Carlo simulations

# Derived lognormal parameters
mu_l = mu - sigma**2 / 2
sigma_l_n = sigma * np.sqrt(n)

# Monte Carlo log-wealth
log_fv = np.random.normal(loc=n * mu_l, scale=sigma_l_n, size=paths)

# Terminal wealth and CAGR
fv = np.exp(log_fv)
cagr = fv ** (1 / n) - 1

# Percentiles for annotation
p5, p50, p95 = np.percentile(cagr, [5, 50, 95])

# Plot
plt.figure(figsize=(10, 6))
plt.hist(cagr * 100, bins=200, density=True, alpha=0.7)
plt.axvline(p5 * 100, color="r", linestyle="--", label=f"5th: {p5*100:.2f}%")
plt.axvline(p50 * 100, color="g", linestyle="--", label=f"Median: {p50*100:.2f}%")
plt.axvline(p95 * 100, color="b", linestyle="--", label=f"95th: {p95*100:.2f}%")

plt.title("Distribution of 30-Year Annualized Returns (CAGR)")
plt.xlabel("Annualized Return (%)")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
