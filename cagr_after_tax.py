def after_tax_cagr(
    initial_amount: float,
    dividend_rate: float,  # annual dividend yield, e.g. 0.03
    gross_return_rate: float,  # before-tax total return (including div), e.g. 0.07
    dividend_tax_rate: float,  # tax rate on dividends, e.g. 0.1032
    capital_gain_tax_rate: float,  # tax rate on capital gains, e.g. 0.1825
    years: int,
):
    """
    Compute the after-tax CAGR of an investment.

    Assumptions:
    - Gross return = price return + dividend yield.
    - Dividends taxed annually; after-tax dividends reinvested.
    - Capital gains taxed at the end on (final value - total contributed).
    - Contributions include initial amount + reinvested after-tax dividends.
    """

    value = initial_amount
    total_contributions = initial_amount

    for _ in range(years):
        # 1. Compute dividend for the year
        dividend = value * dividend_rate
        after_tax_dividend = dividend * (1 - dividend_tax_rate)

        # 2. Reinvest after-tax dividend
        value += after_tax_dividend
        total_contributions += after_tax_dividend

        # 3. Apply price growth (gross return minus dividend portion)
        price_return_rate = gross_return_rate - dividend_rate
        value *= 1 + price_return_rate

    # --- Capital gains tax at the end ---
    capital_gain = value - total_contributions
    after_tax_capital_gain = capital_gain * (1 - capital_gain_tax_rate)

    final_after_tax_value = total_contributions + after_tax_capital_gain

    # CAGR formula
    cagr = (final_after_tax_value / initial_amount) ** (1 / years) - 1

    return final_after_tax_value, cagr


# ---------------- Example Usage ----------------

if __name__ == "__main__":
    initial_amount = 100000
    dividend_rate = 0.0227 * 1.3  #
    gross_return_rate = 0.0779  #
    dividend_tax_rate = 0.1032  # Canadian 0.1032
    capital_gain_tax_rate = 0.1725
    years = 30

    final_value, cagr = after_tax_cagr(
        initial_amount,
        dividend_rate,
        gross_return_rate,
        dividend_tax_rate,
        capital_gain_tax_rate,
        years,
    )

    print(f"Final after-tax value: ${final_value:,.2f}")
    print(f"After-tax CAGR: {cagr*100:.2f}%")
