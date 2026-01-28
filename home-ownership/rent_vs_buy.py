"""
Rent vs Buy Calculator

Compares three housing strategies over 25 years:
1. Rent: Renting with investing in non-registered account
2. Buy Traditional: Standard 25-year mortgage, no HELOC
3. Buy Smith Maneuver: 25-year mortgage + HELOC investing

Reuses assumptions from Smith Maneuver simulator.
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/leohong/backtesting/home-ownership")

from smith_maneuver import SimulationAssumptions, SmithManeuverSimulator


# =============================================================================
# RENT VS BUY ASSUMPTIONS - Rent-specific parameters
# =============================================================================
class RentVsBuyAssumptions:
    """
    Rent-specific assumptions for rent vs buy comparison.

    Key assumption: Renter has same total housing budget as buyer.
    Excess savings are invested in taxable (non-registered) account.

    Investment Strategy:
    - Renter invests in 100% EQUITY PORTFOLIO (same as buyer)
    - Annual return: 6% (same as SimulationAssumptions)
    - Tax-efficient structure in taxable account:
      * Distribution yield: 1.5% (dividends, interest) - taxed annually
      * Capital appreciation: 4.5% (price growth) - taxed only at realization
      * After-tax return: ~5.37% (1.5% × (1-42%) + 4.5%)
    - Investment fees: 0.2% MER
    """

    # RENT ASSUMPTIONS
    # Option 1: Specify as % of property value
    annual_rent_pct_of_home_value = None  # Set to None if using fixed monthly rent

    # Option 2: Specify as fixed monthly rent (RECOMMENDED)
    monthly_rent = 5000  # Fixed monthly rent in dollars ($5k/month = $60k/year)

    # Rent increases 
    annual_rent_increase = 0.025

    # Portfolio Strategy
    # Renter invests in all-equity portfolio (100% stocks, 0% bonds/cash)
    # This matches the buyer's investment behavior for fair comparison
    renter_portfolio_strategy = "all_equity"  # 100% equity portfolio

    # Taxable Account Income Components
    # In a taxable account, income is split into distribution yield (taxed annually)
    # and capital appreciation (taxed only upon realization)
    distribution_yield = 0.015  # 1.5% annual distribution (dividends, interest)
    capital_appreciation = (
        0.045  # 4.5% annual capital appreciation (untaxed until realized)
    )
    # Total return: 6% (1.5% + 4.5%)

    # CLOSING COSTS
    # Typical closing costs for property transactions
    buyer_closing_cost_pct = (
        0.025  # 2.5% of purchase price (includes legal, inspection, title insurance)
    )
    seller_closing_cost_pct = (
        0.05  # 5% of sale price (includes realtor commission ~2.5% + legal/fees)
    )

    # KEY ASSUMPTION: Renter discipline
    # - Renter can afford same total housing expenses as buyer
    # - Excess cash flow (buyer costs - rent) gets invested monthly
    # - Renter maintains investment discipline for 25 years
    # - This assumes: rent < (mortgage + tax + maintenance + insurance)


class RentVsBuyCalculator:
    """Compares renting vs buying strategies over 25 years."""

    def __init__(
        self,
        assumptions: SimulationAssumptions = None,
        rent_assumptions: RentVsBuyAssumptions = None,
        house_price: float = None,
        down_payment_pct: float = None,
        mortgage_rate: float = None,
        heloc_rate: float = None,
        annual_investment_return: float = None,
        annual_property_appreciation: float = None,
        marginal_tax_rate: float = None,
        years: int = None,
    ):
        """Initialize calculator with assumptions."""
        if assumptions is None:
            assumptions = SimulationAssumptions()
        if rent_assumptions is None:
            rent_assumptions = RentVsBuyAssumptions()

        # Buy-side assumptions (from SimulationAssumptions)
        self.house_price = house_price or assumptions.house_price
        self.down_payment_pct = down_payment_pct or assumptions.down_payment_pct
        self.down_payment = self.house_price * self.down_payment_pct
        self.initial_mortgage = self.house_price - self.down_payment

        self.mortgage_rate = mortgage_rate or assumptions.mortgage_rate
        self.heloc_rate = heloc_rate or assumptions.heloc_rate
        self.annual_investment_return = (
            annual_investment_return or assumptions.annual_investment_return
        )
        self.annual_property_appreciation = (
            annual_property_appreciation or assumptions.annual_property_appreciation
        )
        self.marginal_tax_rate = marginal_tax_rate or assumptions.marginal_tax_rate
        self.years = years or assumptions.years

        self.annual_maintenance_pct = assumptions.annual_maintenance_pct
        self.annual_property_tax_pct = assumptions.annual_property_tax_pct
        self.annual_home_insurance_pct = assumptions.annual_home_insurance_pct
        self.investment_fee_pct = assumptions.investment_fee_pct

        # Rent-side assumptions
        self.annual_rent_pct = rent_assumptions.annual_rent_pct_of_home_value
        self.monthly_rent = rent_assumptions.monthly_rent
        self.annual_rent_increase = rent_assumptions.annual_rent_increase
        self.distribution_yield = rent_assumptions.distribution_yield
        self.capital_appreciation = rent_assumptions.capital_appreciation

        # Closing cost assumptions
        self.buyer_closing_cost_pct = rent_assumptions.buyer_closing_cost_pct
        self.seller_closing_cost_pct = rent_assumptions.seller_closing_cost_pct
        self.buyer_closing_cost = self.house_price * self.buyer_closing_cost_pct
        self.seller_closing_cost = (
            None  # Calculated at year 25 based on final house value
        )

        # Calculate initial annual rent based on either percentage or fixed monthly amount
        if self.annual_rent_pct is not None:
            # Use percentage-based rent
            self.initial_annual_rent = self.house_price * self.annual_rent_pct
        elif self.monthly_rent is not None:
            # Use fixed monthly rent
            self.initial_annual_rent = self.monthly_rent * 12
        else:
            # Default to 6% if neither specified
            self.initial_annual_rent = self.house_price * 0.06

    def calculate_monthly_payment(
        self, principal: float, annual_rate: float, months: int
    ) -> float:
        """Calculate fixed monthly payment using mortgage formula."""
        if annual_rate == 0:
            return principal / months
        monthly_rate = annual_rate / 12
        numerator = principal * monthly_rate * (1 + monthly_rate) ** months
        denominator = (1 + monthly_rate) ** months - 1
        return numerator / denominator

    def simulate_rent(self, leverage_ratio: float = 1.0) -> pd.DataFrame:
        """
        Simulate renting scenario where renter invests excess savings.

        KEY ASSUMPTIONS:
        1. Renter has same total housing budget as a buyer
           - Buyer pays: mortgage + property tax + maintenance + insurance
           - Renter pays: rent only
           - Difference is invested monthly (disciplined renter)

        2. Renter invests in 100% EQUITY PORTFOLIO in TAXABLE ACCOUNT:
           - Annual pre-tax return: 6% (1.5% distribution + 4.5% capital appreciation)
           - Distribution yield: 1.5% (taxed annually at 42%)
           - Capital appreciation: 4.5% (taxed only at realization in year 25)
           - After-tax effective return: ~5.37%
           - Investment fees: 0.2% (MER)
           - This matches the buyer's investment behavior

        3. Starting capital: Down payment ($200,000) invested immediately

        4. Optional leverage with margin loans (DYNAMIC REBALANCING):
           - leverage_ratio=1.0: No leverage (standard renting)
           - leverage_ratio=1.25: Maintain 25% borrowed (1.25x invested) - rebalance annually
           - leverage_ratio=1.35: Maintain 35% borrowed (1.35x invested) - rebalance annually
           - Margin loan rate: Same as HELOC rate (5.25%)
           - Interest is tax-deductible (similar to mortgage/HELOC)
           - Key: Portfolio grows faster (6%) than debt (5.25%), so rebalance annually
           - to maintain target leverage by borrowing additional funds when needed.
           - If portfolio shrinks, margin debt stays same (one-way ratchet).

        5. Monthly compound calculations:
           - Excess is invested each month
           - Investment returns compound monthly
           - Distribution taxes deducted annually
           - Capital gains tax deferred until year 25
           - Fees deducted annually
           - Margin interest compounded into debt annually (with tax benefit applied to portfolio)
           - Annual rebalancing: Calculate target debt to maintain leverage_ratio,
             borrow more if portfolio grew, don't repay if it shrunk

        Result: Fair apples-to-apples comparison with realistic tax treatment
        """
        results = []
        annual_rent = self.initial_annual_rent
        # Renter starts with down_payment + buyer_closing_cost
        # (since they don't incur buyer closing costs like home buyers do)
        # This makes the comparison fair: renter has this capital available
        # while buyer uses it for the transaction
        investment_portfolio = (
            self.down_payment + self.buyer_closing_cost
        )  # Start with down payment + avoided closing cost

        # For leverage: calculate initial margin borrowed (upfront)
        upfront_margin_borrowed = (
            investment_portfolio * (leverage_ratio - 1.0) if leverage_ratio > 1.0 else 0
        )
        investment_portfolio += (
            upfront_margin_borrowed  # Add borrowed funds to investment
        )
        margin_debt = upfront_margin_borrowed  # Start with upfront borrowed amount

        # We need to calculate buyer costs to know how much renter can invest
        # Run a quick buy traditional simulation to get annual costs
        simulator = SmithManeuverSimulator(
            house_price=self.house_price,
            down_payment_pct=self.down_payment_pct,
            mortgage_rate=self.mortgage_rate,
            annual_investment_return=self.annual_investment_return,
            annual_property_appreciation=self.annual_property_appreciation,
            marginal_tax_rate=self.marginal_tax_rate,
            years=self.years,
        )
        buy_sim = simulator.simulate_traditional()

        for year in range(self.years):
            buy_costs_this_year = (
                buy_sim.iloc[year]["annual_mortgage_payment"]
                + buy_sim.iloc[year]["annual_property_tax"]
                + buy_sim.iloc[year]["annual_maintenance"]
                + buy_sim.iloc[year]["annual_insurance"]
            )

            # Excess amount renter can invest
            # (buyer's annual costs minus rent, invested each month)
            # Note: buyer_closing_cost is NOT added in year 0 because the renter
            # already starts with this amount in their initial portfolio
            excess_per_month = (buy_costs_this_year - annual_rent) / 12

            yearly_data = {
                "year": year,
                "annual_rent_paid": 0,
                "cumulative_rent_paid": 0,
                "buyer_annual_costs": buy_costs_this_year,
                "excess_available_for_investing": 0,
                "investment_portfolio": investment_portfolio,
                "margin_debt": margin_debt,
                "annual_investment_gain": 0,
                "annual_distribution_yield": 0,
                "annual_distribution_tax": 0,
                "annual_capital_appreciation": 0,
                "annual_investment_fee": 0,
                "annual_margin_interest": 0,
                "annual_margin_interest_tax_benefit": 0,
                "annual_margin_rebalancing": 0,
                "annual_capital_gains_tax": 0,
                "total_liquid_wealth": investment_portfolio - margin_debt,
                "annual_rent_increase": 0,
                "is_final_year": (
                    year == self.years - 1
                ),  # Track final year for capital gains tax
            }

            # Process 12 months with monthly compounding
            for month in range(12):
                # Step 1: Renter pays rent
                monthly_rent = annual_rent / 12
                yearly_data["annual_rent_paid"] += monthly_rent

                # Step 2: Renter invests excess (no additional leverage per month)
                # The excess is the difference between buyer's costs and rent
                monthly_investment = excess_per_month
                yearly_data["excess_available_for_investing"] += monthly_investment
                investment_portfolio += monthly_investment

                # Step 3: Calculate monthly investment return
                # Split into distribution yield (1.5%) and capital appreciation (4.5%)
                monthly_return_rate = (1 + self.annual_investment_return) ** (
                    1 / 12
                ) - 1
                investment_gain = investment_portfolio * monthly_return_rate
                investment_portfolio += investment_gain
                yearly_data["annual_investment_gain"] += investment_gain

            # Step 4: At year-end, handle tax on distribution yield
            # Distribution yield is 1.5% of pre-tax returns
            # This gets taxed annually at marginal rate (42%)
            distribution_portion = yearly_data["annual_investment_gain"] * (
                self.distribution_yield / self.annual_investment_return
            )
            distribution_tax = distribution_portion * self.marginal_tax_rate
            investment_portfolio -= distribution_tax
            yearly_data["annual_distribution_yield"] = distribution_portion
            yearly_data["annual_distribution_tax"] = distribution_tax
            yearly_data["annual_capital_appreciation"] = (
                yearly_data["annual_investment_gain"] - distribution_portion
            )

            # Step 5: Investment fees
            yearly_data["annual_investment_fee"] = (
                investment_portfolio * self.investment_fee_pct
            )
            investment_portfolio -= yearly_data["annual_investment_fee"]

            # Step 6: Margin interest and debt growth (if leveraged)
            if leverage_ratio > 1.0 and margin_debt > 0:
                # Margin debt grows by interest accrual
                annual_margin_interest = margin_debt * self.heloc_rate
                yearly_data["annual_margin_interest"] = annual_margin_interest

                # Margin interest is tax-deductible
                # Tax benefit = interest * marginal rate
                margin_interest_tax_benefit = (
                    annual_margin_interest * self.marginal_tax_rate
                )
                yearly_data["annual_margin_interest_tax_benefit"] = (
                    margin_interest_tax_benefit
                )

                # Apply tax benefit to portfolio (offset the after-tax cost)
                investment_portfolio += margin_interest_tax_benefit

                # Margin debt grows by accrued interest
                margin_debt += annual_margin_interest
            else:
                yearly_data["annual_margin_interest"] = 0
                yearly_data["annual_margin_interest_tax_benefit"] = 0

            # Step 7: Dynamic leverage rebalancing (AFTER all gains/taxes/fees are applied)
            # This is at year-end, so we rebalance to maintain target leverage_ratio
            if leverage_ratio > 1.0:
                # Target debt = portfolio * (leverage_ratio - 1.0)
                # Example: 1.25x leverage means borrow 25% of current portfolio
                target_margin_debt = investment_portfolio * (leverage_ratio - 1.0)

                # Only increase debt if portfolio grew (one-way ratchet)
                # Never pay down debt voluntarily
                if target_margin_debt > margin_debt:
                    new_borrowing = target_margin_debt - margin_debt
                    investment_portfolio += new_borrowing  # Add newly borrowed funds
                    margin_debt = target_margin_debt
                    yearly_data["annual_margin_rebalancing"] = new_borrowing
                else:
                    yearly_data["annual_margin_rebalancing"] = 0

            # Step 8: Capital gains tax in final year
            # Capital gains are only taxed when realized (at end of year 25)
            # In Canada, only 50% of capital gains are taxable
            if year == self.years - 1:
                # Calculate total capital gains realized
                capital_gains_accrued = yearly_data["annual_capital_appreciation"]
                # Only 50% of capital gains are taxable in Canada
                taxable_capital_gain = capital_gains_accrued * 0.5
                capital_gains_tax = taxable_capital_gain * self.marginal_tax_rate
                investment_portfolio -= capital_gains_tax
                yearly_data["annual_capital_gains_tax"] = capital_gains_tax
            else:
                yearly_data["annual_capital_gains_tax"] = 0

            # Annual rent increase
            rent_increase = annual_rent * self.annual_rent_increase
            annual_rent += rent_increase
            yearly_data["annual_rent_increase"] = rent_increase

            # Update totals
            yearly_data["investment_portfolio"] = investment_portfolio
            yearly_data["margin_debt"] = margin_debt
            yearly_data["total_liquid_wealth"] = investment_portfolio - margin_debt

            results.append(yearly_data)

        return pd.DataFrame(results)

    def simulate_buy_traditional(self) -> pd.DataFrame:
        """Simulate traditional buy scenario using embedded simulator."""
        simulator = SmithManeuverSimulator(
            house_price=self.house_price,
            down_payment_pct=self.down_payment_pct,
            mortgage_rate=self.mortgage_rate,
            annual_investment_return=self.annual_investment_return,
            annual_property_appreciation=self.annual_property_appreciation,
            marginal_tax_rate=self.marginal_tax_rate,
            years=self.years,
        )
        return simulator.simulate_traditional()

    def simulate_buy_smith(self) -> pd.DataFrame:
        """Simulate Smith Maneuver buy scenario using embedded simulator."""
        simulator = SmithManeuverSimulator(
            house_price=self.house_price,
            down_payment_pct=self.down_payment_pct,
            mortgage_rate=self.mortgage_rate,
            heloc_rate=self.heloc_rate,
            annual_investment_return=self.annual_investment_return,
            annual_property_appreciation=self.annual_property_appreciation,
            marginal_tax_rate=self.marginal_tax_rate,
            years=self.years,
        )
        return simulator.simulate_smith_maneuver()

    def simulate_buy_smith_light(self) -> pd.DataFrame:
        """Simulate Smith Maneuver with Light leverage (50% LTV - conservative)."""
        simulator = SmithManeuverSimulator(
            house_price=self.house_price,
            down_payment_pct=self.down_payment_pct,
            mortgage_rate=self.mortgage_rate,
            heloc_rate=self.heloc_rate,
            annual_investment_return=self.annual_investment_return,
            annual_property_appreciation=self.annual_property_appreciation,
            marginal_tax_rate=self.marginal_tax_rate,
            years=self.years,
            combined_ltv_target=0.50,
        )
        return simulator.simulate_smith_maneuver()

    def simulate_buy_smith_median(self) -> pd.DataFrame:
        """Simulate Smith Maneuver with Median leverage (65% LTV - moderate)."""
        simulator = SmithManeuverSimulator(
            house_price=self.house_price,
            down_payment_pct=self.down_payment_pct,
            mortgage_rate=self.mortgage_rate,
            heloc_rate=self.heloc_rate,
            annual_investment_return=self.annual_investment_return,
            annual_property_appreciation=self.annual_property_appreciation,
            marginal_tax_rate=self.marginal_tax_rate,
            years=self.years,
            combined_ltv_target=0.65,
        )
        return simulator.simulate_smith_maneuver()

    def simulate_buy_smith_heavy(self) -> pd.DataFrame:
        """Simulate Smith Maneuver with Heavy leverage (80% LTV - aggressive)."""
        simulator = SmithManeuverSimulator(
            house_price=self.house_price,
            down_payment_pct=self.down_payment_pct,
            mortgage_rate=self.mortgage_rate,
            heloc_rate=self.heloc_rate,
            annual_investment_return=self.annual_investment_return,
            annual_property_appreciation=self.annual_property_appreciation,
            marginal_tax_rate=self.marginal_tax_rate,
            years=self.years,
            combined_ltv_target=0.80,
        )
        return simulator.simulate_smith_maneuver()

    def print_comparison(
        self,
        rent: pd.DataFrame,
        rent_margin_125: pd.DataFrame,
        rent_margin_135: pd.DataFrame,
        buy_traditional: pd.DataFrame,
        buy_smith: pd.DataFrame,
    ) -> None:
        """Print detailed comparison of all five scenarios."""
        rent_final = rent.iloc[-1]
        rent_margin_125_final = rent_margin_125.iloc[-1]
        rent_margin_135_final = rent_margin_135.iloc[-1]
        trad_final = buy_traditional.iloc[-1]
        smith_final = buy_smith.iloc[-1]

        # Calculate seller closing costs based on final house value
        final_house_value = trad_final["house_value"]
        seller_closing_cost = final_house_value * self.seller_closing_cost_pct

        print("\n" + "=" * 110)
        print("RENT VS BUY COMPARISON - 25 YEAR HORIZON")
        print("=" * 110)

        print(f"\nAssumptions:")
        print(f"  House Price: ${self.house_price:,.0f}")
        print(
            f"  Down Payment: ${self.down_payment:,.0f} ({self.down_payment_pct*100:.0f}%)"
        )
        print(
            f"  Buyer Closing Costs: ${self.buyer_closing_cost:,.0f} ({self.buyer_closing_cost_pct*100:.1f}%)"
        )
        print(
            f"  Seller Closing Costs (at sale): {self.seller_closing_cost_pct*100:.1f}% of final price"
        )
        print(f"  Mortgage Rate: {self.mortgage_rate*100:.2f}%")
        print(f"  HELOC Rate: {self.heloc_rate*100:.2f}%")
        print(f"  Investment Return: {self.annual_investment_return*100:.2f}%")
        print(f"  Property Appreciation: {self.annual_property_appreciation*100:.2f}%")
        print(f"  Marginal Tax Rate: {self.marginal_tax_rate*100:.0f}%")
        print(f"\nRent Assumptions:")
        if self.annual_rent_pct is not None:
            print(f"  Annual Rent (% of home value): {self.annual_rent_pct*100:.1f}%")
        else:
            print(f"  Monthly Rent: ${self.monthly_rent:,.0f}")
        print(f"  Initial Annual Rent: ${self.initial_annual_rent:,.0f}")
        print(f"  Annual Rent Increase: {self.annual_rent_increase*100:.2f}%")
        print(f"  Investment Return Breakdown (Taxable Account):")
        print(
            f"    - Distribution Yield: {self.distribution_yield*100:.1f}% (taxed annually at 42%)"
        )
        print(
            f"    - Capital Appreciation: {self.capital_appreciation*100:.1f}% (taxed at realization, 50% inclusion)"
        )
        print(f"    - Total Pre-tax Return: {self.annual_investment_return*100:.1f}%")
        print(
            f"    - Effective After-tax Return: ~{(self.distribution_yield * (1 - self.marginal_tax_rate) + self.capital_appreciation)*100:.2f}%"
        )
        print(f"  KEY ASSUMPTION: Renter invests excess (buyer costs - rent)")
        print(f"  This assumes renter has same total housing budget as buyer")

        # Scenario 1: Rent
        print("\n" + "=" * 110)
        print("SCENARIO 1: RENT (no leverage)")
        print("=" * 110)
        print(f"Total Rent Paid (25 years): ${rent['annual_rent_paid'].sum():,.0f}")
        print(
            f"Final Annual Rent: ${self.initial_annual_rent * (1 + self.annual_rent_increase) ** self.years:,.0f}"
        )
        print(
            f"Total Excess Invested (25 years): ${rent['excess_available_for_investing'].sum():,.0f}"
        )
        print(f"Investment Portfolio: ${rent_final['investment_portfolio']:,.0f}")
        print(f"Total Liquid Wealth: ${rent_final['total_liquid_wealth']:,.0f}")
        print(f"\nBreakdown:")
        print(
            f"  Starting investment (down payment equivalent): ${self.down_payment:,.0f}"
        )
        print(
            f"  Cumulative excess from (buyer costs - rent): +${rent['excess_available_for_investing'].sum():,.0f}"
        )
        print(
            f"  Investment growth over 25 years: +${rent_final['investment_portfolio'] - self.down_payment - rent['excess_available_for_investing'].sum():,.0f}"
        )
        print(
            f"  Notes: No real estate asset, fully liquid, but requires investment discipline"
        )

        # Scenario 2: Rent with 1.25x Margin
        print("\n" + "=" * 110)
        print("SCENARIO 2: RENT (with 1.25x margin leverage)")
        print("=" * 110)
        print(
            f"Total Rent Paid (25 years): ${rent_margin_125['annual_rent_paid'].sum():,.0f}"
        )
        print(
            f"Final Annual Rent: ${self.initial_annual_rent * (1 + self.annual_rent_increase) ** self.years:,.0f}"
        )
        print(
            f"Total Excess Invested (25 years): ${rent_margin_125['excess_available_for_investing'].sum():,.0f}"
        )
        print(
            f"Total Amount Invested (with 1.25x leverage): ${rent_margin_125['excess_available_for_investing'].sum() * 1.25:,.0f}"
        )
        print(f"Final Margin Debt: ${rent_margin_125_final['margin_debt']:,.0f}")
        print(
            f"Investment Portfolio: ${rent_margin_125_final['investment_portfolio']:,.0f}"
        )
        print(
            f"Net Liquid Wealth (Portfolio - Margin Debt): ${rent_margin_125_final['total_liquid_wealth']:,.0f}"
        )
        print(
            f"Total Margin Interest Paid: ${rent_margin_125['annual_margin_interest'].sum():,.0f}"
        )
        print(
            f"Total Margin Interest Tax Benefit: ${rent_margin_125['annual_margin_interest_tax_benefit'].sum():,.0f}"
        )
        print(f"\nBreakdown:")
        print(
            f"  Starting investment (down payment equivalent): ${self.down_payment:,.0f}"
        )
        print(f"  Initial margin borrowed: ${self.down_payment * 0.25:,.0f}")
        print(
            f"  Cumulative excess invested (25% borrowed): +${rent_margin_125['excess_available_for_investing'].sum() * 1.25:,.0f}"
        )
        print(
            f"  Investment growth over 25 years: +${rent_margin_125_final['investment_portfolio'] - (self.down_payment * 1.25) - (rent_margin_125['excess_available_for_investing'].sum() * 1.25):,.0f}"
        )
        print(
            f"  Margin interest (net of tax benefit): ${rent_margin_125['annual_margin_interest'].sum() - rent_margin_125['annual_margin_interest_tax_benefit'].sum():,.0f}"
        )

        # Scenario 3: Rent with 1.35x Margin
        print("\n" + "=" * 110)
        print("SCENARIO 3: RENT (with 1.35x margin leverage)")
        print("=" * 110)
        print(
            f"Total Rent Paid (25 years): ${rent_margin_135['annual_rent_paid'].sum():,.0f}"
        )
        print(
            f"Final Annual Rent: ${self.initial_annual_rent * (1 + self.annual_rent_increase) ** self.years:,.0f}"
        )
        print(
            f"Total Excess Invested (25 years): ${rent_margin_135['excess_available_for_investing'].sum():,.0f}"
        )
        print(
            f"Total Amount Invested (with 1.35x leverage): ${rent_margin_135['excess_available_for_investing'].sum() * 1.35:,.0f}"
        )
        print(f"Final Margin Debt: ${rent_margin_135_final['margin_debt']:,.0f}")
        print(
            f"Investment Portfolio: ${rent_margin_135_final['investment_portfolio']:,.0f}"
        )
        print(
            f"Net Liquid Wealth (Portfolio - Margin Debt): ${rent_margin_135_final['total_liquid_wealth']:,.0f}"
        )
        print(
            f"Total Margin Interest Paid: ${rent_margin_135['annual_margin_interest'].sum():,.0f}"
        )
        print(
            f"Total Margin Interest Tax Benefit: ${rent_margin_135['annual_margin_interest_tax_benefit'].sum():,.0f}"
        )
        print(f"\nBreakdown:")
        print(
            f"  Starting investment (down payment equivalent): ${self.down_payment:,.0f}"
        )
        print(f"  Initial margin borrowed: ${self.down_payment * 0.35:,.0f}")
        print(
            f"  Cumulative excess invested (35% borrowed): +${rent_margin_135['excess_available_for_investing'].sum() * 1.35:,.0f}"
        )
        print(
            f"  Investment growth over 25 years: +${rent_margin_135_final['investment_portfolio'] - (self.down_payment * 1.35) - (rent_margin_135['excess_available_for_investing'].sum() * 1.35):,.0f}"
        )
        print(
            f"  Margin interest (net of tax benefit): ${rent_margin_135['annual_margin_interest'].sum() - rent_margin_135['annual_margin_interest_tax_benefit'].sum():,.0f}"
        )

        # Scenario 4: Buy Traditional
        print("\n" + "=" * 110)
        print("SCENARIO 4: BUY TRADITIONAL (Standard 25-year mortgage)")
        print("=" * 110)
        print(f"House Value: ${trad_final['house_value']:,.0f}")
        print(f"Mortgage Balance: ${trad_final['mortgage_balance']:,.0f}")
        net_proceeds_traditional = trad_final["net_net_worth"] - seller_closing_cost
        print(
            f"Total Wealth (House less seller closing): ${net_proceeds_traditional:,.0f}"
        )
        print(f"\nCosts over 25 years:")
        print(f"  Buyer Closing Costs (year 0): ${self.buyer_closing_cost:,.0f}")
        print(
            f"  Total Mortgage Paid: ${buy_traditional['annual_mortgage_payment'].sum():,.0f}"
        )
        print(
            f"  Total Mortgage Interest: ${buy_traditional['annual_interest_paid'].sum():,.0f}"
        )
        print(
            f"  Total Property Tax: ${buy_traditional['annual_property_tax'].sum():,.0f}"
        )
        print(
            f"  Total Maintenance: ${buy_traditional['annual_maintenance'].sum():,.0f}"
        )
        print(f"  Total Insurance: ${buy_traditional['annual_insurance'].sum():,.0f}")
        print(f"  Seller Closing Costs (year 25): ${seller_closing_cost:,.0f}")
        total_costs = (
            self.buyer_closing_cost
            + buy_traditional["annual_mortgage_payment"].sum()
            + buy_traditional["annual_property_tax"].sum()
            + buy_traditional["annual_maintenance"].sum()
            + buy_traditional["annual_insurance"].sum()
            + seller_closing_cost
        )
        print(f"  Total Housing Costs: ${total_costs:,.0f}")
        print(
            f"  Cost vs Initial (down payment): ${total_costs - self.down_payment:,.0f}"
        )

        # Scenario 5: Buy Smith Maneuver
        print("\n" + "=" * 110)
        print("SCENARIO 5: BUY SMITH MANEUVER (25-year mortgage + HELOC investing)")
        print("=" * 110)
        print(f"House Value: ${smith_final['house_value']:,.0f}")
        print(f"Mortgage Balance: ${smith_final['mortgage_balance']:,.0f}")
        print(f"HELOC Balance: ${smith_final['heloc_balance']:,.0f}")
        print(f"Investment Portfolio: ${smith_final['investment_portfolio']:,.0f}")
        net_proceeds_smith = smith_final["net_net_worth"] - seller_closing_cost
        print(
            f"Total Wealth (House + Investments - HELOC - Seller Costs): ${net_proceeds_smith:,.0f}"
        )
        print(f"\nCosts and investments:")
        print(f"  Buyer Closing Costs (year 0): ${self.buyer_closing_cost:,.0f}")
        print(
            f"  Total Mortgage Paid: ${buy_smith['annual_mortgage_payment'].sum():,.0f}"
        )
        print(
            f"  Total Mortgage Interest: ${buy_smith['annual_interest_paid_mortgage'].sum():,.0f}"
        )
        print(
            f"  Total HELOC Interest: ${buy_smith['annual_interest_paid_heloc'].sum():,.0f}"
        )
        print(
            f"  Total HELOC Borrowed: ${buy_smith['annual_heloc_borrowing'].sum():,.0f}"
        )
        print(f"  Total Tax Benefits: ${buy_smith['annual_tax_benefit'].sum():,.0f}")
        print(f"  Total Property Tax: ${buy_smith['annual_property_tax'].sum():,.0f}")
        print(f"  Total Maintenance: ${buy_smith['annual_maintenance'].sum():,.0f}")
        print(f"  Total Insurance: ${buy_smith['annual_insurance'].sum():,.0f}")
        print(f"  Seller Closing Costs (year 25): ${seller_closing_cost:,.0f}")

        # Comparison Summary
        print("\n" + "=" * 110)
        print("WEALTH COMPARISON AT YEAR 25 (After all closing costs)")
        print("=" * 110)
        print(f"\n{'Scenario':<45} {'Total Wealth':<25} {'vs Rent (No Leverage)':<25}")
        print("-" * 95)
        print(
            f"{'1. Rent (no leverage)':<45} ${rent_final['total_liquid_wealth']:>23,.0f} {'Baseline':<25}"
        )

        rent_margin_125_advantage = (
            rent_margin_125_final["total_liquid_wealth"]
            - rent_final["total_liquid_wealth"]
        )
        print(
            f"{'2. Rent (1.25x margin)':<45} ${rent_margin_125_final['total_liquid_wealth']:>23,.0f} +${rent_margin_125_advantage:>22,.0f}"
        )

        rent_margin_135_advantage = (
            rent_margin_135_final["total_liquid_wealth"]
            - rent_final["total_liquid_wealth"]
        )
        print(
            f"{'3. Rent (1.35x margin)':<45} ${rent_margin_135_final['total_liquid_wealth']:>23,.0f} +${rent_margin_135_advantage:>22,.0f}"
        )

        buy_trad_advantage = (
            net_proceeds_traditional - rent_final["total_liquid_wealth"]
        )
        print(
            f"{'4. Buy Traditional':<45} ${net_proceeds_traditional:>23,.0f} +${buy_trad_advantage:>22,.0f}"
        )

        buy_smith_advantage = net_proceeds_smith - rent_final["total_liquid_wealth"]
        print(
            f"{'5. Buy Smith Maneuver':<45} ${net_proceeds_smith:>23,.0f} +${buy_smith_advantage:>22,.0f}"
        )

        # Additional Analysis
        print("\n" + "=" * 110)
        print("KEY INSIGHTS")
        print("=" * 110)

        smith_vs_trad = net_proceeds_smith - net_proceeds_traditional
        print(f"\n1. RENT vs BUY TRADITIONAL:")
        print(f"   Buying is better by: ${buy_trad_advantage:,.0f}")
        if buy_trad_advantage > 0:
            print(
                f"   (Property appreciation + mortgage payoff > opportunity cost + closing costs)"
            )
        else:
            print(f"   (Renting with invested down payment outperforms buying)")

        print(f"\n2. BUY TRADITIONAL vs BUY SMITH MANEUVER:")
        print(f"   Smith Maneuver is better by: ${smith_vs_trad:,.0f}")
        print(f"   (Tax-deductible HELOC leverage creates additional wealth)")

        print(f"\n3. RENT vs BUY SMITH MANEUVER:")
        print(f"   Smith Maneuver is better by: ${buy_smith_advantage:,.0f}")
        print(
            f"   (Buying with leverage beats renting by {(buy_smith_advantage/rent_final['total_liquid_wealth']*100):.1f}%)"
        )

        # Liquidity Analysis
        print(f"\n4. LIQUIDITY AT YEAR 25:")
        print(f"   Rent: ${rent_final['total_liquid_wealth']:,.0f} (fully liquid)")
        print(
            f"   Buy Traditional: ${net_proceeds_traditional:,.0f} (after sale, net proceeds)"
        )
        print(
            f"   Buy Smith: ${net_proceeds_smith:,.0f} (after sale, net proceeds + investments)"
        )
        print(
            f"   Note: Buying creates wealth but in illiquid form (house). Renting maintains liquidity."
        )

        # Cost Analysis
        print(f"\n5. CUMULATIVE HOUSING COSTS (25 YEARS):")
        rent_total = rent["annual_rent_paid"].sum()
        buy_trad_total = (
            self.buyer_closing_cost
            + buy_traditional["annual_mortgage_payment"].sum()
            + buy_traditional["annual_property_tax"].sum()
            + buy_traditional["annual_maintenance"].sum()
            + buy_traditional["annual_insurance"].sum()
            + seller_closing_cost
        )
        buy_smith_total = (
            self.buyer_closing_cost
            + buy_smith["annual_mortgage_payment"].sum()
            + buy_smith["annual_interest_paid_heloc"].sum()
            + buy_smith["annual_property_tax"].sum()
            + buy_smith["annual_maintenance"].sum()
            + buy_smith["annual_insurance"].sum()
            + seller_closing_cost
        )
        print(f"   Rent: ${rent_total:,.0f}")
        print(f"   Buy Traditional (inc. closing): ${buy_trad_total:,.0f}")
        print(f"   Buy Smith (inc. closing): ${buy_smith_total:,.0f}")
        print(
            f"   Rent is {((rent_total/buy_trad_total - 1) * 100):+.1f}% vs Traditional"
        )
        print(f"   Rent is {((rent_total/buy_smith_total - 1) * 100):+.1f}% vs Smith")

        # Break-even Analysis
        print(f"\n6. NET OUTCOME:")
        if buy_trad_advantage > 0:
            print(
                f"   Despite higher housing costs (including closing costs), buying wins because of:"
            )
            print(
                f"   • Property appreciation: ${trad_final['house_value'] - self.house_price:,.0f}"
            )
            print(
                f"   • Mortgage payoff as equity build: ${self.initial_mortgage:,.0f}"
            )
            print(
                f"   • Tax-deductible HELOC leverage (Smith): ${smith_vs_trad:,.0f} advantage"
            )
        else:
            print(
                f"   Renting with invested down payment outperforms buying after all costs."
            )

    def plot_networth(
        self,
        rent: pd.DataFrame,
        rent_margin_125: pd.DataFrame,
        rent_margin_135: pd.DataFrame,
        buy_traditional: pd.DataFrame,
        buy_smith: pd.DataFrame,
    ) -> None:
        """Plot net worth over 25 years for all five scenarios.

        At the end of each year, calculates net worth if everything is sold:
        - Rent (no leverage): Investment portfolio (fully liquid)
        - Rent (1.25x margin): Investment portfolio - margin debt
        - Rent (1.35x margin): Investment portfolio - margin debt
        - Buy Traditional: House value - remaining mortgage - seller closing costs
        - Buy Smith: House value + investments - mortgage - HELOC - seller closing costs
        """
        years = []
        rent_networth = []
        rent_margin_125_networth = []
        rent_margin_135_networth = []
        buy_trad_networth = []
        buy_smith_networth = []

        for idx in range(len(rent)):
            year = int(rent.iloc[idx]["year"])
            years.append(year)

            # RENT (no leverage): Just the investment portfolio (liquid)
            rent_nw = rent.iloc[idx]["investment_portfolio"]
            rent_networth.append(rent_nw)

            # RENT (1.25x margin): Portfolio - margin debt
            rent_margin_125_nw = (
                rent_margin_125.iloc[idx]["investment_portfolio"]
                - rent_margin_125.iloc[idx]["margin_debt"]
            )
            rent_margin_125_networth.append(rent_margin_125_nw)

            # RENT (1.35x margin): Portfolio - margin debt
            rent_margin_135_nw = (
                rent_margin_135.iloc[idx]["investment_portfolio"]
                - rent_margin_135.iloc[idx]["margin_debt"]
            )
            rent_margin_135_networth.append(rent_margin_135_nw)

            # BUY TRADITIONAL: House value - mortgage balance - seller closing costs
            house_value = buy_traditional.iloc[idx]["house_value"]
            mortgage_balance = buy_traditional.iloc[idx]["mortgage_balance"]
            seller_closing = house_value * self.seller_closing_cost_pct
            trad_nw = house_value - mortgage_balance - seller_closing
            buy_trad_networth.append(trad_nw)

            # BUY SMITH: House value + investments - mortgage - HELOC - seller closing costs
            house_value = buy_smith.iloc[idx]["house_value"]
            investments = buy_smith.iloc[idx]["investment_portfolio"]
            mortgage_balance = buy_smith.iloc[idx]["mortgage_balance"]
            heloc_balance = buy_smith.iloc[idx]["heloc_balance"]
            seller_closing = house_value * self.seller_closing_cost_pct
            smith_nw = (
                house_value
                + investments
                - mortgage_balance
                - heloc_balance
                - seller_closing
            )
            buy_smith_networth.append(smith_nw)

        # Create plot
        plt.figure(figsize=(14, 8))
        plt.plot(
            years,
            rent_networth,
            marker="o",
            linewidth=2.5,
            label="Rent (no leverage)",
            color="#2E86AB",
        )
        plt.plot(
            years,
            rent_margin_125_networth,
            marker="s",
            linewidth=2.5,
            label="Rent (1.25x margin)",
            color="#A23B72",
        )
        plt.plot(
            years,
            rent_margin_135_networth,
            marker="^",
            linewidth=2.5,
            label="Rent (1.35x margin)",
            color="#F18F01",
        )
        plt.plot(
            years,
            buy_trad_networth,
            marker="D",
            linewidth=2.5,
            label="Buy Traditional",
            color="#6A994E",
        )
        plt.plot(
            years,
            buy_smith_networth,
            marker="*",
            linewidth=2.5,
            markersize=15,
            label="Buy Smith Maneuver",
            color="#BC4749",
        )

        plt.xlabel("Years", fontsize=12, fontweight="bold")
        plt.ylabel("Net Worth ($)", fontsize=12, fontweight="bold")
        plt.title(
            "Net Worth Comparison: Rent (No/With Leverage) vs Buy (Traditional vs Smith)\n25-Year Horizon",
            fontsize=13,
            fontweight="bold",
        )
        plt.legend(fontsize=10, loc="upper left")
        plt.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        # Add value labels at year 25
        final_year_idx = len(years) - 1
        plt.text(
            years[final_year_idx],
            rent_networth[final_year_idx],
            f"${rent_networth[final_year_idx]/1e6:.2f}M",
            fontsize=8,
            va="bottom",
            ha="right",
            color="#2E86AB",
        )
        plt.text(
            years[final_year_idx],
            rent_margin_125_networth[final_year_idx],
            f"${rent_margin_125_networth[final_year_idx]/1e6:.2f}M",
            fontsize=8,
            va="bottom",
            ha="right",
            color="#A23B72",
        )
        plt.text(
            years[final_year_idx],
            rent_margin_135_networth[final_year_idx],
            f"${rent_margin_135_networth[final_year_idx]/1e6:.2f}M",
            fontsize=8,
            va="bottom",
            ha="right",
            color="#F18F01",
        )
        plt.text(
            years[final_year_idx],
            buy_trad_networth[final_year_idx],
            f"${buy_trad_networth[final_year_idx]/1e6:.2f}M",
            fontsize=8,
            va="bottom",
            ha="right",
            color="#6A994E",
        )
        plt.text(
            years[final_year_idx],
            buy_smith_networth[final_year_idx],
            f"${buy_smith_networth[final_year_idx]/1e6:.2f}M",
            fontsize=8,
            va="bottom",
            ha="right",
            color="#BC4749",
        )

        plt.tight_layout()
        plt.savefig(
            "/Users/leohong/backtesting/home-ownership/networth_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("✓ Net worth comparison plot saved to networth_comparison.png")
        plt.show()

    def plot_networth_extended(
        self,
        rent: pd.DataFrame,
        rent_margin_125: pd.DataFrame,
        rent_margin_135: pd.DataFrame,
        buy_traditional: pd.DataFrame,
        buy_smith_light: pd.DataFrame,
        buy_smith_median: pd.DataFrame,
        buy_smith_heavy: pd.DataFrame,
    ) -> None:
        """Plot net worth over 25 years for all seven scenarios including SM stress levels."""
        years = []
        rent_networth = []
        rent_margin_125_networth = []
        rent_margin_135_networth = []
        buy_trad_networth = []
        buy_smith_light_networth = []
        buy_smith_median_networth = []
        buy_smith_heavy_networth = []

        for idx in range(len(rent)):
            year = int(rent.iloc[idx]["year"])
            years.append(year)

            # RENT (no leverage)
            rent_nw = rent.iloc[idx]["investment_portfolio"]
            rent_networth.append(rent_nw)

            # RENT (1.25x margin)
            rent_margin_125_nw = (
                rent_margin_125.iloc[idx]["investment_portfolio"]
                - rent_margin_125.iloc[idx]["margin_debt"]
            )
            rent_margin_125_networth.append(rent_margin_125_nw)

            # RENT (1.35x margin)
            rent_margin_135_nw = (
                rent_margin_135.iloc[idx]["investment_portfolio"]
                - rent_margin_135.iloc[idx]["margin_debt"]
            )
            rent_margin_135_networth.append(rent_margin_135_nw)

            # BUY TRADITIONAL
            house_value = buy_traditional.iloc[idx]["house_value"]
            mortgage_balance = buy_traditional.iloc[idx]["mortgage_balance"]
            seller_closing = house_value * self.seller_closing_cost_pct
            trad_nw = house_value - mortgage_balance - seller_closing
            buy_trad_networth.append(trad_nw)

            # BUY SMITH LIGHT (50% LTV)
            house_value = buy_smith_light.iloc[idx]["house_value"]
            investments = buy_smith_light.iloc[idx]["investment_portfolio"]
            mortgage_balance = buy_smith_light.iloc[idx]["mortgage_balance"]
            heloc_balance = buy_smith_light.iloc[idx]["heloc_balance"]
            seller_closing = house_value * self.seller_closing_cost_pct
            smith_light_nw = (
                house_value
                + investments
                - mortgage_balance
                - heloc_balance
                - seller_closing
            )
            buy_smith_light_networth.append(smith_light_nw)

            # BUY SMITH MEDIAN (65% LTV)
            house_value = buy_smith_median.iloc[idx]["house_value"]
            investments = buy_smith_median.iloc[idx]["investment_portfolio"]
            mortgage_balance = buy_smith_median.iloc[idx]["mortgage_balance"]
            heloc_balance = buy_smith_median.iloc[idx]["heloc_balance"]
            seller_closing = house_value * self.seller_closing_cost_pct
            smith_median_nw = (
                house_value
                + investments
                - mortgage_balance
                - heloc_balance
                - seller_closing
            )
            buy_smith_median_networth.append(smith_median_nw)

            # BUY SMITH HEAVY (80% LTV)
            house_value = buy_smith_heavy.iloc[idx]["house_value"]
            investments = buy_smith_heavy.iloc[idx]["investment_portfolio"]
            mortgage_balance = buy_smith_heavy.iloc[idx]["mortgage_balance"]
            heloc_balance = buy_smith_heavy.iloc[idx]["heloc_balance"]
            seller_closing = house_value * self.seller_closing_cost_pct
            smith_heavy_nw = (
                house_value
                + investments
                - mortgage_balance
                - heloc_balance
                - seller_closing
            )
            buy_smith_heavy_networth.append(smith_heavy_nw)

        # Create plot with 7 scenarios
        # Color scheme: Rent (orange family), Buy (blue family)
        plt.figure(figsize=(16, 9))

        # RENT SCENARIOS - Orange family (light to dark)
        plt.plot(
            years,
            rent_networth,
            marker="o",
            linewidth=2.5,
            label="Rent (no leverage)",
            color="#FFB84D",  # Light orange
        )
        plt.plot(
            years,
            rent_margin_125_networth,
            marker="s",
            linewidth=2.5,
            label="Rent (1.25x margin)",
            color="#FF9F1C",  # Medium orange
        )
        plt.plot(
            years,
            rent_margin_135_networth,
            marker="^",
            linewidth=2.5,
            label="Rent (1.35x margin)",
            color="#E67E22",  # Dark orange
        )

        # BUY SCENARIOS - Blue family (light to dark)
        plt.plot(
            years,
            buy_trad_networth,
            marker="D",
            linewidth=2.5,
            label="Buy Traditional",
            color="#87CEEB",  # Light blue
        )
        plt.plot(
            years,
            buy_smith_light_networth,
            marker="v",
            linewidth=2.5,
            label="Smith Light (50% LTV)",
            color="#4A90E2",  # Medium blue
        )
        plt.plot(
            years,
            buy_smith_median_networth,
            marker="p",
            linewidth=2.5,
            label="Smith Median (65% LTV)",
            color="#2E5C8A",  # Darker blue
        )
        plt.plot(
            years,
            buy_smith_heavy_networth,
            marker="*",
            linewidth=3,
            markersize=15,
            label="Smith Heavy (80% LTV)",
            color="#1A3A52",  # Very dark blue
        )

        plt.xlabel("Years", fontsize=12, fontweight="bold")
        plt.ylabel("Net Worth ($)", fontsize=12, fontweight="bold")
        plt.title(
            "Net Worth Comparison: Rent vs Buy (Traditional vs Smith with Stress Scenarios)\n25-Year Horizon",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )

        # Legend with color grouping labels
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()

        # Group legend: Rent (light to dark orange), Buy (light to dark blue)
        legend = ax.legend(
            fontsize=10,
            loc="upper left",
            framealpha=0.95,
            edgecolor="black",
            fancybox=True,
            shadow=True,
        )

        # Add group labels as text annotations
        legend_y_top = 0.98
        ax.text(
            0.01,
            legend_y_top,
            "RENT SCENARIOS (Orange)",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="#E67E22",
            verticalalignment="top",
        )

        ax.text(
            0.01,
            legend_y_top - 0.18,
            "BUY SCENARIOS (Blue)",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="#1A3A52",
            verticalalignment="top",
        )

        plt.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        plt.tight_layout()
        plt.savefig(
            "/Users/leohong/backtesting/home-ownership/networth_comparison_extended.png",
            dpi=300,
            bbox_inches="tight",
        )
        print(
            "✓ Extended net worth comparison plot saved to networth_comparison_extended.png"
        )
        plt.show()


def main():
    """Run rent vs buy comparison."""
    calculator = RentVsBuyCalculator()

    print("\nRunning Rent vs Buy calculator - 25 year horizon...\n")

    # Run all scenarios
    rent = calculator.simulate_rent(leverage_ratio=1.0)
    rent_margin_125 = calculator.simulate_rent(leverage_ratio=1.25)
    rent_margin_135 = calculator.simulate_rent(leverage_ratio=1.35)
    buy_traditional = calculator.simulate_buy_traditional()
    buy_smith_light = calculator.simulate_buy_smith_light()
    buy_smith_median = calculator.simulate_buy_smith_median()
    buy_smith_heavy = calculator.simulate_buy_smith_heavy()

    # Print comparison (only keep key scenarios for report)
    print("SMITH MANEUVER STRESS SCENARIOS:")
    print("=" * 110)
    print("Light (50% LTV):   Conservative - can survive 50% property drop")
    print("Median (65% LTV):  Moderate - can survive 35% property drop")
    print("Heavy (80% LTV):   Aggressive - can survive 20% property drop")
    print("=" * 110)

    # For now, print comparison for Heavy (traditional) scenario
    calculator.print_comparison(
        rent, rent_margin_125, rent_margin_135, buy_traditional, buy_smith_heavy
    )

    # Plot net worth comparison (all 7 scenarios)
    calculator.plot_networth_extended(
        rent,
        rent_margin_125,
        rent_margin_135,
        buy_traditional,
        buy_smith_light,
        buy_smith_median,
        buy_smith_heavy,
    )

    # Export results
    rent.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_rent.csv", index=False
    )
    rent_margin_125.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_rent_margin_125.csv",
        index=False,
    )
    rent_margin_135.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_rent_margin_135.csv",
        index=False,
    )
    buy_traditional.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_buy_traditional.csv",
        index=False,
    )
    buy_smith_light.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_buy_smith_light.csv",
        index=False,
    )
    buy_smith_median.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_buy_smith_median.csv",
        index=False,
    )
    buy_smith_heavy.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_buy_smith_heavy.csv",
        index=False,
    )
    print("\n✓ Detailed results exported to CSV files")


if __name__ == "__main__":
    main()
