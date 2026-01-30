"""
Smith Maneuver Simulator

Compares house ownership scenarios:
1. Traditional: Standard mortgage payments, no HELOC
2. Smith Maneuver: Use HELOC to invest while paying down mortgage debt

The Smith Maneuver converts non-deductible mortgage debt into tax-deductible
investment loan debt, potentially providing tax benefits.
"""

import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt


# =============================================================================
# SIMULATION ASSUMPTIONS - All parameters centralized for easy modification
# =============================================================================
class SimulationAssumptions:
    """
    Centralized assumptions for Smith Maneuver simulation.
    All values are evidence-based and documented.
    """

    # PROPERTY ASSUMPTIONS
    house_price = 1_000_000  # Starting property value
    down_payment_pct = 0.20  # 20% down payment

    # MORTGAGE ASSUMPTIONS
    # Mortgage rate based on: Canadian neutral policy rate (~2.5%) + mortgage premium (~2%)
    mortgage_rate = 0.042
    mortgage_term_years = 25  # 25-year amortization

    # HELOC ASSUMPTIONS
    # HELOC rate based on: Prime rate (policy rate + 2.5%) + small premium (0.25%)
    heloc_rate = 0.0545

    margin_rate = 0.0445  # Margin account borrowing rate (for comparison)

    # INVESTMENT ASSUMPTIONS
    annual_investment_return = 0.066
    investment_fee_pct = 0.002  # 0.2% MER

    # PROPERTY COST ASSUMPTIONS
    annual_maintenance_pct = 0.018
    # Property tax: Ontario average 0.6-0.8% of property value
    annual_property_tax_pct = 0.01
    # Home insurance: Scales with property value
    # Ontario average: ~$1,500/year for $1M home = 0.15% of value
    annual_home_insurance_pct = 0.002

    # PROPERTY APPRECIATION
    # Long-term Canadian property appreciation: 2-3% annually
    annual_property_appreciation = 0.021

    # TAX ASSUMPTIONS
    # for mid-to-high income earner
    marginal_tax_rate = 0.42

    # SIMULATION PERIOD
    inflation_rate = 0.021  # 2.1% inflation
    years = 25  # 25-year simulation (matches mortgage term)


class SmithManeuverSimulator:
    """Simulates house ownership with and without Smith Maneuver strategy."""

    def __init__(
        self,
        assumptions: SimulationAssumptions = None,
        house_price: float = None,
        down_payment_pct: float = None,
        mortgage_rate: float = None,
        heloc_rate: float = None,
        margin_rate: float = None,
        mortgage_term_years: int = None,
        annual_investment_return: float = None,
        annual_property_appreciation: float = None,
        annual_maintenance_pct: float = None,
        annual_property_tax_pct: float = None,
        annual_home_insurance_pct: float = None,
        investment_fee_pct: float = None,
        marginal_tax_rate: float = None,
        inflation_rate: float = None,
        years: int = None,
        combined_ltv_target: float = None,
    ):
        """
        Initialize simulator with parameters.
        Uses SimulationAssumptions defaults, but allows overrides via parameters.
        """
        # Use provided assumptions object or create from parameters
        if assumptions is None:
            assumptions = SimulationAssumptions()

        # Allow parameter overrides (useful for sensitivity analysis)
        self.house_price = house_price or assumptions.house_price
        self.down_payment_pct = down_payment_pct or assumptions.down_payment_pct
        self.down_payment = self.house_price * self.down_payment_pct
        self.initial_mortgage = self.house_price - self.down_payment

        self.mortgage_rate = mortgage_rate or assumptions.mortgage_rate
        self.heloc_rate = heloc_rate or assumptions.heloc_rate
        self.margin_rate = margin_rate or assumptions.margin_rate
        self.mortgage_term_years = (
            mortgage_term_years or assumptions.mortgage_term_years
        )

        self.annual_investment_return = (
            annual_investment_return or assumptions.annual_investment_return
        )
        self.annual_property_appreciation = (
            annual_property_appreciation or assumptions.annual_property_appreciation
        )

        self.annual_maintenance_pct = (
            annual_maintenance_pct or assumptions.annual_maintenance_pct
        )
        self.annual_property_tax_pct = (
            annual_property_tax_pct or assumptions.annual_property_tax_pct
        )
        self.annual_home_insurance_pct = (
            annual_home_insurance_pct or assumptions.annual_home_insurance_pct
        )

        self.investment_fee_pct = investment_fee_pct or assumptions.investment_fee_pct
        self.marginal_tax_rate = marginal_tax_rate or assumptions.marginal_tax_rate
        self.inflation_rate = inflation_rate or assumptions.inflation_rate
        self.years = years or assumptions.years

        # Combined LTV target for stress scenarios (0.50 = 50%, 0.65 = 65%, 0.80 = 80%)
        # Determines how much leverage is used in Smith Maneuver
        self.combined_ltv_target = (
            combined_ltv_target or 0.80
        )  # Default to traditional 80% LTV

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

    def simulate_traditional(self) -> pd.DataFrame:
        """Simulate traditional mortgage without Smith Maneuver."""
        months = self.mortgage_term_years * 12
        monthly_mortgage_rate = self.mortgage_rate / 12
        monthly_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, months
        )

        results = []
        mortgage_balance = self.initial_mortgage
        house_value = self.house_price
        cash_balance = 0  # No special investment account
        net_worth = self.down_payment

        for year in range(self.years):
            yearly_data = {
                "year": year,
                "house_value": house_value,
                "mortgage_balance": mortgage_balance,
                "heloc_balance": 0,
                "investment_portfolio": 0,
                "cash_balance": cash_balance,
                "gross_net_worth": house_value,
                "net_net_worth": house_value - mortgage_balance,
                "annual_mortgage_payment": 0,
                "annual_interest_paid": 0,
                "annual_property_tax": 0,
                "annual_maintenance": 0,
                "annual_insurance": 0,
                "annual_property_appreciation": 0,
                "annual_investment_gain": 0,
                "annual_tax_benefit": 0,
            }

            # Process 12 months
            for month in range(12):
                # Only make mortgage payments if mortgage balance exists
                if mortgage_balance > 0:
                    interest_payment = mortgage_balance * monthly_mortgage_rate
                    principal_payment = monthly_payment - interest_payment
                    mortgage_balance -= principal_payment
                    mortgage_balance = max(
                        0, mortgage_balance
                    )  # Avoid negative due to rounding
                    yearly_data["annual_interest_paid"] += interest_payment
                    yearly_data["annual_mortgage_payment"] += monthly_payment

            # Annual costs
            yearly_data["annual_property_tax"] = (
                house_value * self.annual_property_tax_pct
            )
            yearly_data["annual_maintenance"] = (
                house_value * self.annual_maintenance_pct
            )
            yearly_data["annual_insurance"] = (
                house_value * self.annual_home_insurance_pct
            )

            total_annual_costs = (
                yearly_data["annual_property_tax"]
                + yearly_data["annual_maintenance"]
                + yearly_data["annual_insurance"]
            )

            # Property appreciation
            appreciation = house_value * self.annual_property_appreciation
            house_value += appreciation
            yearly_data["annual_property_appreciation"] = appreciation

            # Update net worth
            yearly_data["house_value"] = house_value
            yearly_data["mortgage_balance"] = mortgage_balance
            yearly_data["gross_net_worth"] = house_value
            yearly_data["net_net_worth"] = house_value - mortgage_balance

            results.append(yearly_data)

        return pd.DataFrame(results)

    def simulate_smith_maneuver(self) -> pd.DataFrame:
        """Simulate Smith Maneuver strategy with LTV-based leverage control.

        The combined_ltv_target controls the MAXIMUM safe total debt (mortgage + HELOC):
        - 0.50 (Light): 50% LTV cap - conservative, can survive 50% property drop
        - 0.65 (Median): 65% LTV cap - moderate, can survive 35% property drop
        - 0.80 (Heavy): 80% LTV cap - aggressive, traditional lending standard

        KEY INSIGHT: All scenarios start with SAME down payment ($44,980 = 20%)
        and SAME initial mortgage ($179,920 = 80%), but have different safety caps
        on total debt. This caps how much they can borrow via HELOC while applying
        the Smith Maneuver strategy.

        - Light (50% LTV):  Down=$44,980, Mortgage=$179,920, HELOC cap=0 (already over cap)
        - Median (65% LTV): Down=$44,980, Mortgage=$179,920, HELOC cap=$0 (already at/over)
        - Heavy (80% LTV):  Down=$44,980, Mortgage=$179,920, HELOC cap=$0 (at cap)

        The debt cap works by: when total debt (mortgage + HELOC) exceeds the limit,
        extra principal from mortgage paydown isn't redirected to HELOC - instead it
        pays down debt or goes to savings.
        """
        months = self.mortgage_term_years * 12
        monthly_mortgage_rate = self.mortgage_rate / 12
        monthly_heloc_rate = self.heloc_rate / 12

        # All scenarios use standard 80% mortgage and 20% down payment
        standard_down_payment = self.house_price * self.down_payment_pct
        standard_mortgage = self.house_price * (1 - self.down_payment_pct)

        # Maximum total debt cap based on LTV target
        max_total_debt = self.house_price * self.combined_ltv_target

        # Standard mortgage payment (all scenarios same)
        monthly_mortgage_payment = self.calculate_monthly_payment(
            standard_mortgage, self.mortgage_rate, months
        )

        results = []
        mortgage_balance = standard_mortgage
        heloc_balance = 0
        investment_portfolio = 0  # Down payment is home equity, not investment capital
        house_value = self.house_price

        for year in range(self.years):
            yearly_data = {
                "year": year,
                "house_value": house_value,
                "mortgage_balance": mortgage_balance,
                "heloc_balance": heloc_balance,
                "investment_portfolio": investment_portfolio,
                "annual_mortgage_payment": 0,
                "annual_interest_paid_mortgage": 0,
                "annual_interest_paid_heloc": 0,
                "annual_heloc_borrowing": 0,
                "annual_property_tax": 0,
                "annual_maintenance": 0,
                "annual_insurance": 0,
                "annual_property_appreciation": 0,
                "annual_investment_gain": 0,
                "annual_investment_fee": 0,
                "annual_tax_benefit": 0,
            }

            # Process 12 months
            for month in range(12):
                if mortgage_balance > 0:
                    mortgage_interest = mortgage_balance * monthly_mortgage_rate
                    mortgage_principal = monthly_mortgage_payment - mortgage_interest
                    mortgage_balance -= mortgage_principal
                    mortgage_balance = max(0, mortgage_balance)

                    yearly_data["annual_interest_paid_mortgage"] += mortgage_interest
                    yearly_data["annual_mortgage_payment"] += monthly_mortgage_payment

                    # Only redirect principal to HELOC if we can stay under debt cap
                    current_total_debt = mortgage_balance + heloc_balance

                    if current_total_debt <= max_total_debt:
                        # We're under the cap - can borrow via HELOC
                        available_debt_room = max_total_debt - current_total_debt
                        heloc_borrow = min(mortgage_principal, available_debt_room)

                        if heloc_borrow > 0:
                            heloc_balance += heloc_borrow
                            yearly_data["annual_heloc_borrowing"] += heloc_borrow
                            investment_portfolio += heloc_borrow
                    # If over cap, the principal just paid down debt (no HELOC borrow)

                # Monthly investment return
                monthly_return_rate = (1 + self.annual_investment_return) ** (
                    1 / 12
                ) - 1
                investment_gain = investment_portfolio * monthly_return_rate
                investment_portfolio += investment_gain
                yearly_data["annual_investment_gain"] += investment_gain

                # Deduct investment taxes
                investment_tax = investment_gain * self.marginal_tax_rate
                investment_portfolio -= investment_tax

            # Annual costs
            yearly_data["annual_property_tax"] = (
                house_value * self.annual_property_tax_pct
            )
            yearly_data["annual_maintenance"] = (
                house_value * self.annual_maintenance_pct
            )
            yearly_data["annual_insurance"] = (
                house_value * self.annual_home_insurance_pct
            )

            # Investment fees
            yearly_data["annual_investment_fee"] = (
                investment_portfolio * self.investment_fee_pct
            )
            investment_portfolio -= yearly_data["annual_investment_fee"]

            # Tax benefit: HELOC interest is tax-deductible
            heloc_interest_deduction_tax_saving = (
                yearly_data["annual_interest_paid_heloc"] * self.marginal_tax_rate
            )
            yearly_data["annual_tax_benefit"] = heloc_interest_deduction_tax_saving

            # Property appreciation
            appreciation = house_value * self.annual_property_appreciation
            house_value += appreciation
            yearly_data["annual_property_appreciation"] = appreciation

            # Update totals
            yearly_data["house_value"] = house_value
            yearly_data["mortgage_balance"] = mortgage_balance
            yearly_data["heloc_balance"] = heloc_balance
            yearly_data["investment_portfolio"] = investment_portfolio
            yearly_data["gross_net_worth"] = house_value + investment_portfolio
            yearly_data["net_net_worth"] = (
                house_value + investment_portfolio - mortgage_balance - heloc_balance
            )

            results.append(yearly_data)

        return pd.DataFrame(results)

    def simulate_smith_accelerated_with_leverage(self) -> pd.DataFrame:
        """
        Simulate Smith Maneuver with accelerated mortgage (15-year) AND
        continue using freed-up payments for HELOC investing in years 16-25.
        """
        accelerated_months = 15 * 12
        standard_months = self.mortgage_term_years * 12
        monthly_mortgage_rate = self.mortgage_rate / 12
        monthly_heloc_rate = self.heloc_rate / 12

        # Accelerated mortgage payment (15 years)
        accelerated_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, accelerated_months
        )

        # Standard mortgage payment (25 years)
        standard_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, standard_months
        )

        results = []
        mortgage_balance = self.initial_mortgage
        heloc_balance = 0
        investment_portfolio = 0
        house_value = self.house_price
        mortgage_paid_off_year = None

        for year in range(self.years):
            yearly_data = {
                "year": year,
                "house_value": house_value,
                "mortgage_balance": mortgage_balance,
                "heloc_balance": heloc_balance,
                "investment_portfolio": investment_portfolio,
                "annual_mortgage_payment": 0,
                "annual_interest_paid_mortgage": 0,
                "annual_interest_paid_heloc": 0,
                "annual_heloc_borrowing": 0,
                "annual_property_tax": 0,
                "annual_maintenance": 0,
                "annual_insurance": 0,
                "annual_property_appreciation": 0,
                "annual_investment_gain": 0,
                "annual_investment_fee": 0,
                "annual_tax_benefit": 0,
            }

            # Process 12 months
            for month in range(12):
                # Years 0-14: Pay accelerated payment on mortgage
                if mortgage_balance > 0 and (
                    mortgage_paid_off_year is None or year < 15
                ):
                    mortgage_interest = mortgage_balance * monthly_mortgage_rate
                    mortgage_principal = accelerated_payment - mortgage_interest
                    mortgage_balance -= mortgage_principal
                    mortgage_balance = max(0, mortgage_balance)

                    yearly_data["annual_interest_paid_mortgage"] += mortgage_interest
                    yearly_data["annual_mortgage_payment"] += accelerated_payment

                    # Redirect mortgage principal to HELOC for investing
                    heloc_borrow = mortgage_principal
                    heloc_balance += heloc_borrow
                    yearly_data["annual_heloc_borrowing"] += heloc_borrow
                    investment_portfolio += heloc_borrow

                    # Track when mortgage is paid off
                    if mortgage_balance == 0 and mortgage_paid_off_year is None:
                        mortgage_paid_off_year = year

                # Years 15-24: Mortgage paid off, use freed-up payment for HELOC leverage
                elif mortgage_balance == 0 and mortgage_paid_off_year is not None:
                    # Use the excess payment (accelerated - standard) for continued HELOC
                    excess_payment = accelerated_payment - standard_payment
                    heloc_borrow = excess_payment
                    heloc_balance += heloc_borrow
                    yearly_data["annual_heloc_borrowing"] += heloc_borrow
                    investment_portfolio += heloc_borrow

                # Monthly HELOC interest
                heloc_interest = heloc_balance * monthly_heloc_rate
                yearly_data["annual_interest_paid_heloc"] += heloc_interest

                # Monthly investment return
                monthly_return_rate = (1 + self.annual_investment_return) ** (
                    1 / 12
                ) - 1
                investment_gain = investment_portfolio * monthly_return_rate
                investment_portfolio += investment_gain
                yearly_data["annual_investment_gain"] += investment_gain

                # Deduct investment taxes
                investment_tax = investment_gain * self.marginal_tax_rate
                investment_portfolio -= investment_tax

            # Annual costs
            yearly_data["annual_property_tax"] = (
                house_value * self.annual_property_tax_pct
            )
            yearly_data["annual_maintenance"] = (
                house_value * self.annual_maintenance_pct
            )
            yearly_data["annual_insurance"] = (
                house_value * self.annual_home_insurance_pct
            )

            # Investment fees
            yearly_data["annual_investment_fee"] = (
                investment_portfolio * self.investment_fee_pct
            )
            investment_portfolio -= yearly_data["annual_investment_fee"]

            # Tax benefit: HELOC interest is tax-deductible
            heloc_interest_deduction_tax_saving = (
                yearly_data["annual_interest_paid_heloc"] * self.marginal_tax_rate
            )
            yearly_data["annual_tax_benefit"] = heloc_interest_deduction_tax_saving

            # Property appreciation
            appreciation = house_value * self.annual_property_appreciation
            house_value += appreciation
            yearly_data["annual_property_appreciation"] = appreciation

            # Update totals
            yearly_data["house_value"] = house_value
            yearly_data["mortgage_balance"] = mortgage_balance
            yearly_data["heloc_balance"] = heloc_balance
            yearly_data["investment_portfolio"] = investment_portfolio
            yearly_data["gross_net_worth"] = house_value + investment_portfolio
            yearly_data["net_net_worth"] = (
                house_value + investment_portfolio - mortgage_balance - heloc_balance
            )

            results.append(yearly_data)

        return pd.DataFrame(results)

    def simulate_smith_with_nonreg_investment(self) -> pd.DataFrame:
        """
        Simulate Smith Maneuver (standard 25-year) PLUS
        investing the excess amount (that could fund accelerated payments) in non-registered account.
        This compares: HELOC tax-deductible leverage vs regular taxable account investing.
        """
        standard_months = self.mortgage_term_years * 12
        accelerated_months = 15 * 12
        monthly_mortgage_rate = self.mortgage_rate / 12
        monthly_heloc_rate = self.heloc_rate / 12

        standard_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, standard_months
        )
        accelerated_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, accelerated_months
        )
        excess_payment = accelerated_payment - standard_payment

        results = []
        mortgage_balance = self.initial_mortgage
        heloc_balance = 0
        heloc_investment = 0  # HELOC-funded portfolio (tax-deductible)
        nonreg_investment = 0  # Non-reg funded portfolio (taxable)
        house_value = self.house_price

        for year in range(self.years):
            yearly_data = {
                "year": year,
                "house_value": house_value,
                "mortgage_balance": mortgage_balance,
                "heloc_balance": heloc_balance,
                "heloc_investment": heloc_investment,
                "nonreg_investment": nonreg_investment,
                "total_investment": heloc_investment + nonreg_investment,
                "annual_mortgage_payment": 0,
                "annual_interest_paid_mortgage": 0,
                "annual_interest_paid_heloc": 0,
                "annual_heloc_borrowing": 0,
                "annual_nonreg_contributed": 0,
                "annual_property_tax": 0,
                "annual_maintenance": 0,
                "annual_insurance": 0,
                "annual_property_appreciation": 0,
                "annual_tax_benefit": 0,
            }

            # Process 12 months
            for month in range(12):
                # Standard mortgage payment (25-year)
                if mortgage_balance > 0:
                    mortgage_interest = mortgage_balance * monthly_mortgage_rate
                    mortgage_principal = standard_payment - mortgage_interest
                    mortgage_balance -= mortgage_principal
                    mortgage_balance = max(0, mortgage_balance)

                    yearly_data["annual_interest_paid_mortgage"] += mortgage_interest
                    yearly_data["annual_mortgage_payment"] += standard_payment

                    # Redirect mortgage principal to HELOC for investing (Case 2 behavior)
                    heloc_borrow = mortgage_principal
                    heloc_balance += heloc_borrow
                    yearly_data["annual_heloc_borrowing"] += heloc_borrow
                    heloc_investment += heloc_borrow

                # Also invest the excess amount in non-registered account
                # (money that could be used for accelerated payments)
                nonreg_contribution = excess_payment / 12
                nonreg_investment += nonreg_contribution
                yearly_data["annual_nonreg_contributed"] += nonreg_contribution

                # Monthly HELOC interest (only on HELOC balance)
                heloc_interest = heloc_balance * monthly_heloc_rate
                yearly_data["annual_interest_paid_heloc"] += heloc_interest

                # Monthly investment returns for both portfolios
                monthly_return_rate = (1 + self.annual_investment_return) ** (
                    1 / 12
                ) - 1

                # HELOC portfolio: gains are taxed
                heloc_gain = heloc_investment * monthly_return_rate
                heloc_investment += heloc_gain
                heloc_tax = heloc_gain * self.marginal_tax_rate
                heloc_investment -= heloc_tax

                # Non-reg portfolio: gains are also taxed (no deduction for HELOC interest here)
                nonreg_gain = nonreg_investment * monthly_return_rate
                nonreg_investment += nonreg_gain
                nonreg_tax = nonreg_gain * self.marginal_tax_rate
                nonreg_investment -= nonreg_tax

            # Annual costs
            yearly_data["annual_property_tax"] = (
                house_value * self.annual_property_tax_pct
            )
            yearly_data["annual_maintenance"] = (
                house_value * self.annual_maintenance_pct
            )
            yearly_data["annual_insurance"] = (
                house_value * self.annual_home_insurance_pct
            )

            # Investment fees on both portfolios
            heloc_fee = heloc_investment * self.investment_fee_pct
            heloc_investment -= heloc_fee
            nonreg_fee = nonreg_investment * self.investment_fee_pct
            nonreg_investment -= nonreg_fee

            # Tax benefit: HELOC interest is tax-deductible
            heloc_interest_deduction_tax_saving = (
                yearly_data["annual_interest_paid_heloc"] * self.marginal_tax_rate
            )
            yearly_data["annual_tax_benefit"] = heloc_interest_deduction_tax_saving

            # Property appreciation
            appreciation = house_value * self.annual_property_appreciation
            house_value += appreciation
            yearly_data["annual_property_appreciation"] = appreciation

            # Update totals
            yearly_data["house_value"] = house_value
            yearly_data["mortgage_balance"] = mortgage_balance
            yearly_data["heloc_balance"] = heloc_balance
            yearly_data["heloc_investment"] = heloc_investment
            yearly_data["nonreg_investment"] = nonreg_investment
            yearly_data["total_investment"] = heloc_investment + nonreg_investment
            yearly_data["gross_net_worth"] = (
                house_value + heloc_investment + nonreg_investment
            )
            yearly_data["net_net_worth"] = (
                house_value
                + heloc_investment
                + nonreg_investment
                - mortgage_balance
                - heloc_balance
            )

            results.append(yearly_data)

        return pd.DataFrame(results)

    def simulate_smith_accelerated(self) -> pd.DataFrame:
        """
        Simulate Smith Maneuver with Accelerated Mortgage Payoff.

        Strategy: Pay down mortgage on aggressive schedule (e.g., 15-year amortization),
        but use HELOC to invest the "excess" cash that would go to accelerated principal.
        This maximizes investment growth while still paying down debt faster.
        """
        # Use accelerated amortization (15 years instead of 25)
        accelerated_months = 15 * 12
        standard_months = self.mortgage_term_years * 12

        monthly_mortgage_rate = self.mortgage_rate / 12
        monthly_heloc_rate = self.heloc_rate / 12

        # Calculate both payments for comparison
        accelerated_monthly_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, accelerated_months
        )
        standard_monthly_payment = self.calculate_monthly_payment(
            self.initial_mortgage, self.mortgage_rate, standard_months
        )

        # The "excess" is the difference between accelerated and standard payment
        excess_payment = accelerated_monthly_payment - standard_monthly_payment

        results = []
        mortgage_balance = self.initial_mortgage
        heloc_balance = 0
        investment_portfolio = 0
        house_value = self.house_price

        for year in range(self.years):
            yearly_data = {
                "year": year,
                "house_value": house_value,
                "mortgage_balance": mortgage_balance,
                "heloc_balance": heloc_balance,
                "investment_portfolio": investment_portfolio,
                "annual_mortgage_payment": 0,
                "annual_accelerated_payment": 0,
                "annual_excess_invested": 0,
                "annual_interest_paid_mortgage": 0,
                "annual_interest_paid_heloc": 0,
                "annual_heloc_borrowing": 0,
                "annual_property_tax": 0,
                "annual_maintenance": 0,
                "annual_insurance": 0,
                "annual_property_appreciation": 0,
                "annual_investment_gain": 0,
                "annual_investment_fee": 0,
                "annual_tax_benefit": 0,
            }

            # Process 12 months
            for month in range(12):
                # Only process mortgage if balance exists
                if mortgage_balance > 0:
                    mortgage_interest = mortgage_balance * monthly_mortgage_rate
                    mortgage_principal = accelerated_monthly_payment - mortgage_interest
                    mortgage_balance -= mortgage_principal
                    mortgage_balance = max(0, mortgage_balance)

                    yearly_data["annual_interest_paid_mortgage"] += mortgage_interest
                    yearly_data[
                        "annual_mortgage_payment"
                    ] += accelerated_monthly_payment
                    yearly_data["annual_accelerated_payment"] += excess_payment

                    # Use HELOC to invest the excess payment amount
                    if excess_payment > 0:
                        heloc_balance += excess_payment
                        yearly_data["annual_heloc_borrowing"] += excess_payment
                        investment_portfolio += excess_payment

                # Monthly HELOC interest (continues after mortgage is paid)
                heloc_interest = heloc_balance * monthly_heloc_rate
                yearly_data["annual_interest_paid_heloc"] += heloc_interest

                # Monthly investment return
                monthly_return_rate = (1 + self.annual_investment_return) ** (
                    1 / 12
                ) - 1
                investment_gain = investment_portfolio * monthly_return_rate
                investment_portfolio += investment_gain
                yearly_data["annual_investment_gain"] += investment_gain

                # Deduct investment taxes
                investment_tax = investment_gain * self.marginal_tax_rate
                investment_portfolio -= investment_tax

                yearly_data["annual_excess_invested"] += excess_payment

            # Annual costs
            yearly_data["annual_property_tax"] = (
                house_value * self.annual_property_tax_pct
            )
            yearly_data["annual_maintenance"] = (
                house_value * self.annual_maintenance_pct
            )
            yearly_data["annual_insurance"] = (
                house_value * self.annual_home_insurance_pct
            )

            # Investment fees
            yearly_data["annual_investment_fee"] = (
                investment_portfolio * self.investment_fee_pct
            )
            investment_portfolio -= yearly_data["annual_investment_fee"]

            # Tax benefit: HELOC interest is tax-deductible
            heloc_interest_deduction_tax_saving = (
                yearly_data["annual_interest_paid_heloc"] * self.marginal_tax_rate
            )
            yearly_data["annual_tax_benefit"] = heloc_interest_deduction_tax_saving

            # Property appreciation
            appreciation = house_value * self.annual_property_appreciation
            house_value += appreciation
            yearly_data["annual_property_appreciation"] = appreciation

            # Update totals
            yearly_data["house_value"] = house_value
            yearly_data["mortgage_balance"] = mortgage_balance
            yearly_data["heloc_balance"] = heloc_balance
            yearly_data["investment_portfolio"] = investment_portfolio
            yearly_data["gross_net_worth"] = house_value + investment_portfolio
            yearly_data["net_net_worth"] = (
                house_value + investment_portfolio - mortgage_balance - heloc_balance
            )

            results.append(yearly_data)

        return pd.DataFrame(results)

    def run_simulation(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run all three scenarios and return results."""
        traditional = self.simulate_traditional()
        smith = self.simulate_smith_maneuver()
        smith_accelerated = self.simulate_smith_accelerated()
        return traditional, smith, smith_accelerated

    def print_summary(
        self,
        traditional: pd.DataFrame,
        smith: pd.DataFrame,
        smith_accelerated: pd.DataFrame = None,
    ) -> None:
        """Print summary comparison of scenarios."""
        final_year_trad = traditional.iloc[-1]
        final_year_smith = smith.iloc[-1]
        final_year_smith_accel = (
            smith_accelerated.iloc[-1] if smith_accelerated is not None else None
        )

        print("=" * 80)
        print("SMITH MANEUVER SIMULATION SUMMARY - ALL SCENARIOS")
        print("=" * 80)
        print(f"\nInitial Assumptions:")
        print(f"  House Price: ${self.house_price:,.0f}")
        print(
            f"  Down Payment: ${self.down_payment:,.0f} ({self.down_payment_pct*100:.0f}%)"
        )
        print(f"  Initial Mortgage: ${self.initial_mortgage:,.0f}")
        print(f"  Mortgage Rate: {self.mortgage_rate*100:.2f}%")
        print(f"  HELOC Rate: {self.heloc_rate*100:.2f}%")
        print(f"  Mortgage Term: {self.mortgage_term_years} years")
        print(f"  Investment Return: {self.annual_investment_return*100:.2f}%")
        print(f"  Property Appreciation: {self.annual_property_appreciation*100:.2f}%")
        print(f"  Marginal Tax Rate: {self.marginal_tax_rate*100:.0f}%")
        print(f"  Simulation Period: {self.years} years")

        print("\n" + "=" * 80)
        print("SCENARIO 1: TRADITIONAL (Standard 25-year Mortgage)")
        print("=" * 80)
        print(f"Final House Value: ${final_year_trad['house_value']:,.0f}")
        print(f"Final Mortgage Balance: ${final_year_trad['mortgage_balance']:,.0f}")
        print(
            f"Total Mortgage Paid: ${traditional['annual_mortgage_payment'].sum():,.0f}"
        )
        print(f"Total Interest Paid: ${traditional['annual_interest_paid'].sum():,.0f}")
        print(f"Total Property Tax: ${traditional['annual_property_tax'].sum():,.0f}")
        print(f"Total Maintenance: ${traditional['annual_maintenance'].sum():,.0f}")
        print(f"Total Insurance: ${traditional['annual_insurance'].sum():,.0f}")
        print(
            f"Total Home Costs: ${(traditional['annual_property_tax'].sum() + traditional['annual_maintenance'].sum() + traditional['annual_insurance'].sum()):,.0f}"
        )
        print(f"Gross Net Worth: ${final_year_trad['gross_net_worth']:,.0f}")
        print(
            f"Net Net Worth (Gross - Mortgage): ${final_year_trad['net_net_worth']:,.0f}"
        )

        print("\n" + "=" * 80)
        print(
            "SCENARIO 2: SMITH MANEUVER (Standard 25-year Mortgage + HELOC Investing)"
        )
        print("=" * 80)
        print(f"Final House Value: ${final_year_smith['house_value']:,.0f}")
        print(f"Final Mortgage Balance: ${final_year_smith['mortgage_balance']:,.0f}")
        print(f"Final HELOC Balance: ${final_year_smith['heloc_balance']:,.0f}")
        print(
            f"Final Investment Portfolio: ${final_year_smith['investment_portfolio']:,.0f}"
        )
        print(f"Total Mortgage Paid: ${smith['annual_mortgage_payment'].sum():,.0f}")
        print(
            f"Total Mortgage Interest: ${smith['annual_interest_paid_mortgage'].sum():,.0f}"
        )
        print(
            f"Total HELOC Interest: ${smith['annual_interest_paid_heloc'].sum():,.0f}"
        )
        print(f"Total HELOC Borrowed: ${smith['annual_heloc_borrowing'].sum():,.0f}")
        print(f"Total Property Tax: ${smith['annual_property_tax'].sum():,.0f}")
        print(f"Total Maintenance: ${smith['annual_maintenance'].sum():,.0f}")
        print(f"Total Insurance: ${smith['annual_insurance'].sum():,.0f}")
        print(f"Total Investment Fees: ${smith['annual_investment_fee'].sum():,.0f}")
        print(f"Total Tax Benefits: ${smith['annual_tax_benefit'].sum():,.0f}")
        print(f"Gross Net Worth: ${final_year_smith['gross_net_worth']:,.0f}")
        print(
            f"Net Net Worth (Gross - Mortgage - HELOC): ${final_year_smith['net_net_worth']:,.0f}"
        )

        if smith_accelerated is not None:
            print("\n" + "=" * 80)
            print(
                "SCENARIO 3: SMITH MANEUVER ACCELERATED (15-year Mortgage + HELOC Investing)"
            )
            print("=" * 80)
            print(f"Final House Value: ${final_year_smith_accel['house_value']:,.0f}")
            print(
                f"Final Mortgage Balance: ${final_year_smith_accel['mortgage_balance']:,.0f}"
            )
            print(
                f"Final HELOC Balance: ${final_year_smith_accel['heloc_balance']:,.0f}"
            )
            print(
                f"Final Investment Portfolio: ${final_year_smith_accel['investment_portfolio']:,.0f}"
            )
            print(
                f"Total Mortgage Paid: ${smith_accelerated['annual_mortgage_payment'].sum():,.0f}"
            )
            print(
                f"Total Mortgage Interest: ${smith_accelerated['annual_interest_paid_mortgage'].sum():,.0f}"
            )
            print(
                f"Total HELOC Interest: ${smith_accelerated['annual_interest_paid_heloc'].sum():,.0f}"
            )
            print(
                f"Total HELOC Borrowed: ${smith_accelerated['annual_heloc_borrowing'].sum():,.0f}"
            )
            print(
                f"Total Accelerated Principal: ${smith_accelerated['annual_accelerated_payment'].sum():,.0f}"
            )
            print(
                f"Total Property Tax: ${smith_accelerated['annual_property_tax'].sum():,.0f}"
            )
            print(
                f"Total Maintenance: ${smith_accelerated['annual_maintenance'].sum():,.0f}"
            )
            print(
                f"Total Insurance: ${smith_accelerated['annual_insurance'].sum():,.0f}"
            )
            print(
                f"Total Investment Fees: ${smith_accelerated['annual_investment_fee'].sum():,.0f}"
            )
            print(
                f"Total Tax Benefits: ${smith_accelerated['annual_tax_benefit'].sum():,.0f}"
            )
            print(f"Gross Net Worth: ${final_year_smith_accel['gross_net_worth']:,.0f}")
            print(
                f"Net Net Worth (Gross - Mortgage - HELOC): ${final_year_smith_accel['net_net_worth']:,.0f}"
            )

        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        difference_smith_net = (
            final_year_smith["net_net_worth"] - final_year_trad["net_net_worth"]
        )
        print(
            f"Smith vs Traditional: ${difference_smith_net:,.0f} ({(difference_smith_net/final_year_trad['net_net_worth']*100):.2f}%)"
        )
        print(
            f"  Smith Maneuver is {'BETTER' if difference_smith_net > 0 else 'WORSE'} by ${abs(difference_smith_net):,.0f}"
        )

        if smith_accelerated is not None:
            difference_accel_net = (
                final_year_smith_accel["net_net_worth"]
                - final_year_trad["net_net_worth"]
            )
            difference_accel_vs_smith = (
                final_year_smith_accel["net_net_worth"]
                - final_year_smith["net_net_worth"]
            )
            print(
                f"\nSmith Accelerated vs Traditional: ${difference_accel_net:,.0f} ({(difference_accel_net/final_year_trad['net_net_worth']*100):.2f}%)"
            )
            print(
                f"  Smith Accelerated is {'BETTER' if difference_accel_net > 0 else 'WORSE'} by ${abs(difference_accel_net):,.0f}"
            )

            print(
                f"\nSmith Accelerated vs Smith Standard: ${difference_accel_vs_smith:,.0f}"
            )
            print(
                f"  Accelerated is {'BETTER' if difference_accel_vs_smith > 0 else 'WORSE'} by ${abs(difference_accel_vs_smith):,.0f}"
            )

    def plot_comparison(self, traditional: pd.DataFrame, smith: pd.DataFrame) -> None:
        """Create visualization comparing both scenarios."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Net Net Worth Over Time
        ax = axes[0, 0]
        ax.plot(
            traditional["year"],
            traditional["net_net_worth"],
            label="Traditional",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            smith["year"],
            smith["net_net_worth"],
            label="Smith Maneuver",
            marker="s",
            linewidth=2,
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Net Net Worth ($)")
        ax.set_title("Net Worth Comparison Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        # Plot 2: Mortgage vs HELOC Balance
        ax = axes[0, 1]
        ax.plot(
            traditional["year"],
            traditional["mortgage_balance"],
            label="Traditional Mortgage",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            smith["year"],
            smith["mortgage_balance"],
            label="Smith - Mortgage",
            marker="s",
            linewidth=2,
        )
        ax.plot(
            smith["year"],
            smith["heloc_balance"],
            label="Smith - HELOC",
            marker="^",
            linewidth=2,
            linestyle="--",
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Balance ($)")
        ax.set_title("Debt Balance Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        # Plot 3: Investment Portfolio Growth (Smith Only)
        ax = axes[1, 0]
        ax.bar(
            smith["year"],
            smith["investment_portfolio"],
            label="Investment Portfolio",
            color="green",
            alpha=0.7,
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_title("Investment Portfolio Growth (Smith Maneuver)")
        ax.grid(True, alpha=0.3, axis="y")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        # Plot 4: Cumulative Costs
        ax = axes[1, 1]
        trad_costs = (
            traditional[
                [
                    "annual_interest_paid",
                    "annual_property_tax",
                    "annual_maintenance",
                ]
            ]
            .sum(axis=1)
            .cumsum()
        )
        smith_costs = (
            smith["annual_interest_paid_mortgage"]
            + smith["annual_interest_paid_heloc"]
            + smith["annual_property_tax"]
            + smith["annual_maintenance"]
        ).cumsum()

        ax.plot(
            traditional["year"],
            trad_costs,
            label="Traditional",
            marker="o",
            linewidth=2,
        )
        ax.plot(
            smith["year"], smith_costs, label="Smith Maneuver", marker="s", linewidth=2
        )
        ax.set_xlabel("Year")
        ax.set_ylabel("Cumulative Costs ($)")
        ax.set_title("Cumulative Costs Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x/1e6:.1f}M"))

        plt.tight_layout()
        plt.savefig(
            "/Users/leohong/backtesting/home-ownership/smith_maneuver_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("\n✓ Plot saved to: smith_maneuver_comparison.png")
        plt.show()


def main():
    """Run the Smith Maneuver simulation with 4 scenarios over 25 years."""
    simulator = SmithManeuverSimulator()

    print("\nRunning Smith Maneuver simulation - 4 scenarios (25 years)...\n")

    # Run all 4 scenarios
    traditional = simulator.simulate_traditional()
    smith_standard = simulator.simulate_smith_maneuver()
    smith_accelerated = simulator.simulate_smith_accelerated_with_leverage()
    smith_nonreg = simulator.simulate_smith_with_nonreg_investment()

    # Get final year data for all scenarios
    trad_final = traditional.iloc[-1]
    smith_final = smith_standard.iloc[-1]
    accel_final = smith_accelerated.iloc[-1]
    nonreg_final = smith_nonreg.iloc[-1]

    # Print summary
    print("=" * 100)
    print("SMITH MANEUVER COMPARISON - 4 SCENARIOS (25 YEARS)")
    print("=" * 100)
    print(f"\nInitial Assumptions:")
    print(f"  House Price: ${simulator.house_price:,.0f}")
    print(
        f"  Down Payment: ${simulator.down_payment:,.0f} ({simulator.down_payment_pct*100:.0f}%)"
    )
    print(f"  Initial Mortgage: ${simulator.initial_mortgage:,.0f}")
    print(f"  Mortgage Rate: {simulator.mortgage_rate*100:.2f}%")
    print(f"  HELOC Rate: {simulator.heloc_rate*100:.2f}%")
    print(f"  Investment Return: {simulator.annual_investment_return*100:.2f}%")
    print(f"  Marginal Tax Rate: {simulator.marginal_tax_rate*100:.0f}%")

    # Scenario 1: Traditional
    print("\n" + "=" * 100)
    print("SCENARIO 1: TRADITIONAL (Standard 25-year mortgage)")
    print("=" * 100)
    print(f"Final House Value: ${trad_final['house_value']:,.0f}")
    print(f"Final Mortgage Balance: ${trad_final['mortgage_balance']:,.0f}")
    print(f"Net Net Worth: ${trad_final['net_net_worth']:,.0f}")

    # Scenario 2: Smith Standard
    print("\n" + "=" * 100)
    print("SCENARIO 2: SMITH STANDARD (25-year mortgage + HELOC investing)")
    print("=" * 100)
    print(f"Final House Value: ${smith_final['house_value']:,.0f}")
    print(f"Final Mortgage Balance: ${smith_final['mortgage_balance']:,.0f}")
    print(f"Final HELOC Balance: ${smith_final['heloc_balance']:,.0f}")
    print(f"Final Investment Portfolio: ${smith_final['investment_portfolio']:,.0f}")
    print(f"Gross Net Worth: ${smith_final['gross_net_worth']:,.0f}")
    print(f"Net Net Worth (after debt): ${smith_final['net_net_worth']:,.0f}")
    smith_advantage = smith_final["net_net_worth"] - trad_final["net_net_worth"]
    print(
        f"Advantage vs Traditional: +${smith_advantage:,.0f} ({smith_advantage/trad_final['net_net_worth']*100:.2f}%)"
    )

    # Scenario 3: Smith Accelerated with Leverage
    print("\n" + "=" * 100)
    print(
        "SCENARIO 3: SMITH ACCELERATED + LEVERAGE (15-year mortgage + continued HELOC investing)"
    )
    print("=" * 100)
    print(f"Final House Value: ${accel_final['house_value']:,.0f}")
    print(f"Final Mortgage Balance: ${accel_final['mortgage_balance']:,.0f}")
    print(f"Final HELOC Balance: ${accel_final['heloc_balance']:,.0f}")
    print(f"Final Investment Portfolio: ${accel_final['investment_portfolio']:,.0f}")
    print(f"Gross Net Worth: ${accel_final['gross_net_worth']:,.0f}")
    print(f"Net Net Worth (after debt): ${accel_final['net_net_worth']:,.0f}")
    accel_advantage = accel_final["net_net_worth"] - trad_final["net_net_worth"]
    accel_vs_smith = accel_final["net_net_worth"] - smith_final["net_net_worth"]
    print(
        f"Advantage vs Traditional: +${accel_advantage:,.0f} ({accel_advantage/trad_final['net_net_worth']*100:.2f}%)"
    )
    print(
        f"vs Smith Standard: {'+' if accel_vs_smith > 0 else ''} ${accel_vs_smith:,.0f}"
    )

    # Scenario 4: Smith + Non-Reg
    print("\n" + "=" * 100)
    print(
        "SCENARIO 4: SMITH + NON-REG (Standard mortgage + HELOC + taxable non-reg investing)"
    )
    print("=" * 100)
    print(f"Final House Value: ${nonreg_final['house_value']:,.0f}")
    print(f"Final Mortgage Balance: ${nonreg_final['mortgage_balance']:,.0f}")
    print(f"Final HELOC Balance: ${nonreg_final['heloc_balance']:,.0f}")
    print(f"Final HELOC Investment: ${nonreg_final['heloc_investment']:,.0f}")
    print(f"Final Non-Reg Investment: ${nonreg_final['nonreg_investment']:,.0f}")
    print(f"Total Investment: ${nonreg_final['total_investment']:,.0f}")
    print(f"Gross Net Worth: ${nonreg_final['gross_net_worth']:,.0f}")
    print(f"Net Net Worth (after debt): ${nonreg_final['net_net_worth']:,.0f}")
    nonreg_advantage = nonreg_final["net_net_worth"] - trad_final["net_net_worth"]
    nonreg_vs_smith = nonreg_final["net_net_worth"] - smith_final["net_net_worth"]
    print(
        f"Advantage vs Traditional: +${nonreg_advantage:,.0f} ({nonreg_advantage/trad_final['net_net_worth']*100:.2f}%)"
    )
    print(f"vs Smith Standard: +${nonreg_vs_smith:,.0f}")

    # Comparison Summary
    print("\n" + "=" * 100)
    print("SUMMARY COMPARISON")
    print("=" * 100)
    print(f"\n{'Scenario':<45} {'Net Worth':<20} {'Advantage':<20}")
    print("-" * 85)
    print(
        f"{'1. Traditional (25yr std mortgage)':<45} ${trad_final['net_net_worth']:>18,.0f} {'Baseline':<20}"
    )
    print(
        f"{'2. Smith Standard (HELOC investing)':<45} ${smith_final['net_net_worth']:>18,.0f} +${smith_advantage:>17,.0f}"
    )
    print(
        f"{'3. Smith Accel + Leverage (15yr + HELOC)':<45} ${accel_final['net_net_worth']:>18,.0f} +${accel_advantage:>17,.0f}"
    )
    print(
        f"{'4. Smith + Non-Reg (std + non-reg invest)':<45} ${nonreg_final['net_net_worth']:>18,.0f} +${nonreg_advantage:>17,.0f}"
    )

    # Key Comparisons
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    print(f"\n1. HELOC Tax Deductibility Impact:")
    print(f"   Smith Standard vs Traditional: +${smith_advantage:,.0f}")
    print(f"   (Tax-deductible HELOC debt benefits)")

    print(f"\n2. Accelerated Payoff + Continued Leverage:")
    print(
        f"   Smith Accelerated vs Smith Standard: {'+' if accel_vs_smith > 0 else ''} ${accel_vs_smith:,.0f}"
    )
    print(
        f"   (Mortgage paid in 15 years vs 25, but using excess for continued HELOC investing)"
    )

    print(f"\n3. Non-Registered Account Alternative:")
    print(f"   Smith + Non-Reg vs Smith Standard: +${nonreg_vs_smith:,.0f}")
    print(f"   (Investing excess in taxable account adds extra capital)")

    print(f"\n4. Best Case (Smith + Non-Reg):")
    print(
        f"   Total advantage over Traditional: +${nonreg_advantage:,.0f} ({nonreg_advantage/trad_final['net_net_worth']*100:.1f}%)"
    )

    # Export results
    traditional.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_1_traditional.csv",
        index=False,
    )
    smith_standard.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_2_smith_standard.csv",
        index=False,
    )
    smith_accelerated.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_3_smith_accelerated.csv",
        index=False,
    )
    smith_nonreg.to_csv(
        "/Users/leohong/backtesting/home-ownership/scenario_4_smith_nonreg.csv",
        index=False,
    )
    print("\n✓ Detailed results exported to CSV files")


if __name__ == "__main__":
    main()
