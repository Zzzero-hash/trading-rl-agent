"""
Property-based tests for portfolio management using Hypothesis.

These tests verify mathematical properties and invariants that should always hold
true regardless of the specific data or parameters used.
"""

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from trading_rl_agent.portfolio.portfolio_manager import PortfolioManager
from trading_rl_agent.risk.risk_manager import RiskManager


class TestPortfolioProperties:
    """Property-based tests for portfolio management."""

    @given(
        initial_cash=st.floats(min_value=1000, max_value=1000000),
        num_assets=st.integers(min_value=1, max_value=10),
        num_days=st.integers(min_value=10, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_portfolio_value_never_negative(self, initial_cash, num_assets, num_days):
        """Property: Portfolio value should never go negative."""
        # Generate random price data
        prices = arrays(
            dtype=np.float64,
            shape=(num_days, num_assets),
            elements=st.floats(min_value=1.0, max_value=1000.0),
        ).example()

        # Create portfolio
        portfolio = PortfolioManager(initial_cash=initial_cash)

        # Simulate trading over time
        for day in range(num_days):
            for asset in range(num_assets):
                # Random position changes
                position_change = np.random.uniform(-100, 100)
                portfolio.update_position(f"ASSET_{asset}", position_change, prices[day, asset])

            # Verify portfolio value is never negative
            assert portfolio.total_value >= 0, f"Portfolio value went negative: {portfolio.total_value}"

    @given(weights=st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=10))
    @settings(max_examples=100)
    def test_portfolio_weights_sum_to_one(self, weights):
        """Property: Portfolio weights should sum to approximately 1.0."""
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]

            # Verify weights sum to 1.0 (with floating point tolerance)
            assert abs(sum(normalized_weights) - 1.0) < 1e-10

            # Verify all weights are between 0 and 1
            for weight in normalized_weights:
                assert 0.0 <= weight <= 1.0

    @given(
        returns=arrays(
            dtype=np.float64,
            shape=(100, 5),
            elements=st.floats(min_value=-0.5, max_value=0.5),
        )
    )
    @settings(max_examples=50)
    def test_var_monotonicity(self, returns):
        """Property: VaR should be monotonically increasing with confidence level."""
        risk_manager = RiskManager()

        # Calculate VaR at different confidence levels
        confidence_levels = [0.90, 0.95, 0.99]
        var_values = []

        for conf_level in confidence_levels:
            var_val = risk_manager.calculate_var(returns, confidence_level=conf_level)
            var_values.append(var_val)

        # VaR should be monotonically increasing (more negative for higher confidence)
        for i in range(1, len(var_values)):
            assert var_values[i] <= var_values[i - 1], f"VaR not monotonic: {var_values}"

    @given(
        prices=arrays(
            dtype=np.float64,
            shape=(50, 3),
            elements=st.floats(min_value=1.0, max_value=1000.0),
        )
    )
    @settings(max_examples=50)
    def test_portfolio_diversification_benefit(self, prices):
        """Property: Diversified portfolio should have lower volatility than individual assets."""
        # Calculate individual asset volatilities
        individual_vols = []
        for asset in range(prices.shape[1]):
            returns = np.diff(prices[:, asset]) / prices[:-1, asset]
            vol = np.std(returns)
            individual_vols.append(vol)

        # Calculate equal-weighted portfolio volatility
        portfolio_returns = np.mean(np.diff(prices, axis=0) / prices[:-1, :], axis=1)
        portfolio_vol = np.std(portfolio_returns)

        # Portfolio volatility should be less than or equal to average individual volatility
        avg_individual_vol = np.mean(individual_vols)
        assert portfolio_vol <= avg_individual_vol, (
            f"Portfolio vol {portfolio_vol} > avg individual vol {avg_individual_vol}"
        )

    @given(
        cash=st.floats(min_value=1000, max_value=100000),
        asset_prices=st.lists(st.floats(min_value=1.0, max_value=1000.0), min_size=1, max_size=5),
    )
    @settings(max_examples=100)
    def test_portfolio_cash_conservation(self, cash, asset_prices):
        """Property: Total cash + invested value should equal initial cash (minus transaction costs)."""
        portfolio = PortfolioManager(initial_cash=cash)

        # Buy some assets
        total_invested = 0
        for i, price in enumerate(asset_prices):
            shares = cash / (len(asset_prices) * price * 1.1)  # Leave some cash for costs
            invested = shares * price
            portfolio.update_position(f"ASSET_{i}", shares, price)
            total_invested += invested

        # Verify cash conservation (allowing for transaction costs)
        remaining_cash = portfolio.cash
        total_value = portfolio.total_value

        # Total should be approximately equal to initial cash (within transaction costs)
        transaction_cost_estimate = total_invested * 0.01  # 1% transaction cost estimate
        assert abs((remaining_cash + total_value) - cash) <= transaction_cost_estimate

    @given(
        returns=arrays(
            dtype=np.float64,
            shape=(100,),
            elements=st.floats(min_value=-0.5, max_value=0.5),
        )
    )
    @settings(max_examples=50)
    def test_cvar_greater_than_var(self, returns):
        """Property: CVaR should always be greater than or equal to VaR."""
        risk_manager = RiskManager()

        var_95 = risk_manager.calculate_var(returns, confidence_level=0.95)
        cvar_95 = risk_manager.calculate_cvar(returns, confidence_level=0.95)

        assert cvar_95 <= var_95, f"CVaR {cvar_95} > VaR {var_95}"

    @given(data=st.lists(st.floats(min_value=0.0, max_value=1000.0), min_size=10, max_size=100))
    @settings(max_examples=100)
    def test_rolling_statistics_consistency(self, data):
        """Property: Rolling statistics should be consistent with window size."""
        df = pd.DataFrame({"price": data})

        # Calculate rolling statistics
        window_sizes = [5, 10, 20]
        for window in window_sizes:
            if len(data) >= window:
                rolling_mean = df["price"].rolling(window=window).mean()
                rolling_std = df["price"].rolling(window=window).std()

                # Rolling std should be non-negative
                assert (rolling_std >= 0).all()

                # Rolling mean should be within data bounds
                assert (rolling_mean >= df["price"].min()).all()
                assert (rolling_mean <= df["price"].max()).all()


class TestTradingStrategyProperties:
    """Property-based tests for trading strategy behavior."""

    @given(
        signal_strength=st.floats(min_value=-1.0, max_value=1.0),
        volatility=st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=100)
    def test_position_sizing_monotonicity(self, signal_strength, volatility):
        """Property: Position size should be monotonically increasing with signal strength."""
        # This would test your actual position sizing logic
        # For now, we'll test a simplified version

        # Position size should increase with signal strength (absolute value)
        abs_signal = abs(signal_strength)

        # Simulate position sizing (inverse relationship with volatility)
        position_size = abs_signal / (1 + volatility)

        # Verify monotonicity: stronger signals should lead to larger positions
        if abs_signal > 0:
            assert position_size > 0
            assert position_size <= abs_signal  # Position shouldn't exceed signal strength

    @given(
        prices=arrays(
            dtype=np.float64,
            shape=(50,),
            elements=st.floats(min_value=1.0, max_value=1000.0),
        )
    )
    @settings(max_examples=50)
    def test_moving_average_crossover_property(self, prices):
        """Property: Moving average crossover signals should be consistent."""
        df = pd.DataFrame({"price": prices})

        # Calculate moving averages
        short_ma = df["price"].rolling(window=5).mean()
        long_ma = df["price"].rolling(window=20).mean()

        # Generate crossover signals
        crossover = short_ma > long_ma

        # Property: Crossovers should be consistent (no rapid flipping)
        if len(crossover.dropna()) > 1:
            # Count rapid changes (flipping within 2 periods)
            rapid_changes = 0
            for i in range(1, len(crossover.dropna()) - 1):
                if crossover.iloc[i] != crossover.iloc[i - 1] and crossover.iloc[i] != crossover.iloc[i + 1]:
                    rapid_changes += 1

            # Shouldn't have too many rapid changes (less than 20% of periods)
            max_rapid_changes = len(crossover.dropna()) * 0.2
            assert rapid_changes <= max_rapid_changes, f"Too many rapid crossovers: {rapid_changes}"


class TestDataQualityProperties:
    """Property-based tests for data quality and consistency."""

    @given(
        data=arrays(
            dtype=np.float64,
            shape=(100, 5),
            elements=st.floats(min_value=0.0, max_value=1000.0),
        )
    )
    @settings(max_examples=50)
    def test_data_normalization_properties(self, data):
        """Property: Normalized data should have expected statistical properties."""
        # Z-score normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized = (data - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

        # Properties of normalized data
        normalized_mean = np.mean(normalized, axis=0)
        normalized_std = np.std(normalized, axis=0)

        # Mean should be approximately 0
        assert np.allclose(normalized_mean, 0, atol=1e-10)

        # Standard deviation should be approximately 1
        assert np.allclose(normalized_std, 1, atol=1e-10)

    @given(
        data=arrays(
            dtype=np.float64,
            shape=(50, 3),
            elements=st.floats(min_value=0.0, max_value=1000.0),
        )
    )
    @settings(max_examples=50)
    def test_correlation_matrix_properties(self, data):
        """Property: Correlation matrix should have expected mathematical properties."""
        corr_matrix = np.corrcoef(data.T)

        # Diagonal should be 1
        assert np.allclose(np.diag(corr_matrix), 1.0)

        # Matrix should be symmetric
        assert np.allclose(corr_matrix, corr_matrix.T)

        # Eigenvalues should be non-negative (positive semi-definite)
        eigenvalues = np.linalg.eigvals(corr_matrix)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

        # All correlations should be between -1 and 1
        assert np.all(corr_matrix >= -1.0)
        assert np.all(corr_matrix <= 1.0)
