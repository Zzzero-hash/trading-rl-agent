"""
Integration tests for risk-portfolio interactions.

Tests cover:
- Risk-aware portfolio optimization
- Real-time risk monitoring integration
- Regulatory compliance workflows
- End-to-end trading scenarios
- Performance attribution with risk metrics
- Stress testing with portfolio management
"""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from src.trade_agent.monitoring.alert_manager import AlertManager
from src.trade_agent.portfolio.attribution import PerformanceAttributor
from src.trade_agent.portfolio.manager import PortfolioConfig, PortfolioManager
from src.trade_agent.risk.alert_system import RiskAlertConfig, RiskAlertSystem
from src.trade_agent.risk.manager import RiskLimits, RiskManager


class TestRiskAwarePortfolioOptimization:
    """Test risk-aware portfolio optimization workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        # Risk management setup
        self.risk_limits = RiskLimits(
            max_portfolio_var=0.03,
            max_drawdown=0.15,
            max_leverage=1.5,
            max_position_size=0.2,
        )
        self.risk_manager = RiskManager(self.risk_limits)

        # Portfolio management setup
        self.portfolio_config = PortfolioConfig(
            max_position_size=0.2,
            max_sector_exposure=0.4,
            max_leverage=1.5,
            rebalance_threshold=0.05,
        )
        self.portfolio_manager = PortfolioManager(initial_capital=1000000, config=self.portfolio_config)

        # Create test market data
        self._create_test_market_data()

    def _create_test_market_data(self):
        """Create realistic test market data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate correlated returns for multiple assets
        base_returns = np.random.normal(0.001, 0.02, 252)

        returns_data = {}
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "AMZN"]):
            # Create correlated returns
            correlation = 0.3 + 0.4 * (i / 5)  # Varying correlation
            asset_returns = base_returns * correlation + np.random.normal(0.001, 0.015, 252)
            returns_data[symbol] = pd.Series(asset_returns, index=dates)

        # Update risk manager with returns data
        self.risk_manager.update_returns_data(returns_data)

        # Create price data for portfolio optimization
        self.price_data = {}
        for symbol, returns in returns_data.items():
            prices = pd.DataFrame(
                {
                    "close": 100 * (1 + returns).cumprod(),
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            self.price_data[symbol] = prices

    def test_risk_constrained_optimization(self):
        """Test portfolio optimization with risk constraints."""
        target_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        # Optimize portfolio with risk constraints
        optimal_weights = self.portfolio_manager.optimize_portfolio(
            target_symbols, self.price_data, method="max_sharpe"
        )

        if optimal_weights is not None:
            # Calculate risk metrics for optimal portfolio
            portfolio_var = self.risk_manager.calculate_portfolio_var(optimal_weights)
            self.risk_manager.calculate_portfolio_cvar(optimal_weights)
            max_drawdown = self.risk_manager.calculate_portfolio_drawdown(optimal_weights)

            # Verify risk constraints are satisfied
            assert portfolio_var <= self.risk_limits.max_portfolio_var
            assert max_drawdown <= self.risk_limits.max_drawdown

            # Verify position size constraints
            for weight in optimal_weights.values():
                assert weight <= self.portfolio_config.max_position_size

    def test_risk_parity_optimization(self):
        """Test risk parity optimization with risk management."""
        target_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        # Optimize using risk parity
        optimal_weights = self.portfolio_manager.optimize_portfolio(
            target_symbols, self.price_data, method="risk_parity"
        )

        if optimal_weights is not None:
            # Calculate risk contribution for each asset
            risk_contributions = {}
            total_risk = 0

            for asset, weight in optimal_weights.items():
                # Calculate individual asset risk contribution
                asset_risk = self.risk_manager._returns_data[asset].std()
                risk_contribution = weight * asset_risk
                risk_contributions[asset] = risk_contribution
                total_risk += risk_contribution

            # In risk parity, risk contributions should be approximately equal
            avg_risk_contribution = total_risk / len(optimal_weights)
            for risk_contribution in risk_contributions.values():
                assert abs(risk_contribution - avg_risk_contribution) / avg_risk_contribution < 0.3

    def test_dynamic_risk_adjustment(self):
        """Test dynamic risk adjustment based on market conditions."""
        # Initial optimization
        target_symbols = ["AAPL", "GOOGL", "MSFT"]
        initial_weights = self.portfolio_manager.optimize_portfolio(
            target_symbols, self.price_data, method="max_sharpe"
        )

        if initial_weights is not None:
            initial_var = self.risk_manager.calculate_portfolio_var(initial_weights)

            # Simulate market stress (increased volatility)
            stressed_returns = {}
            for symbol, returns in self.risk_manager._returns_data.items():
                stressed_returns[symbol] = returns * 2.0  # Double volatility

            self.risk_manager.update_returns_data(stressed_returns)

            # Re-optimize under stress conditions
            stressed_weights = self.portfolio_manager.optimize_portfolio(
                target_symbols, self.price_data, method="min_variance"
            )

            if stressed_weights is not None:
                stressed_var = self.risk_manager.calculate_portfolio_var(stressed_weights)

                # Portfolio should be more conservative under stress
                assert stressed_var <= initial_var * 1.5  # Allow some increase but not excessive

    def test_risk_budget_allocation(self):
        """Test risk budget allocation across assets."""
        target_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]

        # Define risk budget (total portfolio risk)
        total_risk_budget = 0.02  # 2% total portfolio risk

        # Allocate risk budget across assets
        risk_allocations = {}
        remaining_budget = total_risk_budget

        for i, symbol in enumerate(target_symbols):
            # Allocate risk proportionally (can be customized)
            if i == len(target_symbols) - 1:
                risk_allocations[symbol] = remaining_budget
            else:
                allocation = total_risk_budget / len(target_symbols)
                risk_allocations[symbol] = allocation
                remaining_budget -= allocation

        # Calculate weights based on risk allocation
        weights = {}
        for symbol, risk_allocation in risk_allocations.items():
            asset_volatility = self.risk_manager._returns_data[symbol].std()
            weights[symbol] = risk_allocation / asset_volatility

        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Verify risk budget is respected
        portfolio_var = self.risk_manager.calculate_portfolio_var(weights)
        assert abs(portfolio_var - total_risk_budget) < 0.005


class TestRealTimeRiskPortfolioIntegration:
    """Test real-time integration between risk and portfolio management."""

    def setup_method(self):
        """Setup test fixtures."""
        # Risk management setup
        self.risk_manager = RiskManager()

        # Portfolio management setup
        self.portfolio_manager = PortfolioManager(initial_capital=1000000)

        # Alert system setup
        self.alert_manager = AlertManager()
        self.alert_config = RiskAlertConfig(
            monitoring_interval_seconds=1,
            real_time_monitoring=True,
            alert_thresholds=[
                {
                    "metric_name": "portfolio_var",
                    "threshold_type": "max",
                    "threshold_value": 0.03,
                    "severity": "warning",
                    "escalation_level": "level_2",
                    "enabled": True,
                }
            ],
        )

        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.alert_config,
        )

        # Create test data
        self._create_test_data()

    def _create_test_data(self):
        """Create test market data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        returns_data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            returns_data[symbol] = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        self.risk_manager.update_returns_data(returns_data)

    @pytest.mark.asyncio
    async def test_real_time_risk_monitoring(self):
        """Test real-time risk monitoring with portfolio updates."""
        # Set up initial portfolio
        self.portfolio_manager.execute_trade("AAPL", 1000, 150.0, "long")
        self.portfolio_manager.execute_trade("GOOGL", 200, 2800.0, "long")

        # Start risk monitoring
        await self.alert_system.start_monitoring()

        # Simulate market movement
        self.portfolio_manager.update_prices(
            {
                "AAPL": 180.0,  # 20% increase
                "GOOGL": 2800.0,  # No change
            }
        )

        # Wait for monitoring cycle
        await asyncio.sleep(0.1)

        # Stop monitoring
        await self.alert_system.stop_monitoring()

        # Verify risk metrics were calculated
        weights = self.portfolio_manager.weights
        portfolio_var = self.risk_manager.calculate_portfolio_var(weights)

        assert portfolio_var > 0
        assert len(self.alert_system.risk_history) > 0

    def test_portfolio_rebalancing_with_risk_limits(self):
        """Test portfolio rebalancing with risk limit enforcement."""
        # Set up portfolio with drift
        self.portfolio_manager.execute_trade("AAPL", 1000, 150.0, "long")
        self.portfolio_manager.execute_trade("GOOGL", 200, 2800.0, "long")

        # Update prices to create drift
        self.portfolio_manager.update_prices(
            {
                "AAPL": 180.0,  # 20% increase
                "GOOGL": 2800.0,  # No change
            }
        )

        # Check current risk metrics
        weights = self.portfolio_manager.weights
        current_var = self.risk_manager.calculate_portfolio_var(weights)

        # If VaR exceeds limit, rebalance to reduce risk
        if current_var > self.risk_limits.max_portfolio_var:
            # Reduce position sizes to meet risk limits
            target_weights = {k: v * 0.8 for k, v in weights.items()}

            # Execute rebalancing trades
            for symbol, target_weight in target_weights.items():
                current_weight = weights.get(symbol, 0)
                if target_weight < current_weight:
                    # Reduce position
                    current_value = self.portfolio_manager.positions[symbol].market_value
                    target_value = target_weight * self.portfolio_manager.total_value
                    reduction_value = current_value - target_value

                    if reduction_value > 0:
                        reduction_quantity = int(
                            reduction_value / self.portfolio_manager.positions[symbol].current_price
                        )
                        if reduction_quantity > 0:
                            self.portfolio_manager.execute_trade(
                                symbol,
                                -reduction_quantity,
                                self.portfolio_manager.positions[symbol].current_price,
                                "long",
                            )

            # Verify risk is reduced
            new_weights = self.portfolio_manager.weights
            new_var = self.risk_manager.calculate_portfolio_var(new_weights)
            assert new_var <= self.risk_limits.max_portfolio_var

    def test_risk_aware_trade_execution(self):
        """Test risk-aware trade execution."""
        # Set up initial portfolio
        self.portfolio_manager.execute_trade("AAPL", 500, 150.0, "long")

        # Calculate pre-trade risk
        pre_trade_weights = self.portfolio_manager.weights
        pre_trade_var = self.risk_manager.calculate_portfolio_var(pre_trade_weights)

        # Attempt to execute a large trade
        large_trade_quantity = 2000
        large_trade_price = 150.0

        # Check if trade would violate risk limits
        potential_portfolio_value = self.portfolio_manager.total_value + (large_trade_quantity * large_trade_price)
        potential_weights = {
            "AAPL": (self.portfolio_manager.positions["AAPL"].market_value + large_trade_quantity * large_trade_price)
            / potential_portfolio_value
        }

        potential_var = self.risk_manager.calculate_portfolio_var(potential_weights)

        # Execute trade only if risk limits are not violated
        if potential_var <= self.risk_limits.max_portfolio_var:
            success = self.portfolio_manager.execute_trade("AAPL", large_trade_quantity, large_trade_price, "long")
            assert success
        else:
            # Execute smaller trade to stay within limits
            max_quantity = int(
                (self.risk_limits.max_portfolio_var - pre_trade_var) * potential_portfolio_value / large_trade_price
            )
            if max_quantity > 0:
                success = self.portfolio_manager.execute_trade("AAPL", max_quantity, large_trade_price, "long")
                assert success

    @pytest.mark.asyncio
    async def test_alert_triggered_portfolio_adjustment(self):
        """Test portfolio adjustment triggered by risk alerts."""
        # Set up portfolio that will trigger alerts
        self.portfolio_manager.execute_trade("AAPL", 2000, 150.0, "long")  # Large position

        # Update prices to increase risk
        self.portfolio_manager.update_prices({"AAPL": 180.0})

        # Mock alert triggering
        with patch.object(self.alert_system, "_trigger_alert", new_callable=AsyncMock) as mock_alert:
            # Check risk metrics
            weights = self.portfolio_manager.weights
            portfolio_var = self.risk_manager.calculate_portfolio_var(weights)

            if portfolio_var > 0.03:  # Alert threshold
                # Trigger portfolio adjustment
                # Reduce position to bring VaR below threshold
                current_position = self.portfolio_manager.positions["AAPL"]
                reduction_quantity = int(current_position.quantity * 0.3)  # Reduce by 30%

                if reduction_quantity > 0:
                    self.portfolio_manager.execute_trade(
                        "AAPL",
                        -reduction_quantity,
                        current_position.current_price,
                        "long",
                    )

                # Verify risk is reduced
                new_weights = self.portfolio_manager.weights
                new_var = self.risk_manager.calculate_portfolio_var(new_weights)
                assert new_var <= 0.03


class TestRegulatoryComplianceWorkflows:
    """Test regulatory compliance workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        # Risk management with regulatory limits
        self.risk_limits = RiskLimits(
            max_portfolio_var=0.02,  # Conservative VaR limit
            max_drawdown=0.10,  # Conservative drawdown limit
            max_leverage=1.0,  # No leverage for regulatory compliance
            max_position_size=0.05,  # Conservative position size limit
        )
        self.risk_manager = RiskManager(self.risk_limits)

        # Portfolio management with regulatory constraints
        self.portfolio_config = PortfolioConfig(
            max_position_size=0.05,
            max_sector_exposure=0.20,
            max_leverage=1.0,
            rebalance_threshold=0.02,
        )
        self.portfolio_manager = PortfolioManager(initial_capital=1000000, config=self.portfolio_config)

        # Create test data
        self._create_test_data()

    def _create_test_data(self):
        """Create test market data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        returns_data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]:
            returns_data[symbol] = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        self.risk_manager.update_returns_data(returns_data)

    def test_regulatory_position_limits(self):
        """Test regulatory position size limits."""
        # Try to execute trades that would exceed regulatory limits
        large_quantity = 10000  # Very large position
        price = 150.0

        # Calculate potential position size
        potential_position_value = large_quantity * price
        potential_position_size = potential_position_value / self.portfolio_manager.total_value

        if potential_position_size > self.risk_limits.max_position_size:
            # Should be rejected or reduced
            max_allowed_quantity = int(self.risk_limits.max_position_size * self.portfolio_manager.total_value / price)

            success = self.portfolio_manager.execute_trade("AAPL", max_allowed_quantity, price, "long")
            assert success

            # Verify position size is within limits
            weights = self.portfolio_manager.weights
            assert weights["AAPL"] <= self.risk_limits.max_position_size

    def test_regulatory_leverage_limits(self):
        """Test regulatory leverage limits."""
        # Try to create leveraged positions
        total_position_value = 0

        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]:
            # Calculate maximum position size for each asset
            max_quantity = int(self.risk_limits.max_position_size * self.portfolio_manager.total_value / 150.0)

            success = self.portfolio_manager.execute_trade(symbol, max_quantity, 150.0, "long")
            if success:
                total_position_value += self.portfolio_manager.positions[symbol].market_value

        # Calculate leverage
        leverage = total_position_value / self.portfolio_manager.total_value

        # Verify leverage is within regulatory limits
        assert leverage <= self.risk_limits.max_leverage

    def test_regulatory_risk_limits(self):
        """Test regulatory risk limits."""
        # Create portfolio and check risk metrics
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            max_quantity = int(self.risk_limits.max_position_size * self.portfolio_manager.total_value / 150.0)
            self.portfolio_manager.execute_trade(symbol, max_quantity, 150.0, "long")

        # Calculate risk metrics
        weights = self.portfolio_manager.weights
        portfolio_var = self.risk_manager.calculate_portfolio_var(weights)
        max_drawdown = self.risk_manager.calculate_portfolio_drawdown(weights)

        # Verify regulatory risk limits are satisfied
        assert portfolio_var <= self.risk_limits.max_portfolio_var
        assert max_drawdown <= self.risk_limits.max_drawdown

    def test_regulatory_reporting(self):
        """Test regulatory reporting requirements."""
        # Create portfolio with regulatory constraints
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            max_quantity = int(self.risk_limits.max_position_size * self.portfolio_manager.total_value / 150.0)
            self.portfolio_manager.execute_trade(symbol, max_quantity, 150.0, "long")

        # Generate regulatory reports
        portfolio_summary = self.portfolio_manager.get_performance_summary()
        risk_report = self.risk_manager.generate_risk_report(
            self.portfolio_manager.weights, self.portfolio_manager.total_value
        )

        # Verify regulatory reporting requirements
        assert "total_value" in portfolio_summary
        assert "leverage" in portfolio_summary
        assert "portfolio_var" in risk_report
        assert "max_drawdown" in risk_report
        assert "risk_limits_violations" in risk_report

        # Verify no violations
        violations = risk_report["risk_limits_violations"]
        assert len(violations) == 0


class TestEndToEndTradingScenarios:
    """Test end-to-end trading scenarios with risk management."""

    def setup_method(self):
        """Setup test fixtures."""
        # Comprehensive setup
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager(initial_capital=1000000)
        self.alert_manager = AlertManager()

        # Create test data
        self._create_test_data()

    def _create_test_data(self):
        """Create comprehensive test data."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        returns_data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]:
            returns_data[symbol] = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)

        self.risk_manager.update_returns_data(returns_data)

    def test_complete_trading_workflow(self):
        """Test complete trading workflow with risk management."""
        # 1. Initial portfolio optimization
        target_symbols = ["AAPL", "GOOGL", "MSFT"]
        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, {}, method="max_sharpe")

        if optimal_weights is not None:
            # 2. Execute initial trades
            for symbol, target_weight in optimal_weights.items():
                target_value = target_weight * self.portfolio_manager.total_value
                quantity = int(target_value / 150.0)  # Assume price of 150
                if quantity > 0:
                    self.portfolio_manager.execute_trade(symbol, quantity, 150.0, "long")

            # 3. Monitor portfolio performance
            initial_weights = self.portfolio_manager.weights
            initial_var = self.risk_manager.calculate_portfolio_var(initial_weights)

            # 4. Simulate market movement
            self.portfolio_manager.update_prices(
                {
                    "AAPL": 165.0,  # 10% increase
                    "GOOGL": 2800.0,  # No change
                    "MSFT": 315.0,  # 5% increase
                }
            )

            # 5. Check risk metrics
            current_weights = self.portfolio_manager.weights
            current_var = self.risk_manager.calculate_portfolio_var(current_weights)

            # 6. Rebalance if necessary
            if abs(current_var - initial_var) > 0.01:  # 1% change threshold
                # Re-optimize and rebalance
                new_optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, {}, method="max_sharpe")

                if new_optimal_weights is not None:
                    # Execute rebalancing trades
                    for symbol, new_weight in new_optimal_weights.items():
                        current_weight = current_weights.get(symbol, 0)
                        weight_diff = new_weight - current_weight

                        if abs(weight_diff) > 0.01:  # 1% weight difference threshold
                            # Calculate trade quantity
                            portfolio_value = self.portfolio_manager.total_value
                            trade_value = weight_diff * portfolio_value
                            trade_quantity = int(trade_value / self.portfolio_manager.positions[symbol].current_price)

                            if trade_quantity != 0:
                                self.portfolio_manager.execute_trade(
                                    symbol,
                                    trade_quantity,
                                    self.portfolio_manager.positions[symbol].current_price,
                                    "long",
                                )

            # 7. Final performance analysis
            final_weights = self.portfolio_manager.weights
            final_var = self.risk_manager.calculate_portfolio_var(final_weights)
            performance_summary = self.portfolio_manager.get_performance_summary()

            # Verify workflow completed successfully
            assert len(self.portfolio_manager.positions) > 0
            assert final_var > 0
            assert "total_return" in performance_summary

    def test_stress_testing_scenario(self):
        """Test stress testing scenario with portfolio management."""
        # Set up initial portfolio
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            self.portfolio_manager.execute_trade(symbol, 500, 150.0, "long")

        # Calculate baseline metrics
        baseline_weights = self.portfolio_manager.weights
        baseline_var = self.risk_manager.calculate_portfolio_var(baseline_weights)
        baseline_value = self.portfolio_manager.total_value

        # Simulate market stress (significant price declines)
        stress_prices = {
            "AAPL": 120.0,  # 20% decline
            "GOOGL": 2240.0,  # 20% decline
            "MSFT": 240.0,  # 20% decline
        }

        self.portfolio_manager.update_prices(stress_prices)

        # Calculate stress metrics
        stress_weights = self.portfolio_manager.weights
        stress_var = self.risk_manager.calculate_portfolio_var(stress_weights)
        stress_value = self.portfolio_manager.total_value

        # Calculate drawdown
        drawdown = (baseline_value - stress_value) / baseline_value

        # Verify stress scenario
        assert stress_value < baseline_value
        assert drawdown > 0.15  # Should have significant drawdown
        assert stress_var > baseline_var  # Risk should increase under stress

    def test_performance_attribution_with_risk(self):
        """Test performance attribution with risk metrics."""
        # Set up portfolio
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            self.portfolio_manager.execute_trade(symbol, 500, 150.0, "long")

        # Create performance attributor
        attributor = PerformanceAttributor()

        # Generate portfolio returns
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, 252))

        # Create asset returns and weights data
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        asset_returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.025, 252),
                "MSFT": np.random.normal(0.0012, 0.018, 252),
            },
            index=dates,
        )

        portfolio_weights = pd.DataFrame(
            {"AAPL": [0.4] * 252, "GOOGL": [0.35] * 252, "MSFT": [0.25] * 252},
            index=dates,
        )

        benchmark_weights = pd.DataFrame(
            {"AAPL": [0.3] * 252, "GOOGL": [0.4] * 252, "MSFT": [0.3] * 252},
            index=dates,
        )

        # Perform attribution analysis
        attribution_results = attributor.analyze_performance(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            asset_returns=asset_returns,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
        )

        # Verify attribution results
        assert "factor_attribution" in attribution_results
        assert "brinson_attribution" in attribution_results
        assert "risk_metrics" in attribution_results

        # Add risk metrics to attribution
        current_weights = self.portfolio_manager.weights
        portfolio_var = self.risk_manager.calculate_portfolio_var(current_weights)
        portfolio_cvar = self.risk_manager.calculate_portfolio_cvar(current_weights)

        attribution_results["risk_metrics"]["portfolio_var"] = portfolio_var
        attribution_results["risk_metrics"]["portfolio_cvar"] = portfolio_cvar

        # Verify risk-adjusted attribution
        assert attribution_results["risk_metrics"]["portfolio_var"] > 0
        assert attribution_results["risk_metrics"]["portfolio_cvar"] > portfolio_var


class TestIntegrationPerformance:
    """Test performance of integrated risk-portfolio systems."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager(initial_capital=1000000)

        # Create large dataset for performance testing
        self._create_large_dataset()

    def _create_large_dataset(self):
        """Create large dataset for performance testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D")

        # Create 50 assets
        returns_data = {}
        for i in range(50):
            symbol = f"ASSET_{i}"
            returns_data[symbol] = pd.Series(np.random.normal(0.001, 0.02, 1000), index=dates)

        self.risk_manager.update_returns_data(returns_data)

    def test_large_portfolio_optimization_performance(self):
        """Test performance of large portfolio optimization."""
        # Create large portfolio
        start_time = time.time()

        for i in range(50):
            symbol = f"ASSET_{i}"
            self.portfolio_manager.execute_trade(symbol, 100, 100.0, "long")

        # Calculate risk metrics for large portfolio
        weights = self.portfolio_manager.weights
        portfolio_var = self.risk_manager.calculate_portfolio_var(weights)
        portfolio_cvar = self.risk_manager.calculate_portfolio_cvar(weights)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert len(self.portfolio_manager.positions) == 50
        assert portfolio_var > 0
        assert portfolio_cvar > portfolio_var

    def test_real_time_monitoring_performance(self):
        """Test performance of real-time monitoring."""
        # Set up portfolio
        for i in range(20):
            symbol = f"ASSET_{i}"
            self.portfolio_manager.execute_trade(symbol, 100, 100.0, "long")

        # Simulate real-time monitoring
        start_time = time.time()

        for _ in range(100):  # 100 monitoring cycles
            # Update prices
            new_prices = {}
            for symbol in self.portfolio_manager.positions:
                new_prices[symbol] = 100.0 + np.random.normal(0, 1)

            self.portfolio_manager.update_prices(new_prices)

            # Calculate risk metrics
            weights = self.portfolio_manager.weights
            self.risk_manager.calculate_portfolio_var(weights)
            self.risk_manager.calculate_portfolio_cvar(weights)

        end_time = time.time()
        monitoring_time = end_time - start_time

        # Verify performance
        assert monitoring_time < 10.0  # Should complete within 10 seconds

    def test_stress_testing_performance(self):
        """Test performance of stress testing."""
        # Set up portfolio
        for i in range(30):
            symbol = f"ASSET_{i}"
            self.portfolio_manager.execute_trade(symbol, 100, 100.0, "long")

        # Perform stress testing
        start_time = time.time()

        # Multiple stress scenarios
        stress_scenarios = [
            {"volatility_multiplier": 2.0, "correlation_increase": 0.3},
            {"volatility_multiplier": 3.0, "correlation_increase": 0.5},
            {"volatility_multiplier": 1.5, "correlation_increase": 0.2},
        ]

        for scenario in stress_scenarios:
            # Apply stress scenario
            stressed_returns = {}
            for symbol, returns in self.risk_manager._returns_data.items():
                stressed_returns[symbol] = returns * scenario["volatility_multiplier"]

            self.risk_manager.update_returns_data(stressed_returns)

            # Calculate stressed risk metrics
            weights = self.portfolio_manager.weights
            self.risk_manager.calculate_portfolio_var(weights)
            self.risk_manager.calculate_portfolio_cvar(weights)

        end_time = time.time()
        stress_testing_time = end_time - start_time

        # Verify performance
        assert stress_testing_time < 15.0  # Should complete within 15 seconds


if __name__ == "__main__":
    pytest.main([__file__])
