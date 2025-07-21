"""
Comprehensive tests for portfolio management components.

Tests cover:
- Portfolio optimization algorithms
- Portfolio rebalancing logic
- Transaction cost modeling
- Performance attribution analysis
- Risk-adjusted portfolio management
- Integration with risk management
- Performance benchmarks
"""

import time
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.trade_agent.portfolio.attribution import (
    AttributionConfig,
    BrinsonAttributor,
    FactorModel,
    PerformanceAttributor,
)
from src.trade_agent.portfolio.manager import (
    PortfolioConfig,
    PortfolioManager,
    Position,
)
from src.trade_agent.portfolio.transaction_costs import (
    BrokerType,
    MarketCondition,
    MarketData,
    OrderType,
    TransactionCostModel,
)


class TestPortfolioManager:
    """Test portfolio manager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = PortfolioConfig(
            max_position_size=0.1,
            max_sector_exposure=0.3,
            max_leverage=1.0,
            rebalance_frequency="monthly",
            rebalance_threshold=0.05,
        )

        self.portfolio_manager = PortfolioManager(initial_capital=100000, config=self.config)

    def test_portfolio_initialization(self):
        """Test portfolio manager initialization."""
        assert self.portfolio_manager.initial_capital == 100000
        assert self.portfolio_manager.cash == 100000
        assert len(self.portfolio_manager.positions) == 0
        assert self.portfolio_manager.total_value == 100000

    def test_position_creation(self):
        """Test position creation and properties."""
        position = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now(),
            side="long",
        )

        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.market_value == 15500.0
        assert position.unrealized_pnl == 500.0
        assert position.unrealized_pnl_pct == pytest.approx(0.0333, abs=1e-3)

    def test_short_position(self):
        """Test short position calculations."""
        position = Position(
            symbol="TSLA",
            quantity=50,
            entry_price=200.0,
            current_price=180.0,
            timestamp=datetime.now(),
            side="short",
        )

        assert position.market_value == 9000.0
        assert position.unrealized_pnl == 1000.0
        assert position.unrealized_pnl_pct == 0.1

    def test_portfolio_weights_calculation(self):
        """Test portfolio weights calculation."""
        # Add some positions
        self.portfolio_manager.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now(),
        )

        self.portfolio_manager.positions["GOOGL"] = Position(
            symbol="GOOGL",
            quantity=50,
            entry_price=2800.0,
            current_price=2850.0,
            timestamp=datetime.now(),
        )

        weights = self.portfolio_manager.weights

        assert isinstance(weights, dict)
        assert "AAPL" in weights
        assert "GOOGL" in weights
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_leverage_calculation(self):
        """Test leverage calculation."""
        # Add positions with leverage
        self.portfolio_manager.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=1000,  # Large position
            entry_price=150.0,
            current_price=155.0,
            timestamp=datetime.now(),
        )

        leverage = self.portfolio_manager.leverage
        assert leverage > 0.5  # Should have some leverage

    def test_price_update(self):
        """Test price update functionality."""
        # Add a position
        self.portfolio_manager.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            current_price=150.0,
            timestamp=datetime.now(),
        )

        initial_value = self.portfolio_manager.total_value

        # Update prices
        new_prices = {"AAPL": 155.0}
        self.portfolio_manager.update_prices(new_prices)

        # Total value should increase
        assert self.portfolio_manager.total_value > initial_value
        assert self.portfolio_manager.positions["AAPL"].current_price == 155.0

    def test_trade_execution(self):
        """Test trade execution."""
        # Mock the partial fill model to ensure full fills
        with patch.object(
            self.portfolio_manager.transaction_cost_model.partial_fill_model,
            "simulate_fill",
        ) as mock_fill:
            mock_fill.return_value = (50, [(50, 150.0, datetime.now())])  # Full fill

            # Execute a smaller trade to avoid max position size limits
            success = self.portfolio_manager.execute_trade(symbol="AAPL", quantity=50, price=150.0, side="long")

            assert success
            assert "AAPL" in self.portfolio_manager.positions
            assert self.portfolio_manager.positions["AAPL"].quantity == 50
            assert self.portfolio_manager.cash < 100000  # Cash should decrease

    def test_trade_validation(self):
        """Test trade validation."""
        # Try to buy more than available cash
        success = self.portfolio_manager.execute_trade(
            symbol="AAPL",
            quantity=10000,  # Very large quantity
            price=150.0,
            side="long",
        )

        assert not success  # Should fail due to insufficient cash

    def test_portfolio_optimization(self):
        """Test portfolio optimization."""
        # Create price data for optimization
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        price_data = {}

        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            # Generate price series
            prices = pd.DataFrame(
                {
                    "close": np.random.normal(150, 10, 252).cumsum() + 150,
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            price_data[symbol] = prices

        # Optimize portfolio
        target_symbols = ["AAPL", "GOOGL", "MSFT"]
        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, price_data, method="max_sharpe")

        if optimal_weights is not None:
            assert isinstance(optimal_weights, dict)
            assert sum(optimal_weights.values()) == pytest.approx(1.0, abs=1e-6)
            assert all(w >= 0 for w in optimal_weights.values())

    def test_performance_summary(self):
        """Test performance summary generation."""
        # Add some trading history
        self.portfolio_manager.execute_trade("AAPL", 50, 150.0, "long")
        self.portfolio_manager.update_prices({"AAPL": 155.0})

        summary = self.portfolio_manager.get_performance_summary()

        assert isinstance(summary, dict)
        assert "total_return" in summary
        assert "max_drawdown" in summary
        assert "win_rate" in summary
        # Sharpe ratio may not be available if empyrical is not available
        if "sharpe_ratio" in summary:
            assert isinstance(summary["sharpe_ratio"], int | float)

    def test_transaction_cost_analysis(self):
        """Test transaction cost analysis."""
        # Add some trades first to avoid division by zero
        self.portfolio_manager.execute_trade("AAPL", 50, 150.0, "long")

        analysis = self.portfolio_manager.get_transaction_cost_analysis()

        assert isinstance(analysis, dict)
        assert "cost_summary" in analysis
        assert "cost_trends" in analysis
        assert "efficiency_metrics" in analysis


class TestPortfolioOptimization:
    """Test portfolio optimization algorithms."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = PortfolioConfig(max_position_size=0.2, max_sector_exposure=0.4, max_leverage=1.5)

        self.portfolio_manager = PortfolioManager(initial_capital=1000000, config=self.config)

    def test_max_sharpe_optimization(self):
        """Test maximum Sharpe ratio optimization."""
        # Create realistic price data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        price_data = {}
        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]:
            # Generate correlated price series
            base_prices = 100 + np.cumsum(np.random.normal(0.001, 0.02, 252))
            prices = pd.DataFrame(
                {
                    "close": base_prices,
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            price_data[symbol] = prices

        target_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, price_data, method="max_sharpe")

        if optimal_weights is not None:
            assert isinstance(optimal_weights, dict)
            assert len(optimal_weights) == len(target_symbols)
            assert sum(optimal_weights.values()) == pytest.approx(1.0, abs=1e-6)
            assert all(w >= 0 for w in optimal_weights.values())

    def test_min_variance_optimization(self):
        """Test minimum variance optimization."""
        # Create price data
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        price_data = {}

        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            prices = pd.DataFrame(
                {
                    "close": 100 + np.cumsum(np.random.normal(0.001, 0.02, 252)),
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            price_data[symbol] = prices

        target_symbols = ["AAPL", "GOOGL", "MSFT"]

        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, price_data, method="min_variance")

        if optimal_weights is not None:
            assert isinstance(optimal_weights, dict)
            assert sum(optimal_weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_risk_parity_optimization(self):
        """Test risk parity optimization."""
        # Create price data with different volatilities
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        price_data = {}

        volatilities = [0.15, 0.25, 0.35]  # Different volatilities
        for i, symbol in enumerate(["AAPL", "GOOGL", "MSFT"]):
            prices = pd.DataFrame(
                {
                    "close": 100 + np.cumsum(np.random.normal(0.001, volatilities[i], 252)),
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            price_data[symbol] = prices

        target_symbols = ["AAPL", "GOOGL", "MSFT"]

        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, price_data, method="risk_parity")

        if optimal_weights is not None:
            assert isinstance(optimal_weights, dict)
            assert sum(optimal_weights.values()) == pytest.approx(1.0, abs=1e-6)


class TestTransactionCostModel:
    """Test transaction cost modeling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.retail_model = TransactionCostModel.create_broker_model(BrokerType.RETAIL)
        self.institutional_model = TransactionCostModel.create_broker_model(BrokerType.INSTITUTIONAL)

    def test_commission_calculation(self):
        """Test commission calculation."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.02,
            mid_price=100.01,
            volume=1000000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )

        # Test retail commission calculation
        retail_cost_breakdown = self.retail_model.calculate_total_cost(
            trade_value=100000,  # $100k trade
            quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        # Test institutional commission calculation
        inst_cost_breakdown = self.institutional_model.calculate_total_cost(
            trade_value=100000,  # $100k trade
            quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        assert retail_cost_breakdown["commission"] > 0
        assert inst_cost_breakdown["commission"] > 0
        # Retail should generally have higher commission rates
        assert retail_cost_breakdown["commission"] >= inst_cost_breakdown["commission"]

    def test_spread_cost_calculation(self):
        """Test bid-ask spread cost calculation."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.02,
            mid_price=100.01,
            volume=1000000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )

        cost_breakdown = self.retail_model.calculate_total_cost(
            trade_value=100000,  # $100k trade
            quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        assert cost_breakdown["spread_cost"] > 0
        # Spread cost should be approximately half the spread * trade value
        expected_spread_cost = 100000 * (0.02 / 100.01)  # Half spread
        assert cost_breakdown["spread_cost"] == pytest.approx(expected_spread_cost, rel=0.1)

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.02,
            mid_price=100.01,
            volume=1000000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )

        cost_breakdown = self.retail_model.calculate_total_cost(
            trade_value=1000000,  # Large $1M trade
            quantity=10000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        assert cost_breakdown["slippage"] >= 0
        # Large orders should have higher slippage
        assert cost_breakdown["slippage"] > 0

    def test_total_transaction_cost(self):
        """Test total transaction cost calculation."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.02,
            mid_price=100.01,
            volume=1000000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )

        cost_breakdown = self.retail_model.calculate_total_cost(
            trade_value=100000,  # $100k trade
            quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        assert cost_breakdown["total_cost"] > 0
        assert "commission" in cost_breakdown
        assert "spread_cost" in cost_breakdown
        assert "slippage" in cost_breakdown
        assert "total_cost" in cost_breakdown

    def test_market_condition_impact(self):
        """Test impact of market conditions on transaction costs."""
        market_data = MarketData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.02,
            mid_price=100.01,
            volume=1000000,
            volatility=0.02,
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )

        # Normal market
        normal_cost = self.retail_model.calculate_total_cost(
            trade_value=100000,  # $100k trade
            quantity=1000,
            market_data=market_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.NORMAL,
        )

        # High volatility market
        high_vol_data = MarketData(
            timestamp=datetime.now(),
            bid=100.0,
            ask=100.02,
            mid_price=100.01,
            volume=1000000,
            volatility=0.05,  # Higher volatility
            avg_daily_volume=1000000,
            market_cap=1000000000,
        )

        high_vol_cost = self.retail_model.calculate_total_cost(
            trade_value=100000,  # $100k trade
            quantity=1000,
            market_data=high_vol_data,
            order_type=OrderType.MARKET,
            market_condition=MarketCondition.VOLATILE,
        )

        # High volatility should have higher costs
        assert high_vol_cost["total_cost"] > normal_cost["total_cost"]


class TestPerformanceAttribution:
    """Test performance attribution analysis."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = AttributionConfig(risk_free_rate=0.02, confidence_level=0.95, lookback_period=252)

        self.attributor = PerformanceAttributor(self.config)

    def test_factor_model_creation(self):
        """Test factor model creation and fitting."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Generate asset returns
        asset_returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.025, 252),
                "MSFT": np.random.normal(0.0012, 0.018, 252),
                "TSLA": np.random.normal(0.002, 0.04, 252),
            },
            index=dates,
        )

        market_returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)

        # Create factor model
        factor_model = FactorModel(self.config)
        factor_model.fit(asset_returns, market_returns)

        assert factor_model.factors is not None
        assert factor_model.factor_loadings is not None
        assert factor_model.factor_returns is not None

    def test_return_decomposition(self):
        """Test return decomposition into systematic and idiosyncratic components."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        asset_returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.025, 252),
                "MSFT": np.random.normal(0.0012, 0.018, 252),
            },
            index=dates,
        )

        market_returns = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)

        # Create and fit factor model
        factor_model = FactorModel(self.config)
        factor_model.fit(asset_returns, market_returns)

        # Decompose returns
        decomposition = factor_model.decompose_returns(asset_returns)

        assert "systematic" in decomposition
        assert "idiosyncratic" in decomposition
        assert decomposition["systematic"].shape == asset_returns.shape
        assert decomposition["idiosyncratic"].shape == asset_returns.shape

    def test_brinson_attribution(self):
        """Test Brinson attribution analysis."""
        # Create test data
        portfolio_weights = pd.Series({"AAPL": 0.4, "GOOGL": 0.35, "MSFT": 0.25})

        benchmark_weights = pd.Series({"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3})

        returns = pd.Series({"AAPL": 0.05, "GOOGL": 0.03, "MSFT": 0.04})

        benchmark_returns = pd.Series({"AAPL": 0.04, "GOOGL": 0.035, "MSFT": 0.045})

        # Create Brinson attributor
        brinson_attributor = BrinsonAttributor(self.config)

        attribution = brinson_attributor.calculate_attribution(
            portfolio_weights, benchmark_weights, returns, benchmark_returns
        )

        assert isinstance(attribution, dict)
        assert "allocation" in attribution
        assert "selection" in attribution
        assert "interaction" in attribution

    def test_comprehensive_attribution_analysis(self):
        """Test comprehensive performance attribution analysis."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # Portfolio and benchmark returns
        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, 252), index=dates)

        # Asset returns
        asset_returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.025, 252),
                "MSFT": np.random.normal(0.0012, 0.018, 252),
            },
            index=dates,
        )

        # Portfolio and benchmark weights
        portfolio_weights = pd.DataFrame(
            {"AAPL": [0.4] * 252, "GOOGL": [0.35] * 252, "MSFT": [0.25] * 252},
            index=dates,
        )

        benchmark_weights = pd.DataFrame(
            {"AAPL": [0.3] * 252, "GOOGL": [0.4] * 252, "MSFT": [0.3] * 252},
            index=dates,
        )

        # Sector data
        sector_data = pd.DataFrame(
            {"sector": ["Technology", "Technology", "Technology"]},
            index=["AAPL", "GOOGL", "MSFT"],
        )

        # Perform comprehensive analysis
        results = self.attributor.analyze_performance(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            asset_returns=asset_returns,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
            sector_data=sector_data,
        )

        assert isinstance(results, dict)
        assert "factor_attribution" in results
        assert "brinson_attribution" in results
        assert "risk_adjusted" in results
        assert "decomposition" in results

    def test_attribution_dashboard_creation(self):
        """Test attribution dashboard creation."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, 252), index=dates)

        # Asset returns
        asset_returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.025, 252),
                "MSFT": np.random.normal(0.0012, 0.018, 252),
            },
            index=dates,
        )

        # Portfolio and benchmark weights
        portfolio_weights = pd.DataFrame(
            {"AAPL": [0.4] * 252, "GOOGL": [0.35] * 252, "MSFT": [0.25] * 252},
            index=dates,
        )

        benchmark_weights = pd.DataFrame(
            {"AAPL": [0.3] * 252, "GOOGL": [0.4] * 252, "MSFT": [0.3] * 252},
            index=dates,
        )

        # Run analysis first
        self.attributor.analyze_performance(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            asset_returns=asset_returns,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
        )

        # Create dashboard
        dashboard = self.attributor.create_dashboard(portfolio_returns, benchmark_returns)

        assert dashboard is not None

    def test_attribution_report_generation(self):
        """Test attribution report generation."""
        # Create test data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, 252), index=dates)

        # Asset returns
        asset_returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.001, 0.02, 252),
                "GOOGL": np.random.normal(0.0008, 0.025, 252),
                "MSFT": np.random.normal(0.0012, 0.018, 252),
            },
            index=dates,
        )

        # Portfolio and benchmark weights
        portfolio_weights = pd.DataFrame(
            {"AAPL": [0.4] * 252, "GOOGL": [0.35] * 252, "MSFT": [0.25] * 252},
            index=dates,
        )

        benchmark_weights = pd.DataFrame(
            {"AAPL": [0.3] * 252, "GOOGL": [0.4] * 252, "MSFT": [0.3] * 252},
            index=dates,
        )

        # Run analysis first
        self.attributor.analyze_performance(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            asset_returns=asset_returns,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
        )

        # Generate report
        report = self.attributor.generate_report()

        assert isinstance(report, str)
        assert len(report) > 0


class TestPortfolioRebalancing:
    """Test portfolio rebalancing logic."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = PortfolioConfig(
            rebalance_frequency="monthly",
            rebalance_threshold=0.05,
            max_position_size=0.2,
        )

        self.portfolio_manager = PortfolioManager(initial_capital=100000, config=self.config)

    def test_rebalancing_needs_detection(self):
        """Test detection of rebalancing needs."""
        # Set up portfolio with drift
        self.portfolio_manager.positions["AAPL"] = Position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            current_price=150.0,
            timestamp=datetime.now(),
        )

        self.portfolio_manager.positions["GOOGL"] = Position(
            symbol="GOOGL",
            quantity=50,
            entry_price=2800.0,
            current_price=2800.0,
            timestamp=datetime.now(),
        )

        # Update prices to create drift
        self.portfolio_manager.update_prices(
            {
                "AAPL": 180.0,  # 20% increase
                "GOOGL": 2800.0,  # No change
            }
        )

        # Check if rebalancing is needed
        weights = self.portfolio_manager.weights
        target_weights = {"AAPL": 0.5, "GOOGL": 0.5}

        # Calculate drift
        drift = sum(
            abs(weights.get(asset, 0) - target_weights.get(asset, 0))
            for asset in set(weights.keys()) | set(target_weights.keys())
        )

        assert drift > 0.05  # Should exceed threshold

    def test_rebalancing_execution(self):
        """Test rebalancing execution."""
        # Set up initial portfolio
        self.portfolio_manager.execute_trade("AAPL", 100, 150.0, "long")
        self.portfolio_manager.execute_trade("GOOGL", 50, 2800.0, "long")

        # Update prices to create drift
        self.portfolio_manager.update_prices({"AAPL": 180.0, "GOOGL": 2800.0})

        # Define target weights
        target_weights = {"AAPL": 0.5, "GOOGL": 0.5}

        # Execute rebalancing
        rebalance_trades = self.portfolio_manager._calculate_rebalance_trades(target_weights)

        assert isinstance(rebalance_trades, list)
        # Should contain trades to achieve target weights

    def test_rebalancing_cost_analysis(self):
        """Test rebalancing cost analysis."""
        # Set up portfolio
        self.portfolio_manager.execute_trade("AAPL", 100, 150.0, "long")

        # Calculate rebalancing costs
        market_data = MarketData(bid=150.0, ask=150.02, volume=1000000, volatility=0.02)

        rebalance_cost = self.portfolio_manager.transaction_cost_model.calculate_total_cost(
            quantity=50,
            price=150.0,
            market_data=market_data,
            order_type=OrderType.MARKET,
            side="buy",
            market_condition=MarketCondition.NORMAL,
        )

        assert rebalance_cost["total_cost"] > 0


class TestPortfolioIntegration:
    """Test portfolio integration scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config = PortfolioConfig(max_position_size=0.15, max_sector_exposure=0.4, max_leverage=1.2)

        self.portfolio_manager = PortfolioManager(initial_capital=1000000, config=self.config)

    def test_comprehensive_portfolio_workflow(self):
        """Test comprehensive portfolio management workflow."""
        # 1. Initial portfolio setup
        self.portfolio_manager.execute_trade("AAPL", 1000, 150.0, "long")
        self.portfolio_manager.execute_trade("GOOGL", 200, 2800.0, "long")
        self.portfolio_manager.execute_trade("MSFT", 500, 300.0, "long")

        # 2. Price updates
        self.portfolio_manager.update_prices({"AAPL": 155.0, "GOOGL": 2850.0, "MSFT": 310.0})

        # 3. Performance analysis
        performance = self.portfolio_manager.get_performance_summary()

        # 4. Risk analysis
        weights = self.portfolio_manager.weights
        total_value = self.portfolio_manager.total_value

        # 5. Transaction cost analysis
        cost_analysis = self.portfolio_manager.get_transaction_cost_analysis()

        # Validate results
        assert len(self.portfolio_manager.positions) == 3
        assert total_value > 1000000
        assert isinstance(performance, dict)
        assert isinstance(weights, dict)
        assert isinstance(cost_analysis, dict)

    def test_portfolio_optimization_workflow(self):
        """Test portfolio optimization workflow."""
        # Create price data
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        price_data = {}

        for symbol in ["AAPL", "GOOGL", "MSFT", "TSLA"]:
            prices = pd.DataFrame(
                {
                    "close": 100 + np.cumsum(np.random.normal(0.001, 0.02, 252)),
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            price_data[symbol] = prices

        # Optimize portfolio
        target_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, price_data, method="max_sharpe")

        if optimal_weights is not None:
            # Execute trades to achieve optimal weights
            for symbol, target_weight in optimal_weights.items():
                target_value = target_weight * self.portfolio_manager.total_value
                current_value = self.portfolio_manager.positions.get(
                    symbol,
                    Position(
                        symbol=symbol,
                        quantity=0,
                        entry_price=0,
                        current_price=0,
                        timestamp=datetime.now(),
                    ),
                ).market_value

                if target_value > current_value:
                    # Buy more
                    quantity = int((target_value - current_value) / price_data[symbol]["close"].iloc[-1])
                    if quantity > 0:
                        self.portfolio_manager.execute_trade(
                            symbol,
                            quantity,
                            price_data[symbol]["close"].iloc[-1],
                            "long",
                        )

            # Verify weights are close to optimal
            final_weights = self.portfolio_manager.weights
            for symbol in optimal_weights:
                if symbol in final_weights:
                    assert abs(final_weights[symbol] - optimal_weights[symbol]) < 0.1


class TestPortfolioPerformanceBenchmarks:
    """Test portfolio performance benchmarks."""

    def setup_method(self):
        """Setup test fixtures."""
        self.portfolio_manager = PortfolioManager(initial_capital=1000000)

    def test_trade_execution_performance(self):
        """Test trade execution performance."""
        start_time = time.time()

        # Execute multiple trades
        for i in range(100):
            self.portfolio_manager.execute_trade(f"ASSET_{i}", 100, 100.0, "long")

        end_time = time.time()
        execution_time = end_time - start_time

        assert execution_time < 1.0  # Should complete within 1 second
        assert len(self.portfolio_manager.positions) == 100

    def test_portfolio_optimization_performance(self):
        """Test portfolio optimization performance."""
        # Create large dataset
        dates = pd.date_range("2023-01-01", periods=252, freq="D")
        price_data = {}

        for i in range(50):  # 50 assets
            symbol = f"ASSET_{i}"
            prices = pd.DataFrame(
                {
                    "close": 100 + np.cumsum(np.random.normal(0.001, 0.02, 252)),
                    "volume": np.random.randint(1000000, 10000000, 252),
                },
                index=dates,
            )
            price_data[symbol] = prices

        target_symbols = list(price_data.keys())

        start_time = time.time()
        optimal_weights = self.portfolio_manager.optimize_portfolio(target_symbols, price_data, method="max_sharpe")
        end_time = time.time()

        optimization_time = end_time - start_time

        assert optimization_time < 10.0  # Should complete within 10 seconds
        if optimal_weights is not None:
            assert len(optimal_weights) == 50

    def test_attribution_analysis_performance(self):
        """Test attribution analysis performance."""
        # Create large dataset
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        portfolio_returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates)
        benchmark_returns = pd.Series(np.random.normal(0.0008, 0.018, 252), index=dates)

        asset_returns = pd.DataFrame(
            {f"ASSET_{i}": np.random.normal(0.001, 0.02, 252) for i in range(20)},
            index=dates,
        )

        portfolio_weights = pd.DataFrame({f"ASSET_{i}": [0.05] * 252 for i in range(20)}, index=dates)

        benchmark_weights = pd.DataFrame({f"ASSET_{i}": [0.05] * 252 for i in range(20)}, index=dates)

        attributor = PerformanceAttributor()

        start_time = time.time()
        results = attributor.analyze_performance(
            portfolio_returns=portfolio_returns,
            benchmark_returns=benchmark_returns,
            asset_returns=asset_returns,
            portfolio_weights=portfolio_weights,
            benchmark_weights=benchmark_weights,
        )
        end_time = time.time()

        analysis_time = end_time - start_time

        assert analysis_time < 5.0  # Should complete within 5 seconds
        assert isinstance(results, dict)


class TestPortfolioErrorHandling:
    """Test portfolio error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.portfolio_manager = PortfolioManager(initial_capital=100000)

    def test_invalid_trade_parameters(self):
        """Test handling of invalid trade parameters."""
        # Negative quantity
        success = self.portfolio_manager.execute_trade("AAPL", -100, 150.0, "long")
        assert not success

        # Zero price
        success = self.portfolio_manager.execute_trade("AAPL", 100, 0.0, "long")
        assert not success

        # Invalid side
        success = self.portfolio_manager.execute_trade("AAPL", 100, 150.0, "invalid")
        assert not success

    def test_insufficient_cash_handling(self):
        """Test handling of insufficient cash."""
        # Try to buy more than available cash
        success = self.portfolio_manager.execute_trade("AAPL", 10000, 150.0, "long")
        assert not success

    def test_invalid_price_data(self):
        """Test handling of invalid price data."""
        # Empty price data
        self.portfolio_manager.update_prices({})

        # Invalid price values
        self.portfolio_manager.update_prices({"AAPL": -100.0})

        # Should handle gracefully without errors

    def test_optimization_with_insufficient_data(self):
        """Test optimization with insufficient data."""
        # Create minimal price data
        dates = pd.date_range("2023-01-01", periods=10, freq="D")  # Very short period
        price_data = {"AAPL": pd.DataFrame({"close": [100] * 10, "volume": [1000000] * 10}, index=dates)}

        optimal_weights = self.portfolio_manager.optimize_portfolio(["AAPL"], price_data, method="max_sharpe")

        # Should handle insufficient data gracefully
        assert optimal_weights is None or isinstance(optimal_weights, dict)


if __name__ == "__main__":
    pytest.main([__file__])
