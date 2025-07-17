"""
Tests for the unified BacktestEvaluator.

This module tests the integration of backtesting framework with model evaluation pipeline.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.core.unified_config import BacktestConfig
from trading_rl_agent.eval.backtest_evaluator import (
    BacktestEvaluator,
    BacktestResult,
    TradeRecord,
    TransactionCostModel,
)


class TestTransactionCostModel:
    """Test transaction cost modeling functionality."""

    def test_basic_cost_calculation(self):
        """Test basic transaction cost calculation."""
        model = TransactionCostModel(
            commission_rate=0.001,
            slippage_rate=0.0001,
            market_impact_rate=0.00005,
            bid_ask_spread=0.0002,
        )

        costs = model.calculate_total_cost(
            trade_value=10000.0,
            trade_volume=100,
            avg_daily_volume=1000000.0,
        )

        assert "commission" in costs
        assert "slippage" in costs
        assert "market_impact" in costs
        assert "spread_cost" in costs
        assert "total_cost" in costs
        assert "cost_pct" in costs

        assert costs["commission"] == 10.0  # 0.1% of 10000
        assert costs["slippage"] == 1.0  # 0.01% of 10000
        assert costs["spread_cost"] == 2.0  # 0.02% of 10000
        assert costs["total_cost"] > 0
        assert costs["cost_pct"] > 0

    def test_square_root_slippage_model(self):
        """Test square root slippage model."""
        model = TransactionCostModel(
            slippage_model="square_root",
            slippage_rate=0.0001,
        )

        # Test with different trade volumes
        costs_small = model.calculate_total_cost(10000.0, 100)
        costs_large = model.calculate_total_cost(10000.0, 10000)

        # Larger trades should have higher slippage
        assert costs_large["slippage"] > costs_small["slippage"]

    def test_market_impact_model(self):
        """Test market impact modeling."""
        model = TransactionCostModel(
            impact_model="square_root",
            market_impact_rate=0.00005,
        )

        # Test with different volume ratios
        costs_low_volume = model.calculate_total_cost(10000.0, 100, 1000000.0)
        costs_high_volume = model.calculate_total_cost(10000.0, 100000, 1000000.0)

        # Higher volume ratio should have higher market impact
        assert costs_high_volume["market_impact"] > costs_low_volume["market_impact"]


class TestBacktestEvaluator:
    """Test the unified BacktestEvaluator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        np.random.seed(42)

        # Generate realistic price data
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Daily returns
        prices = 100 * np.exp(np.cumsum(returns))  # Starting at $100

        return pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, len(dates))),
                "high": prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
                "low": prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
                "close": prices,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            },
            index=dates,
        )

    @pytest.fixture
    def sample_signals(self, sample_data):
        """Create sample trading signals."""
        # Simple momentum strategy
        returns = sample_data["close"].pct_change()
        signals = pd.Series(0, index=sample_data.index)
        signals[returns > returns.rolling(20).mean()] = 1
        signals[returns < returns.rolling(20).mean()] = -1
        return signals

    @pytest.fixture
    def backtest_config(self):
        """Create backtest configuration."""
        return BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            symbols=["AAPL"],
            initial_capital=100000.0,
            commission_rate=0.001,
            slippage_rate=0.0001,
            max_position_size=0.1,
            max_leverage=1.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )

    @pytest.fixture
    def evaluator(self, backtest_config):
        """Create BacktestEvaluator instance."""
        cost_model = TransactionCostModel(
            commission_rate=backtest_config.commission_rate,
            slippage_rate=backtest_config.slippage_rate,
        )
        return BacktestEvaluator(backtest_config, cost_model)

    def test_initialization(self, evaluator, backtest_config):
        """Test evaluator initialization."""
        assert evaluator.config == backtest_config
        assert evaluator.portfolio_manager.initial_capital == backtest_config.initial_capital
        assert len(evaluator.trades) == 0
        assert len(evaluator.equity_curve) == 0

    def test_input_validation(self, evaluator, sample_data):
        """Test input validation."""
        # Test valid inputs
        signals = pd.Series(0, index=sample_data.index)
        evaluator._validate_inputs(sample_data, signals)

        # Test invalid inputs
        with pytest.raises(ValueError, match="same length"):
            evaluator._validate_inputs(sample_data, signals.iloc[:-1])

        with pytest.raises(ValueError, match="Missing required columns"):
            invalid_data = sample_data.drop(columns=["close"])
            evaluator._validate_inputs(invalid_data, signals)

    def test_position_size_calculation(self, evaluator):
        """Test position size calculation."""
        # Test with different signal strengths
        size_weak = evaluator._calculate_position_size(100.0, 0.5)
        size_strong = evaluator._calculate_position_size(100.0, 1.0)

        assert size_strong > size_weak
        assert size_weak > 0
        assert size_strong > 0

    def test_consecutive_calculation(self, evaluator):
        """Test consecutive wins/losses calculation."""
        # Test consecutive wins
        wins_series = pd.Series([True, True, False, True, True, True, False])
        max_consecutive = evaluator._calculate_max_consecutive(wins_series)
        assert max_consecutive == 3

        # Test consecutive losses
        losses_series = pd.Series([False, False, True, False, False, True, False])
        max_consecutive = evaluator._calculate_max_consecutive(losses_series)
        assert max_consecutive == 2

    def test_drawdown_calculation(self, evaluator):
        """Test drawdown series calculation."""
        # Create sample equity curve
        equity_series = pd.Series([100, 110, 105, 120, 115, 130, 125])
        drawdown_series = evaluator._calculate_drawdown_series(equity_series)

        assert len(drawdown_series) == len(equity_series)
        assert drawdown_series.iloc[0] == 0  # First value should be 0
        assert drawdown_series.min() <= 0  # Should have negative drawdowns

    def test_basic_backtest_execution(self, evaluator, sample_data, sample_signals):
        """Test basic backtest execution."""
        # Run backtest
        results = evaluator.run_backtest(sample_data, sample_signals, strategy_name="test")

        # Verify results structure
        assert isinstance(results, BacktestResult)
        assert results.num_trades >= 0
        assert results.total_return is not None
        assert results.sharpe_ratio is not None
        assert results.max_drawdown is not None
        assert results.win_rate is not None

        # Verify trade records
        assert len(evaluator.trades) == results.num_trades
        if len(evaluator.trades) > 0:
            assert isinstance(evaluator.trades[0], TradeRecord)

    def test_strategy_comparison(self, evaluator, sample_data):
        """Test strategy comparison functionality."""
        # Create multiple strategies
        strategies = {}

        # Momentum strategy
        returns = sample_data["close"].pct_change()
        momentum_signals = pd.Series(0, index=sample_data.index)
        momentum_signals[returns > returns.rolling(20).mean()] = 1
        momentum_signals[returns < returns.rolling(20).mean()] = -1
        strategies["momentum"] = momentum_signals

        # Mean reversion strategy
        sma_20 = sample_data["close"].rolling(20).mean()
        mean_rev_signals = pd.Series(0, index=sample_data.index)
        mean_rev_signals[sample_data["close"] < sma_20 * 0.95] = 1
        mean_rev_signals[sample_data["close"] > sma_20 * 1.05] = -1
        strategies["mean_reversion"] = mean_rev_signals

        # Compare strategies
        comparison_results = evaluator.compare_strategies(sample_data, strategies)

        assert len(comparison_results) == 2
        assert "momentum" in comparison_results
        assert "mean_reversion" in comparison_results

        for results in comparison_results.values():
            assert isinstance(results, BacktestResult)
            assert results.num_trades >= 0

    def test_performance_report_generation(self, evaluator, sample_data, sample_signals):
        """Test performance report generation."""
        # Run backtest
        results = evaluator.run_backtest(sample_data, sample_signals, strategy_name="test")

        # Generate report
        report = evaluator.generate_performance_report(results)

        assert isinstance(report, str)
        assert "Backtest Performance Report" in report
        assert "Total Return" in report
        assert "Sharpe Ratio" in report
        assert "Max Drawdown" in report

    def test_transaction_cost_impact(self, backtest_config, sample_data, sample_signals):
        """Test that transaction costs are properly accounted for."""
        # Test with different cost models
        low_cost_model = TransactionCostModel(
            commission_rate=0.0001,  # Very low costs
            slippage_rate=0.00001,
        )

        high_cost_model = TransactionCostModel(
            commission_rate=0.01,  # High costs
            slippage_rate=0.001,
        )

        # Run backtests with different cost models
        evaluator_low = BacktestEvaluator(backtest_config, low_cost_model)
        evaluator_high = BacktestEvaluator(backtest_config, high_cost_model)

        results_low = evaluator_low.run_backtest(sample_data, sample_signals, strategy_name="low_cost")
        results_high = evaluator_high.run_backtest(sample_data, sample_signals, strategy_name="high_cost")

        # High costs should result in lower total transaction costs (due to fewer trades)
        # but higher cost drag on returns
        assert results_high.total_transaction_costs > results_low.total_transaction_costs
        assert results_high.cost_drag > results_low.cost_drag


class TestTradeRecord:
    """Test TradeRecord functionality."""

    def test_trade_record_creation(self):
        """Test TradeRecord creation and conversion."""
        timestamp = datetime.now()
        costs = {
            "commission": 10.0,
            "slippage": 5.0,
            "market_impact": 2.0,
            "spread_cost": 1.0,
            "total_cost": 18.0,
            "cost_pct": 0.0018,
        }

        trade = TradeRecord(
            timestamp=timestamp,
            symbol="AAPL",
            side="buy",
            quantity=100,
            price=100.0,
            trade_value=10000.0,
            costs=costs,
            portfolio_value=100000.0,
            cash=90000.0,
            positions={"AAPL": 10000.0},
        )

        # Test conversion to dictionary
        trade_dict = trade.to_dict()
        assert trade_dict["symbol"] == "AAPL"
        assert trade_dict["side"] == "buy"
        assert trade_dict["quantity"] == 100
        assert trade_dict["price"] == 100.0
        assert trade_dict["commission"] == 10.0
        assert trade_dict["total_cost"] == 18.0


if __name__ == "__main__":
    pytest.main([__file__])
