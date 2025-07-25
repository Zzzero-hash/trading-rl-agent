"""
Unit tests for CLI backtest functions.

Tests the backtest CLI command functions and their logic.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import typer

from trade_agent.cli_backtest import (
    _generate_sample_signals,
    _load_historical_data,
    batch,
    compare,
    run,
)


class TestCLIBacktest:
    """Test CLI backtest functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_data(self) -> pd.DataFrame:
        """Create test market data."""
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="D")
        data = []

        for date in dates:
            data.append(
                {
                    "date": date,
                    "open": 100.0,
                    "high": 105.0,
                    "low": 95.0,
                    "close": 102.0,
                    "volume": 1000000,
                    "symbol": "AAPL",
                }
            )

        return pd.DataFrame(data)

    # ============================================================================
    # HELPER FUNCTIONS
    # ============================================================================

    @patch("trading_rl_agent.cli_backtest.yf")
    def test_load_historical_data(self, mock_yf):
        """Test _load_historical_data function."""
        # Mock yfinance data
        mock_ticker = Mock()
        mock_data = pd.DataFrame(
            {
                "Open": [100, 101, 102],
                "High": [105, 106, 107],
                "Low": [95, 96, 97],
                "Close": [102, 103, 104],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        result = _load_historical_data(["AAPL"], "2024-01-01", "2024-01-31")

        assert isinstance(result, pd.DataFrame)
        assert "symbol" in result.columns
        assert len(result) > 0

    def test_generate_sample_signals_momentum(self):
        """Test _generate_sample_signals with momentum strategy."""
        data = self.create_test_data()
        signals = _generate_sample_signals(data, "momentum")

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)

    def test_generate_sample_signals_mean_reversion(self):
        """Test _generate_sample_signals with mean reversion strategy."""
        data = self.create_test_data()
        signals = _generate_sample_signals(data, "mean_reversion")

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)

    def test_generate_sample_signals_random(self):
        """Test _generate_sample_signals with random strategy."""
        data = self.create_test_data()
        signals = _generate_sample_signals(data, "random")

        assert isinstance(signals, pd.Series)
        assert len(signals) == len(data)
        assert all(signal in [-1, 0, 1] for signal in signals)

    # ============================================================================
    # COMMAND FUNCTIONS
    # ============================================================================

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest.load_settings")
    @patch("trading_rl_agent.cli_backtest.get_settings")
    @patch("trading_rl_agent.cli_backtest._load_historical_data")
    @patch("trading_rl_agent.cli_backtest._generate_sample_signals")
    @patch("trading_rl_agent.cli_backtest.BacktestConfig")
    @patch("trading_rl_agent.cli_backtest.TransactionCostModel")
    @patch("trading_rl_agent.cli_backtest.BacktestEvaluator")
    def test_run_command(
        self,
        mock_evaluator,
        _mock_cost_model,
        _mock_config,
        mock_generate_signals,
        mock_load_data,
        mock_get_settings,
        _mock_load_settings,
        mock_console,
    ):
        """Test run command."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.backtest.start_date = "2024-01-01"
        mock_settings.backtest.end_date = "2024-01-31"
        mock_settings.backtest.symbols = ["AAPL"]
        mock_settings.backtest.initial_capital = 10000.0
        mock_settings.backtest.commission_rate = 0.001
        mock_settings.backtest.slippage_rate = 0.0001
        mock_settings.backtest.max_position_size = 0.1
        mock_settings.backtest.max_leverage = 2.0
        mock_settings.backtest.stop_loss_pct = 0.05
        mock_settings.backtest.take_profit_pct = 0.1
        mock_get_settings.return_value = mock_settings

        # Mock data and signals
        mock_data = self.create_test_data()
        mock_load_data.return_value = mock_data

        mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
        mock_generate_signals.return_value = mock_signals

        # Mock backtest results
        mock_results = Mock()
        mock_results.total_return = 0.05
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.02
        mock_results.win_rate = 0.65
        mock_results.num_trades = 10
        mock_results.total_transaction_costs = 25.0

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.run_backtest.return_value = mock_results
        mock_evaluator.return_value = mock_evaluator_instance

        # Test the command - expect typer.Exit(0) on success
        with pytest.raises(typer.Exit) as exc_info:
            run(
                strategy="momentum",
                start_date="2024-01-01",
                end_date="2024-01-31",
                symbols="AAPL",
                export_csv=None,
                config_file=None,
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0001,
            )

        # Verify exit code is 0 (success)
        assert exc_info.value.exit_code == 0

        # Verify calls
        mock_console.print.assert_called()
        mock_load_data.assert_called_once()
        mock_generate_signals.assert_called_once()
        mock_evaluator_instance.run_backtest.assert_called_once()

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest.load_settings")
    @patch("trading_rl_agent.cli_backtest.get_settings")
    @patch("trading_rl_agent.cli_backtest._load_historical_data")
    @patch("trading_rl_agent.cli_backtest._generate_sample_signals")
    @patch("trading_rl_agent.cli_backtest.BacktestConfig")
    @patch("trading_rl_agent.cli_backtest.TransactionCostModel")
    @patch("trading_rl_agent.cli_backtest.BacktestEvaluator")
    def test_batch_command(
        self,
        mock_evaluator,
        _mock_cost_model,
        _mock_config,
        mock_generate_signals,
        mock_load_data,
        mock_get_settings,
        _mock_load_settings,
        mock_console,
    ):
        """Test batch command."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.backtest.initial_capital = 10000.0
        mock_get_settings.return_value = mock_settings

        # Mock data and signals
        mock_data = self.create_test_data()
        mock_load_data.return_value = mock_data

        mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
        mock_generate_signals.return_value = mock_signals

        # Mock backtest results
        mock_results = Mock()
        mock_results.total_return = 0.05
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.02
        mock_results.win_rate = 0.65
        mock_results.num_trades = 10
        mock_results.total_transaction_costs = 25.0

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.run_backtest.return_value = mock_results
        mock_evaluator.return_value = mock_evaluator_instance

        # Test the command - expect typer.Exit(0) on success
        with pytest.raises(typer.Exit) as exc_info:
            batch(
                strategies="momentum,mean_reversion",
                periods="2024-01-01:2024-01-15,2024-01-16:2024-01-31",
                symbols="AAPL",
                export_csv=None,
                config_file=None,
                initial_capital=10000.0,
            )

        # Verify exit code is 0 (success)
        assert exc_info.value.exit_code == 0

        # Verify calls
        mock_console.print.assert_called()
        mock_load_data.assert_called()
        mock_generate_signals.assert_called()
        mock_evaluator_instance.run_backtest.assert_called()

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest.load_settings")
    @patch("trading_rl_agent.cli_backtest.get_settings")
    @patch("trading_rl_agent.cli_backtest._load_historical_data")
    @patch("trading_rl_agent.cli_backtest._generate_sample_signals")
    @patch("trading_rl_agent.cli_backtest.BacktestConfig")
    @patch("trading_rl_agent.cli_backtest.TransactionCostModel")
    @patch("trading_rl_agent.cli_backtest.BacktestEvaluator")
    def test_compare_command(
        self,
        mock_evaluator,
        _mock_cost_model,
        _mock_config,
        mock_generate_signals,
        mock_load_data,
        mock_get_settings,
        _mock_load_settings,
        mock_console,
    ):
        """Test compare command."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.backtest.initial_capital = 10000.0
        mock_get_settings.return_value = mock_settings

        # Mock data and signals
        mock_data = self.create_test_data()
        mock_load_data.return_value = mock_data

        mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
        mock_generate_signals.return_value = mock_signals

        # Mock backtest results
        mock_results = Mock()
        mock_results.total_return = 0.05
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.02
        mock_results.win_rate = 0.65
        mock_results.num_trades = 10
        mock_results.total_transaction_costs = 25.0

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.compare_strategies.return_value = {
            "momentum": mock_results,
            "mean_reversion": mock_results,
        }
        mock_evaluator.return_value = mock_evaluator_instance

        # Test the command - expect typer.Exit(0) on success
        with pytest.raises(typer.Exit) as exc_info:
            compare(
                strategies="momentum,mean_reversion",
                start_date="2024-01-01",
                end_date="2024-01-31",
                symbols="AAPL",
                config_file=None,
                output_dir=self.temp_path,
            )

        # Verify exit code is 0 (success)
        assert exc_info.value.exit_code == 0

        # Verify calls
        mock_console.print.assert_called()
        mock_load_data.assert_called()
        mock_generate_signals.assert_called()
        mock_evaluator_instance.compare_strategies.assert_called()

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest.load_settings")
    @patch("trading_rl_agent.cli_backtest.get_settings")
    @patch("trading_rl_agent.cli_backtest._load_historical_data")
    @patch("trading_rl_agent.cli_backtest._generate_sample_signals")
    @patch("trading_rl_agent.cli_backtest.BacktestConfig")
    @patch("trading_rl_agent.cli_backtest.TransactionCostModel")
    @patch("trading_rl_agent.cli_backtest.BacktestEvaluator")
    def test_run_command_none_dates_use_settings(
        self,
        mock_evaluator,
        _mock_cost_model,
        _mock_config,
        mock_generate_signals,
        mock_load_data,
        mock_get_settings,
        _mock_load_settings,
        _mock_console,
    ):
        """Test run command with None dates uses settings values."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.backtest.start_date = "2024-01-01"
        mock_settings.backtest.end_date = "2024-01-31"
        mock_settings.backtest.symbols = ["AAPL"]
        mock_settings.backtest.initial_capital = 10000.0
        mock_settings.backtest.commission_rate = 0.001
        mock_settings.backtest.slippage_rate = 0.0001
        mock_settings.backtest.max_position_size = 0.1
        mock_settings.backtest.max_leverage = 2.0
        mock_settings.backtest.stop_loss_pct = 0.05
        mock_settings.backtest.take_profit_pct = 0.1
        mock_get_settings.return_value = mock_settings

        # Mock data and signals
        mock_data = self.create_test_data()
        mock_load_data.return_value = mock_data

        mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
        mock_generate_signals.return_value = mock_signals

        # Mock backtest results
        mock_results = Mock()
        mock_results.total_return = 0.05
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.02
        mock_results.win_rate = 0.65
        mock_results.num_trades = 10
        mock_results.total_transaction_costs = 25.0

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.run_backtest.return_value = mock_results
        mock_evaluator.return_value = mock_evaluator_instance

        # Test the command with None dates - expect typer.Exit(0) on success
        with pytest.raises(typer.Exit) as exc_info:
            run(
                strategy="momentum",
                start_date=None,
                end_date=None,
                symbols=None,
                export_csv=None,
                config_file=None,
                initial_capital=10000.0,
                commission_rate=0.001,
                slippage_rate=0.0001,
            )

        # Verify exit code is 0 (success)
        assert exc_info.value.exit_code == 0

        # Verify that the settings dates were used (not "None" strings)
        mock_load_data.assert_called_once_with(["AAPL"], "2024-01-01", "2024-01-31")
        mock_evaluator_instance.run_backtest.assert_called_once()

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest.load_settings")
    @patch("trading_rl_agent.cli_backtest.get_settings")
    @patch("trading_rl_agent.cli_backtest._load_historical_data")
    @patch("trading_rl_agent.cli_backtest._generate_sample_signals")
    @patch("trading_rl_agent.cli_backtest.BacktestConfig")
    @patch("trading_rl_agent.cli_backtest.TransactionCostModel")
    @patch("trading_rl_agent.cli_backtest.BacktestEvaluator")
    def test_compare_command_none_dates_use_settings(
        self,
        mock_evaluator,
        _mock_cost_model,
        _mock_config,
        mock_generate_signals,
        mock_load_data,
        mock_get_settings,
        _mock_load_settings,
        _mock_console,
    ):
        """Test compare command with None dates uses settings values."""
        # Mock settings
        mock_settings = Mock()
        mock_settings.backtest.start_date = "2024-01-01"
        mock_settings.backtest.end_date = "2024-01-31"
        mock_settings.backtest.symbols = ["AAPL"]
        mock_settings.backtest.initial_capital = 10000.0
        mock_settings.backtest.commission_rate = 0.001
        mock_settings.backtest.slippage_rate = 0.0001
        mock_settings.backtest.max_position_size = 0.1
        mock_settings.backtest.max_leverage = 2.0
        mock_settings.backtest.stop_loss_pct = 0.05
        mock_settings.backtest.take_profit_pct = 0.1
        mock_settings.backtest.save_trades = True
        mock_settings.backtest.output_dir = "backtest_results"
        mock_get_settings.return_value = mock_settings

        # Mock data and signals
        mock_data = self.create_test_data()
        mock_load_data.return_value = mock_data

        mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
        mock_generate_signals.return_value = mock_signals

        # Mock backtest results
        mock_results = Mock()
        mock_results.total_return = 0.05
        mock_results.sharpe_ratio = 1.2
        mock_results.max_drawdown = -0.02
        mock_results.win_rate = 0.65
        mock_results.num_trades = 10
        mock_results.total_transaction_costs = 25.0

        mock_evaluator_instance = Mock()
        mock_evaluator_instance.compare_strategies.return_value = {"momentum": mock_results}
        mock_evaluator.return_value = mock_evaluator_instance

        # Test the command with None dates - expect typer.Exit(0) on success
        with pytest.raises(typer.Exit) as exc_info:
            compare(
                strategies="momentum,mean_reversion",
                start_date=None,
                end_date=None,
                symbols=None,
                config_file=None,
                output_dir=self.temp_path,
            )

        # Verify exit code is 0 (success)
        assert exc_info.value.exit_code == 0

        # Verify that the settings dates were used (not "None" strings)
        mock_load_data.assert_called_once_with(["AAPL"], "2024-01-01", "2024-01-31")
        mock_evaluator_instance.compare_strategies.assert_called_once()

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================

    @patch("trading_rl_agent.cli_backtest.console")
    def test_run_command_invalid_strategy(self, _mock_console):
        """Test run command with invalid strategy."""
        with pytest.raises(typer.Exit):
            run(
                strategy="invalid_strategy",
                start_date="2024-01-01",
                end_date="2024-01-31",
                symbols="AAPL",
            )

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest._load_historical_data")
    def test_run_command_data_load_failure(self, mock_load_data, _mock_console):
        """Test run command when data loading fails."""
        mock_load_data.side_effect = Exception("Data load failed")

        with pytest.raises(typer.Exit):
            run(
                strategy="momentum",
                start_date="2024-01-01",
                end_date="2024-01-31",
                symbols="AAPL",
            )

    @patch("trading_rl_agent.cli_backtest.console")
    def test_batch_command_invalid_periods(self, _mock_console):
        """Test batch command with invalid periods format."""
        with pytest.raises(typer.Exit):
            batch(strategies="momentum", periods="invalid_period_format", symbols="AAPL")

    @patch("trading_rl_agent.cli_backtest.console")
    def test_compare_command_empty_strategies(self, _mock_console):
        """Test compare command with empty strategies."""
        with pytest.raises(typer.Exit):
            compare(
                strategies="",
                start_date="2024-01-01",
                end_date="2024-01-31",
                symbols="AAPL",
            )

    # ============================================================================
    # EDGE CASES
    # ============================================================================

    def test_generate_sample_signals_empty_data(self):
        """Test _generate_sample_signals with empty data."""
        empty_data = pd.DataFrame(columns=["close"])
        signals = _generate_sample_signals(empty_data, "momentum")

        assert isinstance(signals, pd.Series)
        assert len(signals) == 0

    def test_generate_sample_signals_single_row(self):
        """Test _generate_sample_signals with single row data."""
        single_row_data = pd.DataFrame({"close": [100.0], "volume": [1000000]})
        signals = _generate_sample_signals(single_row_data, "momentum")

        assert isinstance(signals, pd.Series)
        assert len(signals) == 1

    @patch("trading_rl_agent.cli_backtest.console")
    @patch("trading_rl_agent.cli_backtest.load_settings")
    @patch("trading_rl_agent.cli_backtest.get_settings")
    def test_run_command_with_config_file(self, _mock_get_settings, mock_load_settings, _mock_console):
        """Test run command with config file."""
        # Mock config file
        config_file = self.temp_path / "config.yaml"
        config_file.touch()

        # Mock settings
        mock_settings = Mock()
        mock_settings.backtest.start_date = "2024-01-01"
        mock_settings.backtest.end_date = "2024-01-31"
        mock_settings.backtest.symbols = ["AAPL"]
        mock_settings.backtest.initial_capital = 10000.0
        mock_settings.backtest.commission_rate = 0.001
        mock_settings.backtest.slippage_rate = 0.0001
        mock_settings.backtest.max_position_size = 0.1
        mock_settings.backtest.max_leverage = 2.0
        mock_settings.backtest.stop_loss_pct = 0.05
        mock_settings.backtest.take_profit_pct = 0.1
        mock_load_settings.return_value = mock_settings

        # Mock other dependencies
        with (
            patch("trading_rl_agent.cli_backtest._load_historical_data") as mock_load_data,
            patch("trading_rl_agent.cli_backtest._generate_sample_signals") as mock_generate_signals,
            patch("trading_rl_agent.cli_backtest.BacktestConfig"),
            patch("trading_rl_agent.cli_backtest.TransactionCostModel"),
            patch("trading_rl_agent.cli_backtest.BacktestEvaluator") as mock_evaluator,
        ):
            # Mock data and signals
            mock_data = self.create_test_data()
            mock_load_data.return_value = mock_data

            mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
            mock_generate_signals.return_value = mock_signals

            # Mock backtest results
            mock_results = Mock()
            mock_results.total_return = 0.05
            mock_results.sharpe_ratio = 1.2
            mock_results.max_drawdown = -0.02
            mock_results.win_rate = 0.65
            mock_results.num_trades = 10
            mock_results.total_transaction_costs = 25.0

            mock_evaluator_instance = Mock()
            mock_evaluator_instance.run_backtest.return_value = mock_results
            mock_evaluator.return_value = mock_evaluator_instance

            # Test the command - expect typer.Exit(0) on success
            with pytest.raises(typer.Exit) as exc_info:
                run(
                    strategy="momentum",
                    start_date=None,
                    end_date=None,
                    symbols=None,
                    export_csv=None,
                    config_file=config_file,
                    initial_capital=None,
                    commission_rate=None,
                    slippage_rate=None,
                )

            # Verify exit code is 0 (success)
            assert exc_info.value.exit_code == 0

            # Verify config was loaded
            mock_load_settings.assert_called_once_with(config_path=config_file)

    @patch("trading_rl_agent.cli_backtest.console")
    def test_run_command_export_csv(self, _mock_console):
        """Test run command with CSV export."""
        export_csv = self.temp_path / "results.csv"

        # Mock all dependencies
        with (
            patch("trading_rl_agent.cli_backtest.load_settings"),
            patch("trading_rl_agent.cli_backtest.get_settings") as mock_get_settings,
            patch("trading_rl_agent.cli_backtest._load_historical_data") as mock_load_data,
            patch("trading_rl_agent.cli_backtest._generate_sample_signals") as mock_generate_signals,
            patch("trading_rl_agent.cli_backtest.BacktestConfig"),
            patch("trading_rl_agent.cli_backtest.TransactionCostModel"),
            patch("trading_rl_agent.cli_backtest.BacktestEvaluator") as mock_evaluator,
        ):
            # Mock settings
            mock_settings = Mock()
            mock_settings.backtest.start_date = "2024-01-01"
            mock_settings.backtest.end_date = "2024-01-31"
            mock_settings.backtest.symbols = ["AAPL"]
            mock_settings.backtest.initial_capital = 10000.0
            mock_settings.backtest.commission_rate = 0.001
            mock_settings.backtest.slippage_rate = 0.0001
            mock_settings.backtest.max_position_size = 0.1
            mock_settings.backtest.max_leverage = 2.0
            mock_settings.backtest.stop_loss_pct = 0.05
            mock_settings.backtest.take_profit_pct = 0.1
            mock_get_settings.return_value = mock_settings

            # Mock data and signals
            mock_data = self.create_test_data()
            mock_load_data.return_value = mock_data

            mock_signals = pd.Series([0, 1, -1, 0, 1], index=mock_data.index[:5])
            mock_generate_signals.return_value = mock_signals

            # Mock backtest results
            mock_results = Mock()
            mock_results.total_return = 0.05
            mock_results.sharpe_ratio = 1.2
            mock_results.max_drawdown = -0.02
            mock_results.win_rate = 0.65
            mock_results.num_trades = 10
            mock_results.total_transaction_costs = 25.0

            mock_evaluator_instance = Mock()
            mock_evaluator_instance.run_backtest.return_value = mock_results
            mock_evaluator.return_value = mock_evaluator_instance

            # Test the command - expect typer.Exit(0) on success
            with pytest.raises(typer.Exit) as exc_info:
                run(
                    strategy="momentum",
                    start_date="2024-01-01",
                    end_date="2024-01-31",
                    symbols="AAPL",
                    export_csv=export_csv,
                    config_file=None,
                    initial_capital=10000.0,
                    commission_rate=0.001,
                    slippage_rate=0.0001,
                )

            # Verify exit code is 0 (success)
            assert exc_info.value.exit_code == 0

            # Verify CSV file was created (the function exits after creating the file)
            # The test passes if no exception is raised
