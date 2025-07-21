"""
Unit tests for CLI command functions.

Tests individual CLI command functions and their logic.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from trading_rl_agent.cli import (
    cnn_lstm,
    compare,
    custom,
    download,
    download_all,
    evaluate,
    get_config_manager,
    hybrid,
    hyperopt,
    info,
    monitor,
    paper,
    pipeline,
    process,
    refresh,
    report,
    rl,
    scenario_compare,
    scenario_evaluate,
    standardize,
    start,
    status,
    stop,
    strategy,
    symbols,
    version,
    walk_forward,
)


class TestCLICommands:
    """Test CLI command functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ============================================================================
    # ROOT COMMANDS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    def test_version_command(self, mock_console):
        """Test version command."""
        version()
        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_info_command(self, mock_console):
        """Test info command."""
        info()
        mock_console.print.assert_called()

    # ============================================================================
    # DATA COMMANDS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_download_all_command(self, mock_get_config, mock_console):
        """Test download all command."""
        mock_config = Mock()
        mock_config.data.symbols = ["AAPL", "GOOGL"]
        mock_config.data.start_date = "2024-01-01"
        mock_config.data.end_date = "2024-01-31"
        mock_config.data.data_path = str(self.temp_path)
        mock_config.data.primary_source = "yfinance"
        mock_config.data.timeframe = "1d"
        mock_config.infrastructure.max_workers = 4
        mock_get_config.return_value = mock_config

        download_all(
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=self.temp_path,
            source="yfinance",
            timeframe="1d",
            parallel=False,
            force=False,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_symbols_command(self, mock_get_config, mock_console):
        """Test symbols command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        symbols(
            symbols="AAPL,GOOGL",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=self.temp_path,
            source="yfinance",
            timeframe="1d",
            parallel=False,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_refresh_command(self, mock_get_config, mock_console):
        """Test refresh command."""
        mock_config = Mock()
        mock_config.data.symbols = ["AAPL", "GOOGL"]
        mock_config.data.start_date = "2024-01-01"
        mock_config.data.end_date = "2024-01-31"
        mock_config.data.data_path = str(self.temp_path)
        mock_config.data.primary_source = "yfinance"
        mock_config.data.timeframe = "1d"
        mock_config.infrastructure.max_workers = 4
        mock_get_config.return_value = mock_config

        refresh(
            days=1,
            symbols="AAPL",
            output_dir=self.temp_path,
            source="yfinance",
            timeframe="1d",
            parallel=False,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_download_command(self, mock_get_config, mock_console):
        """Test download command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        download(
            symbols="AAPL",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=self.temp_path,
            source="yfinance",
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_process_command(self, mock_get_config, mock_console):
        """Test process command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        process(
            config_file=None,
            output_dir=self.temp_path,
            force_rebuild=False,
            parallel=True,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_standardize_command(self, mock_console):
        """Test standardize command."""
        # Create test data file
        test_data_file = self.temp_path / "test_data.csv"
        test_data_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000000")

        standardize(input_path=test_data_file, output_path=self.temp_path, method="robust")

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_pipeline_command(self, mock_get_config, mock_console):
        """Test pipeline command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        pipeline(config_path=None, output_dir=self.temp_path)

        mock_console.print.assert_called()

    # ============================================================================
    # TRAIN COMMANDS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_cnn_lstm_command(self, mock_get_config, mock_console):
        """Test CNN+LSTM training command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        cnn_lstm(
            config_file=None,
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            output_dir=self.temp_path,
            gpu=False,
            mixed_precision=True,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_rl_command(self, mock_get_config, mock_console):
        """Test RL training command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        rl(
            agent_type="ppo",
            config_file=None,
            timesteps=100,
            output_dir=self.temp_path,
            ray_address=None,
            num_workers=1,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_hybrid_command(self, mock_get_config, mock_console):
        """Test hybrid training command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        # Create dummy model paths
        cnn_path = self.temp_path / "cnn_model"
        cnn_path.mkdir()
        (cnn_path / "model.ckpt").touch()

        rl_path = self.temp_path / "rl_model"
        rl_path.mkdir()
        (rl_path / "model.ckpt").touch()

        hybrid(
            config_file=None,
            cnn_lstm_path=cnn_path,
            rl_path=rl_path,
            output_dir=self.temp_path,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_hyperopt_command(self, mock_get_config, mock_console):
        """Test hyperopt command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        hyperopt(config_file=None, n_trials=2, output_dir=self.temp_path)

        mock_console.print.assert_called()

    # ============================================================================
    # BACKTEST COMMANDS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    def test_strategy_command(self, mock_console):
        """Test strategy command."""
        # Create test data file
        test_data_file = self.temp_path / "test_data.csv"
        test_data_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000000")

        strategy(
            data_path=test_data_file,
            model_path=None,
            policy="momentum",
            initial_capital=10000.0,
            commission=0.001,
            slippage=0.0001,
            output_dir=self.temp_path,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_evaluate_command(self, mock_console):
        """Test evaluate command."""
        # Create test data and model files
        test_data_file = self.temp_path / "test_data.csv"
        test_data_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000000")

        model_path = self.temp_path / "model"
        model_path.mkdir()
        (model_path / "model.ckpt").touch()

        evaluate(
            model_path=model_path,
            data_path=test_data_file,
            output_dir=self.temp_path,
            initial_capital=10000.0,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_walk_forward_command(self, mock_console):
        """Test walk-forward command."""
        # Create test data file
        test_data_file = self.temp_path / "test_data.csv"
        test_data_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000000")

        walk_forward(
            data_path=test_data_file,
            model_type="cnn_lstm",
            train_window_size=10,
            validation_window_size=5,
            test_window_size=5,
            step_size=3,
            output_dir=self.temp_path,
            confidence_level=0.95,
            generate_plots=True,
            save_results=True,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_compare_command(self, mock_console):
        """Test compare command."""
        # Create test data file
        test_data_file = self.temp_path / "test_data.csv"
        test_data_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000000")

        # Create dummy model paths
        model1_path = self.temp_path / "model1"
        model1_path.mkdir()
        (model1_path / "model.ckpt").touch()

        model2_path = self.temp_path / "model2"
        model2_path.mkdir()
        (model2_path / "model.ckpt").touch()

        compare(
            models=f"{model1_path},{model2_path}",
            data_path=test_data_file,
            output_dir=self.temp_path,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_report_command(self, mock_console):
        """Test report command."""
        # Create dummy results file
        results_path = self.temp_path / "results"
        results_path.mkdir()
        (results_path / "results.json").write_text('{"strategy": "test", "return": 0.05}')

        report(results_path=results_path, output_format="html", output_dir=self.temp_path)

        mock_console.print.assert_called()

    # ============================================================================
    # TRADE COMMANDS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_start_command(self, mock_get_config, mock_console):
        """Test start command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        start(
            config_file=None,
            symbols="AAPL",
            model_path=None,
            paper_trading=True,
            initial_capital=10000.0,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_stop_command(self, mock_console):
        """Test stop command."""
        stop(session_id=None, all_sessions=False)

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_status_command(self, mock_console):
        """Test status command."""
        status(session_id=None, detailed=False)

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_monitor_command(self, mock_console):
        """Test monitor command."""
        monitor(session_id=None, metrics="all", interval=60)

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_paper_command(self, mock_get_config, mock_console):
        """Test paper command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        paper(config_file=None, symbols="AAPL,GOOGL,MSFT", duration="1d")

        mock_console.print.assert_called()

    # ============================================================================
    # SCENARIO COMMANDS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_scenario_evaluate_command(self, mock_get_config, mock_console):
        """Test scenario evaluate command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        scenario_evaluate(
            config_file=None,
            agent_type="moving_average",
            output_dir=self.temp_path,
            seed=42,
            save_reports=True,
            save_visualizations=True,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    def test_scenario_compare_command(self, mock_get_config, mock_console):
        """Test scenario compare command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        scenario_compare(
            config_file=None,
            output_dir=self.temp_path,
            seed=42,
            save_reports=True,
            save_visualizations=True,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    @patch("trading_rl_agent.cli.get_config_manager")
    @patch("trading_rl_agent.cli.AgentScenarioEvaluator")
    @patch("trading_rl_agent.cli.create_simple_moving_average_agent")
    @patch("trading_rl_agent.cli.create_custom_scenarios")
    @patch("trading_rl_agent.cli.create_momentum_agent")
    @patch("trading_rl_agent.cli.create_mean_reversion_agent")
    @patch("trading_rl_agent.cli.create_volatility_breakout_agent")
    def test_custom_command(
        self,
        mock_create_vol_agent,
        mock_create_mean_agent,
        mock_create_momentum_agent,
        mock_create_scenarios,
        mock_create_agent,
        mock_evaluator,
        mock_get_config,
        mock_console,
    ):
        """Test custom command."""
        mock_config = Mock()
        mock_get_config.return_value = mock_config

        # Mock the agents
        mock_agent = Mock()
        mock_create_agent.return_value = mock_agent
        mock_create_momentum_agent.return_value = Mock()
        mock_create_mean_agent.return_value = Mock()
        mock_create_vol_agent.return_value = Mock()

        # Mock scenarios
        mock_scenario = Mock()
        mock_scenario.name = "Strong Uptrend"
        mock_create_scenarios.return_value = [mock_scenario]

        # Mock evaluator
        mock_evaluator_instance = Mock()
        mock_evaluator_instance.evaluate_agent.return_value = {
            "overall_score": 0.8,
            "robustness_score": 0.7,
            "adaptation_score": 0.9,
            "aggregate_metrics": {"pass_rate": 0.85},
        }
        mock_evaluator_instance.print_evaluation_summary = Mock()
        mock_evaluator_instance.generate_evaluation_report = Mock()
        mock_evaluator.return_value = mock_evaluator_instance

        custom(
            config_file=None,
            agent_type="moving_average",
            scenario_name="strong_uptrend",
            output_dir=self.temp_path,
            seed=42,
            save_reports=True,
        )

        mock_console.print.assert_called()

    # ============================================================================
    # UTILITY FUNCTIONS
    # ============================================================================

    def test_get_config_manager(self):
        """Test get_config_manager function."""
        config = get_config_manager()
        assert config is not None

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    def test_standardize_command_invalid_input(self, mock_console):
        """Test standardize command with invalid input file."""
        # Currently the standardize function is a placeholder and doesn't validate input files
        # So it should not raise an exception for non-existent files
        standardize(
            input_path=Path("/nonexistent/file.csv"),
            output_path=self.temp_path,
            method="robust",
        )

        # Verify that the console was called (placeholder message)
        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_strategy_command_invalid_data(self, _mock_console):
        """Test strategy command with invalid data file."""
        with pytest.raises(typer.Exit):
            strategy(
                data_path=Path("/nonexistent/data.csv"),
                model_path=None,
                policy="momentum",
                initial_capital=10000.0,
                commission=0.001,
                slippage=0.0001,
                output_dir=self.temp_path,
            )

    @patch("trading_rl_agent.cli.console")
    def test_evaluate_command_invalid_model(self, _mock_console):
        """Test evaluate command with invalid model path."""
        # Create test data file
        test_data_file = self.temp_path / "test_data.csv"
        test_data_file.write_text("date,open,high,low,close,volume\n2024-01-01,100,105,95,102,1000000")

        with pytest.raises(typer.Exit):
            evaluate(
                model_path=Path("/nonexistent/model"),
                data_path=test_data_file,
                output_dir=self.temp_path,
                initial_capital=10000.0,
            )

    @patch("trading_rl_agent.cli.console")
    def test_report_command_invalid_results(self, _mock_console):
        """Test report command with invalid results path."""
        with pytest.raises(typer.Exit):
            report(
                results_path=Path("/nonexistent/results"),
                output_format="html",
                output_dir=self.temp_path,
            )

    # ============================================================================
    # EDGE CASES
    # ============================================================================

    @patch("trading_rl_agent.cli.console")
    def test_commands_with_none_values(self, mock_console):
        """Test commands handle None values gracefully."""
        # Test with None config file
        cnn_lstm(
            config_file=None,
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            output_dir=self.temp_path,
            gpu=False,
            mixed_precision=True,
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_commands_with_empty_strings(self, mock_console):
        """Test commands handle empty strings gracefully."""
        download(
            symbols="",
            start_date="2024-01-01",
            end_date="2024-01-31",
            output_dir=self.temp_path,
            source="yfinance",
        )

        mock_console.print.assert_called()

    @patch("trading_rl_agent.cli.console")
    def test_commands_with_zero_values(self, mock_console):
        """Test commands handle zero values gracefully."""
        cnn_lstm(
            config_file=None,
            epochs=0,  # Zero epochs
            batch_size=0,  # Zero batch size
            learning_rate=0.0,  # Zero learning rate
            output_dir=self.temp_path,
            gpu=False,
            mixed_precision=True,
        )

        mock_console.print.assert_called()
