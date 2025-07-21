"""
Integration Tests for CLI Workflows.

This module provides comprehensive integration testing for:
- Complete trading workflows
- Data pipeline integration
- Model training and evaluation
- Risk management integration
- End-to-end user scenarios
- Performance under realistic conditions
"""

import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trading_rl_agent.cli import app as main_app
from trading_rl_agent.cli_health import app as health_app


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows through CLI."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.workflow_dir = Path(self.temp_dir) / "workflow"
        self.workflow_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.download_all")
    @patch("trading_rl_agent.cli.process_data")
    @patch("trading_rl_agent.cli.standardize_data")
    @patch("trading_rl_agent.cli.build_pipeline")
    @patch("trading_rl_agent.cli.train_cnn_lstm")
    @patch("trading_rl_agent.cli.train_rl_agent")
    @patch("trading_rl_agent.cli.train_hybrid_model")
    @patch("trading_rl_agent.cli.run_backtest_strategy")
    @patch("trading_rl_agent.cli.evaluate_model")
    @patch("trading_rl_agent.cli.start_trading")
    def test_complete_trading_workflow(
        self,
        mock_start_trading,
        mock_evaluate,
        mock_backtest,
        mock_hybrid,
        mock_rl,
        mock_cnn_lstm,
        mock_pipeline,
        mock_standardize,
        mock_process,
        mock_download,
    ):
        """Test complete trading workflow from data to live trading."""
        # Setup mocks
        mock_download.return_value = None
        mock_process.return_value = None
        mock_standardize.return_value = None
        mock_pipeline.return_value = None
        mock_cnn_lstm.return_value = None
        mock_rl.return_value = None
        mock_hybrid.return_value = None
        mock_backtest.return_value = None
        mock_evaluate.return_value = None
        mock_start_trading.return_value = None

        # Step 1: Data Pipeline
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--output-dir",
                str(self.workflow_dir / "data"),
                "--source",
                "yfinance",
                "--timeframe",
                "1d",
                "--parallel",
            ],
        )
        assert result.exit_code == 0
        mock_download.assert_called_once()

        result = self.runner.invoke(
            main_app,
            [
                "data",
                "process",
                "--output-dir",
                str(self.workflow_dir / "processed"),
                "--force-rebuild",
                "--parallel",
            ],
        )
        assert result.exit_code == 0
        mock_process.assert_called_once()

        result = self.runner.invoke(
            main_app,
            [
                "data",
                "standardize",
                "--input-path",
                str(self.workflow_dir / "processed"),
                "--output-path",
                str(self.workflow_dir / "standardized"),
                "--method",
                "robust",
            ],
        )
        assert result.exit_code == 0
        mock_standardize.assert_called_once()

        result = self.runner.invoke(
            main_app,
            [
                "data",
                "pipeline",
                "--config-path",
                str(self.workflow_dir),
                "--output-dir",
                str(self.workflow_dir / "pipeline"),
            ],
        )
        assert result.exit_code == 0
        mock_pipeline.assert_called_once()

        # Step 2: Model Training
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "cnn-lstm",
                "--epochs",
                "5",
                "--batch-size",
                "32",
                "--learning-rate",
                "0.001",
                "--output-dir",
                str(self.workflow_dir / "models" / "cnn_lstm"),
                "--gpu",
                "--mixed-precision",
            ],
        )
        assert result.exit_code == 0
        mock_cnn_lstm.assert_called_once()

        result = self.runner.invoke(
            main_app,
            [
                "train",
                "rl",
                "--agent-type",
                "ppo",
                "--timesteps",
                "10000",
                "--output-dir",
                str(self.workflow_dir / "models" / "rl"),
                "--ray-address",
                "auto",
                "--num-workers",
                "2",
            ],
        )
        assert result.exit_code == 0
        mock_rl.assert_called_once()

        result = self.runner.invoke(
            main_app,
            [
                "train",
                "hybrid",
                "--cnn-lstm-path",
                str(self.workflow_dir / "models" / "cnn_lstm"),
                "--rl-path",
                str(self.workflow_dir / "models" / "rl"),
                "--output-dir",
                str(self.workflow_dir / "models" / "hybrid"),
            ],
        )
        assert result.exit_code == 0
        mock_hybrid.assert_called_once()

        # Step 3: Backtesting and Evaluation
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "strategy",
                "--data-path",
                str(self.workflow_dir / "pipeline"),
                "--model-path",
                str(self.workflow_dir / "models" / "hybrid"),
                "--policy",
                "ppo",
                "--initial-capital",
                "10000",
                "--commission",
                "0.001",
                "--slippage",
                "0.0001",
                "--output-dir",
                str(self.workflow_dir / "backtest"),
            ],
        )
        assert result.exit_code == 0
        mock_backtest.assert_called_once()

        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "evaluate",
                "--model-path",
                str(self.workflow_dir / "models" / "hybrid"),
                "--data-path",
                str(self.workflow_dir / "pipeline"),
                "--output-dir",
                str(self.workflow_dir / "evaluation"),
                "--initial-capital",
                "10000",
            ],
        )
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()

        # Step 4: Live Trading
        result = self.runner.invoke(
            main_app,
            [
                "trade",
                "start",
                "--symbols",
                "AAPL,GOOGL",
                "--model-path",
                str(self.workflow_dir / "models" / "hybrid"),
                "--paper-trading",
                "--initial-capital",
                "100000",
            ],
        )
        assert result.exit_code == 0
        mock_start_trading.assert_called_once()

    @patch("trading_rl_agent.cli.download_all")
    @patch("trading_rl_agent.cli.process_data")
    @patch("trading_rl_agent.cli.train_cnn_lstm")
    @patch("trading_rl_agent.cli.run_walk_forward")
    @patch("trading_rl_agent.cli.compare_models")
    @patch("trading_rl_agent.cli.generate_report")
    def test_research_workflow(
        self,
        mock_report,
        mock_compare,
        mock_walk_forward,
        mock_train,
        mock_process,
        mock_download,
    ):
        """Test research workflow with walk-forward analysis."""
        # Setup mocks
        mock_download.return_value = None
        mock_process.return_value = None
        mock_train.return_value = None
        mock_walk_forward.return_value = None
        mock_compare.return_value = None
        mock_report.return_value = None

        # Data preparation
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2022-01-01",
                "--end-date",
                "2023-12-31",
                "--output-dir",
                str(self.workflow_dir / "research_data"),
            ],
        )
        assert result.exit_code == 0

        result = self.runner.invoke(
            main_app,
            [
                "data",
                "process",
                "--output-dir",
                str(self.workflow_dir / "research_processed"),
            ],
        )
        assert result.exit_code == 0

        # Model training
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "cnn-lstm",
                "--epochs",
                "3",
                "--output-dir",
                str(self.workflow_dir / "research_models"),
            ],
        )
        assert result.exit_code == 0

        # Walk-forward analysis
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "walk-forward",
                "--data-path",
                str(self.workflow_dir / "research_processed"),
                "--model-type",
                "cnn_lstm",
                "--train-window-size",
                "252",
                "--validation-window-size",
                "63",
                "--test-window-size",
                "63",
                "--step-size",
                "21",
                "--output-dir",
                str(self.workflow_dir / "walk_forward"),
                "--confidence-level",
                "0.95",
                "--generate-plots",
                "--save-results",
            ],
        )
        assert result.exit_code == 0

        # Model comparison
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "compare",
                "--models",
                "model1,model2,model3",
                "--data-path",
                str(self.workflow_dir / "research_processed"),
                "--output-dir",
                str(self.workflow_dir / "comparison"),
            ],
        )
        assert result.exit_code == 0

        # Report generation
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "report",
                "--results-path",
                str(self.workflow_dir / "walk_forward"),
                "--output-format",
                "html",
                "--output-dir",
                str(self.workflow_dir / "reports"),
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.evaluate_scenario")
    @patch("trading_rl_agent.cli.compare_scenarios")
    @patch("trading_rl_agent.cli.run_custom_scenario")
    def test_scenario_analysis_workflow(self, mock_custom, mock_compare, mock_evaluate):
        """Test scenario analysis workflow."""
        # Setup mocks
        mock_evaluate.return_value = None
        mock_compare.return_value = None
        mock_custom.return_value = None

        # Scenario evaluation
        result = self.runner.invoke(
            main_app,
            [
                "scenario",
                "scenario-evaluate",
                "--agent-type",
                "moving_average",
                "--output-dir",
                str(self.workflow_dir / "scenarios"),
                "--seed",
                "42",
                "--save-reports",
                "--save-visualizations",
            ],
        )
        assert result.exit_code == 0

        # Scenario comparison
        result = self.runner.invoke(
            main_app,
            [
                "scenario",
                "scenario-compare",
                "--output-dir",
                str(self.workflow_dir / "scenario_comparison"),
                "--seed",
                "42",
                "--save-reports",
                "--save-visualizations",
            ],
        )
        assert result.exit_code == 0

        # Custom scenario
        result = self.runner.invoke(
            main_app,
            [
                "scenario",
                "custom",
                "--agent-type",
                "moving_average",
                "--scenario-name",
                "strong_uptrend",
                "--output-dir",
                str(self.workflow_dir / "custom_scenarios"),
                "--seed",
                "42",
                "--save-reports",
            ],
        )
        assert result.exit_code == 0


class TestDataPipelineIntegration:
    """Test data pipeline integration workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir) / "data_pipeline"
        self.data_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.download_all")
    @patch("trading_rl_agent.cli.refresh_data")
    @patch("trading_rl_agent.cli.process_data")
    @patch("trading_rl_agent.cli.standardize_data")
    def test_data_refresh_workflow(self, mock_standardize, mock_process, mock_refresh, mock_download):
        """Test data refresh workflow."""
        # Setup mocks
        mock_download.return_value = None
        mock_refresh.return_value = None
        mock_process.return_value = None
        mock_standardize.return_value = None

        # Initial data download
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--output-dir",
                str(self.data_dir / "raw"),
                "--source",
                "yfinance",
            ],
        )
        assert result.exit_code == 0

        # Data refresh
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "refresh",
                "--days",
                "7",
                "--symbols",
                "AAPL,GOOGL,MSFT",
                "--output-dir",
                str(self.data_dir / "raw"),
            ],
        )
        assert result.exit_code == 0

        # Process updated data
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "process",
                "--output-dir",
                str(self.data_dir / "processed"),
                "--force-rebuild",
            ],
        )
        assert result.exit_code == 0

        # Standardize updated data
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "standardize",
                "--input-path",
                str(self.data_dir / "processed"),
                "--output-path",
                str(self.data_dir / "standardized"),
                "--method",
                "robust",
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.download_symbols")
    @patch("trading_rl_agent.cli.process_data")
    def test_multi_symbol_workflow(self, mock_process, mock_download):
        """Test multi-symbol data workflow."""
        # Setup mocks
        mock_download.return_value = None
        mock_process.return_value = None

        # Download specific symbols
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "symbols",
                "--symbols",
                "AAPL,GOOGL,MSFT,TSLA,NVDA",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
                "--output-dir",
                str(self.data_dir / "multi_symbol"),
                "--source",
                "yfinance",
                "--timeframe",
                "1d",
                "--parallel",
            ],
        )
        assert result.exit_code == 0

        # Process multi-symbol data
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "process",
                "--output-dir",
                str(self.data_dir / "multi_symbol_processed"),
                "--force-rebuild",
                "--parallel",
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.build_pipeline")
    def test_pipeline_building_workflow(self, mock_pipeline):
        """Test pipeline building workflow."""
        # Setup mock
        mock_pipeline.return_value = None

        # Create config for pipeline
        config_content = """
        data:
          source: "yfinance"
          symbols: ["AAPL", "GOOGL", "MSFT"]
          start_date: "2023-01-01"
          end_date: "2023-12-31"
        features:
          technical_indicators: true
          fundamental_data: false
        """
        config_file = self.data_dir / "pipeline_config.yaml"
        config_file.write_text(config_content)

        # Build pipeline
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "pipeline",
                "--config-path",
                str(config_file),
                "--output-dir",
                str(self.data_dir / "pipeline_output"),
            ],
        )
        assert result.exit_code == 0


class TestModelTrainingIntegration:
    """Test model training integration workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    @patch("trading_rl_agent.cli.train_rl_agent")
    @patch("trading_rl_agent.cli.train_hybrid_model")
    @patch("trading_rl_agent.cli.run_hyperopt")
    def test_model_training_workflow(self, mock_hyperopt, mock_hybrid, mock_rl, mock_cnn_lstm):
        """Test complete model training workflow."""
        # Setup mocks
        mock_cnn_lstm.return_value = None
        mock_rl.return_value = None
        mock_hybrid.return_value = None
        mock_hyperopt.return_value = None

        # CNN+LSTM training
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "cnn-lstm",
                "--epochs",
                "5",
                "--batch-size",
                "32",
                "--learning-rate",
                "0.001",
                "--output-dir",
                str(self.models_dir / "cnn_lstm"),
                "--gpu",
                "--mixed-precision",
            ],
        )
        assert result.exit_code == 0

        # RL agent training
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "rl",
                "--agent-type",
                "ppo",
                "--timesteps",
                "10000",
                "--output-dir",
                str(self.models_dir / "rl"),
                "--ray-address",
                "auto",
                "--num-workers",
                "2",
            ],
        )
        assert result.exit_code == 0

        # Hybrid model training
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "hybrid",
                "--cnn-lstm-path",
                str(self.models_dir / "cnn_lstm"),
                "--rl-path",
                str(self.models_dir / "rl"),
                "--output-dir",
                str(self.models_dir / "hybrid"),
            ],
        )
        assert result.exit_code == 0

        # Hyperparameter optimization
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "hyperopt",
                "--n-trials",
                "10",
                "--output-dir",
                str(self.models_dir / "optimization"),
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    def test_cnn_lstm_training_variations(self, mock_train):
        """Test CNN+LSTM training with different configurations."""
        # Setup mock
        mock_train.return_value = None

        # Test different batch sizes
        for batch_size in [16, 32, 64]:
            result = self.runner.invoke(
                main_app,
                [
                    "train",
                    "cnn-lstm",
                    "--epochs",
                    "2",
                    "--batch-size",
                    str(batch_size),
                    "--output-dir",
                    str(self.models_dir / f"cnn_lstm_batch_{batch_size}"),
                ],
            )
            assert result.exit_code == 0

        # Test different learning rates
        for lr in [0.0001, 0.001, 0.01]:
            result = self.runner.invoke(
                main_app,
                [
                    "train",
                    "cnn-lstm",
                    "--epochs",
                    "2",
                    "--learning-rate",
                    str(lr),
                    "--output-dir",
                    str(self.models_dir / f"cnn_lstm_lr_{lr}"),
                ],
            )
            assert result.exit_code == 0

    @patch("trading_rl_agent.cli.train_rl_agent")
    def test_rl_training_variations(self, mock_train):
        """Test RL training with different agent types."""
        # Setup mock
        mock_train.return_value = None

        # Test different agent types
        for agent_type in ["ppo", "a2c", "dqn"]:
            result = self.runner.invoke(
                main_app,
                [
                    "train",
                    "rl",
                    "--agent-type",
                    agent_type,
                    "--timesteps",
                    "5000",
                    "--output-dir",
                    str(self.models_dir / f"rl_{agent_type}"),
                ],
            )
            assert result.exit_code == 0


class TestTradingSessionIntegration:
    """Test trading session integration workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.trading_dir = Path(self.temp_dir) / "trading"
        self.trading_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.start_trading")
    @patch("trading_rl_agent.cli.get_trading_status")
    @patch("trading_rl_agent.cli.monitor_trading")
    @patch("trading_rl_agent.cli.stop_trading")
    def test_trading_session_workflow(self, mock_stop, mock_monitor, mock_status, mock_start):
        """Test complete trading session workflow."""
        # Setup mocks
        mock_start.return_value = None
        mock_status.return_value = {"status": "running", "sessions": 1}
        mock_monitor.return_value = None
        mock_stop.return_value = None

        # Start trading session
        result = self.runner.invoke(
            main_app,
            [
                "trade",
                "start",
                "--symbols",
                "AAPL,GOOGL",
                "--model-path",
                str(self.trading_dir / "model"),
                "--paper-trading",
                "--initial-capital",
                "100000",
            ],
        )
        assert result.exit_code == 0

        # Check trading status
        result = self.runner.invoke(main_app, ["trade", "status", "--detailed"])
        assert result.exit_code == 0

        # Monitor trading (async operation)
        result = self.runner.invoke(main_app, ["trade", "monitor", "--metrics", "all", "--interval", "30"])
        assert result.exit_code == 0

        # Stop trading session
        result = self.runner.invoke(main_app, ["trade", "stop", "--all-sessions"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.start_paper_trading")
    def test_paper_trading_workflow(self, mock_paper):
        """Test paper trading workflow."""
        # Setup mock
        mock_paper.return_value = None

        # Start paper trading
        result = self.runner.invoke(
            main_app,
            ["trade", "paper", "--symbols", "AAPL,GOOGL,MSFT,TSLA", "--duration", "1d"],
        )
        assert result.exit_code == 0

        # Test different durations
        for duration in ["1h", "4h", "1d", "1w"]:
            result = self.runner.invoke(
                main_app,
                ["trade", "paper", "--symbols", "AAPL", "--duration", duration],
            )
            assert result.exit_code == 0


class TestBacktestIntegration:
    """Test backtesting integration workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.backtest_dir = Path(self.temp_dir) / "backtest"
        self.backtest_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.run_backtest_strategy")
    @patch("trading_rl_agent.cli.evaluate_model")
    @patch("trading_rl_agent.cli.run_walk_forward")
    @patch("trading_rl_agent.cli.compare_models")
    @patch("trading_rl_agent.cli.generate_report")
    def test_comprehensive_backtest_workflow(
        self, mock_report, mock_compare, mock_walk_forward, mock_evaluate, mock_backtest
    ):
        """Test comprehensive backtesting workflow."""
        # Setup mocks
        mock_backtest.return_value = None
        mock_evaluate.return_value = None
        mock_walk_forward.return_value = None
        mock_compare.return_value = None
        mock_report.return_value = None

        # Strategy backtesting
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "strategy",
                "--data-path",
                str(self.backtest_dir / "data"),
                "--model-path",
                str(self.backtest_dir / "model"),
                "--policy",
                "ppo",
                "--initial-capital",
                "10000",
                "--commission",
                "0.001",
                "--slippage",
                "0.0001",
                "--output-dir",
                str(self.backtest_dir / "strategy_results"),
            ],
        )
        assert result.exit_code == 0

        # Model evaluation
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "evaluate",
                "--model-path",
                str(self.backtest_dir / "model"),
                "--data-path",
                str(self.backtest_dir / "data"),
                "--output-dir",
                str(self.backtest_dir / "evaluation_results"),
                "--initial-capital",
                "10000",
            ],
        )
        assert result.exit_code == 0

        # Walk-forward analysis
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "walk-forward",
                "--data-path",
                str(self.backtest_dir / "data"),
                "--model-type",
                "cnn_lstm",
                "--train-window-size",
                "252",
                "--validation-window-size",
                "63",
                "--test-window-size",
                "63",
                "--step-size",
                "21",
                "--output-dir",
                str(self.backtest_dir / "walk_forward_results"),
                "--confidence-level",
                "0.95",
                "--generate-plots",
                "--save-results",
            ],
        )
        assert result.exit_code == 0

        # Model comparison
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "compare",
                "--models",
                "model1,model2,model3",
                "--data-path",
                str(self.backtest_dir / "data"),
                "--output-dir",
                str(self.backtest_dir / "comparison_results"),
            ],
        )
        assert result.exit_code == 0

        # Report generation
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "report",
                "--results-path",
                str(self.backtest_dir / "strategy_results"),
                "--output-format",
                "html",
                "--output-dir",
                str(self.backtest_dir / "reports"),
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.run_backtest_strategy")
    def test_backtest_parameter_variations(self, mock_backtest):
        """Test backtesting with different parameters."""
        # Setup mock
        mock_backtest.return_value = None

        # Test different initial capitals
        for capital in [5000, 10000, 50000, 100000]:
            result = self.runner.invoke(
                main_app,
                [
                    "backtest",
                    "strategy",
                    "--data-path",
                    str(self.backtest_dir / "data"),
                    "--model-path",
                    str(self.backtest_dir / "model"),
                    "--initial-capital",
                    str(capital),
                    "--output-dir",
                    str(self.backtest_dir / f"capital_{capital}"),
                ],
            )
            assert result.exit_code == 0

        # Test different commission rates
        for commission in [0.0005, 0.001, 0.002, 0.005]:
            result = self.runner.invoke(
                main_app,
                [
                    "backtest",
                    "strategy",
                    "--data-path",
                    str(self.backtest_dir / "data"),
                    "--model-path",
                    str(self.backtest_dir / "model"),
                    "--commission",
                    str(commission),
                    "--output-dir",
                    str(self.backtest_dir / f"commission_{commission}"),
                ],
            )
            assert result.exit_code == 0


class TestHealthMonitoringIntegration:
    """Test health monitoring integration workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.health_dir = Path(self.temp_dir) / "health"
        self.health_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli_health.monitor")
    def test_health_monitoring_workflow(self, mock_monitor):
        """Test health monitoring workflow."""
        # Setup mock
        mock_monitor.return_value = None

        # Start health monitoring
        result = self.runner.invoke(
            health_app,
            [
                "monitor",
                "--duration",
                "60",
                "--interval",
                "10",
                "--output",
                str(self.health_dir),
                "--live",
                "--html",
                "--json",
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli_health.status")
    def test_health_status_workflow(self, mock_status):
        """Test health status workflow."""
        # Setup mock
        mock_status.return_value = None

        # Check system health status
        result = self.runner.invoke(health_app, ["status"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli_health.report")
    def test_health_report_workflow(self, mock_report):
        """Test health report workflow."""
        # Setup mock
        mock_report.return_value = None

        # Generate health report
        result = self.runner.invoke(
            health_app,
            [
                "report",
                "--output",
                str(self.health_dir / "health_report.html"),
                "--format",
                "html",
            ],
        )
        assert result.exit_code == 0

        # Generate JSON report
        result = self.runner.invoke(
            health_app,
            [
                "report",
                "--output",
                str(self.health_dir / "health_report.json"),
                "--format",
                "json",
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli_health.alerts")
    def test_health_alerts_workflow(self, mock_alerts):
        """Test health alerts workflow."""
        # Setup mock
        mock_alerts.return_value = None

        # Check alerts
        result = self.runner.invoke(health_app, ["alerts", "--severity", "error", "--limit", "5"])
        assert result.exit_code == 0


class TestPerformanceUnderLoad:
    """Test CLI performance under realistic load conditions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.download_all")
    def test_large_dataset_performance(self, mock_download):
        """Test performance with large datasets."""
        # Setup mock
        mock_download.return_value = None

        start_time = time.time()
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2023-12-31",
                "--output-dir",
                str(self.temp_dir),
                "--parallel",
            ],
        )
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 10.0  # Should complete within 10 seconds

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    def test_training_performance(self, mock_train):
        """Test training performance."""
        # Setup mock
        mock_train.return_value = None

        start_time = time.time()
        result = self.runner.invoke(
            main_app,
            [
                "train",
                "cnn-lstm",
                "--epochs",
                "10",
                "--batch-size",
                "64",
                "--output-dir",
                str(self.temp_dir),
            ],
        )
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 5.0  # Should complete within 5 seconds

    @patch("trading_rl_agent.cli.run_backtest_strategy")
    def test_backtest_performance(self, mock_backtest):
        """Test backtesting performance."""
        # Setup mock
        mock_backtest.return_value = None

        start_time = time.time()
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "strategy",
                "--data-path",
                str(self.temp_dir),
                "--model-path",
                str(self.temp_dir),
                "--output-dir",
                str(self.temp_dir),
            ],
        )
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 5.0  # Should complete within 5 seconds

    def test_concurrent_operations(self):
        """Test performance under concurrent operations."""
        import threading
        import time

        results = []
        errors = []

        def run_command(cmd_args):
            try:
                start_time = time.time()
                result = self.runner.invoke(main_app, cmd_args)
                end_time = time.time()
                results.append((result.exit_code, end_time - start_time))
            except Exception as e:
                errors.append(e)

        # Run multiple commands concurrently
        threads = []
        commands = [
            ["--help"],
            ["version"],
            ["info"],
            ["data", "--help"],
            ["train", "--help"],
            ["backtest", "--help"],
            ["trade", "--help"],
        ]

        for cmd in commands:
            thread = threading.Thread(target=run_command, args=(cmd,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == len(commands)

        # All commands should succeed
        for exit_code, duration in results:
            assert exit_code == 0
            assert duration < 2.0  # Each command should complete within 2 seconds


class TestErrorRecoveryScenarios:
    """Test error recovery scenarios in integration workflows."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.download_all")
    @patch("trading_rl_agent.cli.process_data")
    def test_recovery_from_data_failure(self, mock_process, mock_download):
        """Test recovery from data download failure."""
        # First call fails, second succeeds
        mock_download.side_effect = [Exception("Network error"), None]
        mock_process.return_value = None

        # First attempt fails
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
        )
        assert result.exit_code != 0

        # Second attempt succeeds
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-12-31",
            ],
        )
        assert result.exit_code == 0

        # Processing should still work
        result = self.runner.invoke(main_app, ["data", "process", "--output-dir", str(self.temp_dir)])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    @patch("trading_rl_agent.cli.run_backtest_strategy")
    def test_recovery_from_training_failure(self, mock_backtest, mock_train):
        """Test recovery from training failure."""
        # Training fails first, then succeeds
        mock_train.side_effect = [MemoryError("Out of memory"), None]
        mock_backtest.return_value = None

        # First training attempt fails
        result = self.runner.invoke(main_app, ["train", "cnn-lstm", "--epochs", "10"])
        assert result.exit_code != 0

        # Second training attempt succeeds
        result = self.runner.invoke(main_app, ["train", "cnn-lstm", "--epochs", "5"])
        assert result.exit_code == 0

        # Backtesting should still work
        result = self.runner.invoke(
            main_app,
            [
                "backtest",
                "strategy",
                "--data-path",
                str(self.temp_dir),
                "--model-path",
                str(self.temp_dir),
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.start_trading")
    @patch("trading_rl_agent.cli.stop_trading")
    def test_recovery_from_trading_failure(self, mock_stop, mock_start):
        """Test recovery from trading failure."""
        # Start trading fails first, then succeeds
        mock_start.side_effect = [Exception("API error"), None]
        mock_stop.return_value = None

        # First start attempt fails
        result = self.runner.invoke(main_app, ["trade", "start", "--symbols", "AAPL", "--paper-trading"])
        assert result.exit_code != 0

        # Second start attempt succeeds
        result = self.runner.invoke(main_app, ["trade", "start", "--symbols", "AAPL", "--paper-trading"])
        assert result.exit_code == 0

        # Stop should still work
        result = self.runner.invoke(main_app, ["trade", "stop", "--all-sessions"])
        assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])
