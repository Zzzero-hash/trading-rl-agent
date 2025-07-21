"""
Smoke Tests for CLI Critical Paths.

This module provides smoke tests for critical CLI functionality:
- Basic command availability
- Help system functionality
- Version information
- Configuration loading
- Essential workflows
- Error handling basics
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trading_rl_agent.cli import app as main_app
from trading_rl_agent.cli_backtest import app as backtest_app
from trading_rl_agent.cli_health import app as health_app
from trading_rl_agent.cli_trade import app as trade_app


class TestCLISmokeBasic:
    """Smoke tests for basic CLI functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_imports(self):
        """Test that CLI modules can be imported without errors."""
        # This test ensures the CLI modules are properly structured
        assert main_app is not None
        assert backtest_app is not None
        assert health_app is not None
        assert trade_app is not None

    def test_help_command(self):
        """Test that help command works."""
        result = self.runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        assert "trading-rl-agent" in result.output
        assert "Production-grade live trading system" in result.output

    def test_version_command(self):
        """Test that version command works."""
        result = self.runner.invoke(main_app, ["version"])
        assert result.exit_code == 0
        assert "Trading RL Agent" in result.output

    def test_info_command(self):
        """Test that info command works."""
        result = self.runner.invoke(main_app, ["info"])
        assert result.exit_code == 0
        assert "System Information" in result.output

    def test_subcommand_help(self):
        """Test that subcommand help works."""
        # Test data subcommand help
        result = self.runner.invoke(main_app, ["data", "--help"])
        assert result.exit_code == 0
        assert "Data pipeline operations" in result.output

        # Test train subcommand help
        result = self.runner.invoke(main_app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Model training operations" in result.output

        # Test backtest subcommand help
        result = self.runner.invoke(main_app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "Backtesting operations" in result.output

        # Test trade subcommand help
        result = self.runner.invoke(main_app, ["trade", "--help"])
        assert result.exit_code == 0
        assert "Live trading operations" in result.output

        # Test scenario subcommand help
        result = self.runner.invoke(main_app, ["scenario", "--help"])
        assert result.exit_code == 0
        assert "Agent scenario evaluation" in result.output

    def test_verbose_flag(self):
        """Test that verbose flag works."""
        result = self.runner.invoke(main_app, ["-v", "info"])
        assert result.exit_code == 0

        result = self.runner.invoke(main_app, ["-vv", "info"])
        assert result.exit_code == 0

        result = self.runner.invoke(main_app, ["-vvv", "info"])
        assert result.exit_code == 0

    def test_invalid_command_handling(self):
        """Test that invalid commands are handled gracefully."""
        result = self.runner.invoke(main_app, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output or "Error" in result.output


class TestCLISmokeData:
    """Smoke tests for data CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.download_all")
    def test_download_all_smoke(self, mock_download):
        """Smoke test for download-all command."""
        mock_download.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-01-31",
            ],
        )
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.download_symbols")
    def test_symbols_smoke(self, mock_download):
        """Smoke test for symbols command."""
        mock_download.return_value = None

        result = self.runner.invoke(main_app, ["data", "symbols", "--symbols", "AAPL,GOOGL"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.refresh_data")
    def test_refresh_smoke(self, mock_refresh):
        """Smoke test for refresh command."""
        mock_refresh.return_value = None

        result = self.runner.invoke(main_app, ["data", "refresh", "--days", "1"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.download_data")
    def test_download_smoke(self, mock_download):
        """Smoke test for download command."""
        mock_download.return_value = None

        result = self.runner.invoke(main_app, ["data", "download", "--symbols", "AAPL"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.process_data")
    def test_process_smoke(self, mock_process):
        """Smoke test for process command."""
        mock_process.return_value = None

        result = self.runner.invoke(main_app, ["data", "process"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.standardize_data")
    def test_standardize_smoke(self, mock_standardize):
        """Smoke test for standardize command."""
        mock_standardize.return_value = None

        result = self.runner.invoke(main_app, ["data", "standardize"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.build_pipeline")
    def test_pipeline_smoke(self, mock_pipeline):
        """Smoke test for pipeline command."""
        mock_pipeline.return_value = None

        result = self.runner.invoke(main_app, ["data", "pipeline"])
        assert result.exit_code == 0


class TestCLISmokeTraining:
    """Smoke tests for training CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    def test_cnn_lstm_smoke(self, mock_train):
        """Smoke test for cnn-lstm command."""
        mock_train.return_value = None

        result = self.runner.invoke(main_app, ["train", "cnn-lstm", "--epochs", "1"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.train_rl_agent")
    def test_rl_smoke(self, mock_train):
        """Smoke test for rl command."""
        mock_train.return_value = None

        result = self.runner.invoke(main_app, ["train", "rl", "--agent-type", "ppo", "--timesteps", "1000"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.train_hybrid_model")
    def test_hybrid_smoke(self, mock_train):
        """Smoke test for hybrid command."""
        mock_train.return_value = None

        result = self.runner.invoke(main_app, ["train", "hybrid"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.run_hyperopt")
    def test_hyperopt_smoke(self, mock_hyperopt):
        """Smoke test for hyperopt command."""
        mock_hyperopt.return_value = None

        result = self.runner.invoke(main_app, ["train", "hyperopt", "--n-trials", "1"])
        assert result.exit_code == 0


class TestCLISmokeBacktest:
    """Smoke tests for backtesting CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.run_backtest_strategy")
    def test_strategy_smoke(self, mock_backtest):
        """Smoke test for strategy command."""
        mock_backtest.return_value = None

        result = self.runner.invoke(main_app, ["backtest", "strategy"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.evaluate_model")
    def test_evaluate_smoke(self, mock_evaluate):
        """Smoke test for evaluate command."""
        mock_evaluate.return_value = None

        result = self.runner.invoke(main_app, ["backtest", "evaluate"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.run_walk_forward")
    def test_walk_forward_smoke(self, mock_walk_forward):
        """Smoke test for walk-forward command."""
        mock_walk_forward.return_value = None

        result = self.runner.invoke(main_app, ["backtest", "walk-forward"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.compare_models")
    def test_compare_smoke(self, mock_compare):
        """Smoke test for compare command."""
        mock_compare.return_value = None

        result = self.runner.invoke(main_app, ["backtest", "compare", "--models", "model1,model2"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.generate_report")
    def test_report_smoke(self, mock_report):
        """Smoke test for report command."""
        mock_report.return_value = None

        result = self.runner.invoke(main_app, ["backtest", "report"])
        assert result.exit_code == 0


class TestCLISmokeTrade:
    """Smoke tests for trading CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.start_trading")
    def test_start_smoke(self, mock_start):
        """Smoke test for start command."""
        mock_start.return_value = None

        result = self.runner.invoke(main_app, ["trade", "start", "--paper-trading"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.stop_trading")
    def test_stop_smoke(self, mock_stop):
        """Smoke test for stop command."""
        mock_stop.return_value = None

        result = self.runner.invoke(main_app, ["trade", "stop"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.get_trading_status")
    def test_status_smoke(self, mock_status):
        """Smoke test for status command."""
        mock_status.return_value = {"status": "idle"}

        result = self.runner.invoke(main_app, ["trade", "status"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.monitor_trading")
    def test_monitor_smoke(self, mock_monitor):
        """Smoke test for monitor command."""
        mock_monitor.return_value = None

        result = self.runner.invoke(main_app, ["trade", "monitor"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.start_paper_trading")
    def test_paper_smoke(self, mock_paper):
        """Smoke test for paper command."""
        mock_paper.return_value = None

        result = self.runner.invoke(main_app, ["trade", "paper"])
        assert result.exit_code == 0


class TestCLISmokeScenario:
    """Smoke tests for scenario CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli.evaluate_scenario")
    def test_scenario_evaluate_smoke(self, mock_evaluate):
        """Smoke test for scenario-evaluate command."""
        mock_evaluate.return_value = None

        result = self.runner.invoke(main_app, ["scenario", "scenario-evaluate"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.compare_scenarios")
    def test_scenario_compare_smoke(self, mock_compare):
        """Smoke test for scenario-compare command."""
        mock_compare.return_value = None

        result = self.runner.invoke(main_app, ["scenario", "scenario-compare"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli.run_custom_scenario")
    def test_custom_smoke(self, mock_custom):
        """Smoke test for custom command."""
        mock_custom.return_value = None

        result = self.runner.invoke(main_app, ["scenario", "custom"])
        assert result.exit_code == 0


class TestCLISmokeHealth:
    """Smoke tests for health CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trading_rl_agent.cli_health.monitor")
    def test_monitor_smoke(self, mock_monitor):
        """Smoke test for health monitor command."""
        mock_monitor.return_value = None

        result = self.runner.invoke(health_app, ["monitor", "--duration", "10"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli_health.status")
    def test_status_smoke(self, mock_status):
        """Smoke test for health status command."""
        mock_status.return_value = None

        result = self.runner.invoke(health_app, ["status"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli_health.report")
    def test_report_smoke(self, mock_report):
        """Smoke test for health report command."""
        mock_report.return_value = None

        result = self.runner.invoke(health_app, ["report"])
        assert result.exit_code == 0

    @patch("trading_rl_agent.cli_health.alerts")
    def test_alerts_smoke(self, mock_alerts):
        """Smoke test for health alerts command."""
        mock_alerts.return_value = None

        result = self.runner.invoke(health_app, ["alerts"])
        assert result.exit_code == 0


class TestCLISmokeErrorHandling:
    """Smoke tests for error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_missing_required_args(self):
        """Test handling of missing required arguments."""
        # Test data download without required args
        result = self.runner.invoke(main_app, ["data", "download"])
        assert result.exit_code != 0

        # Test training without required args
        result = self.runner.invoke(main_app, ["train", "cnn-lstm"])
        assert result.exit_code != 0

        # Test backtest without required args
        result = self.runner.invoke(main_app, ["backtest", "strategy"])
        assert result.exit_code != 0

    def test_invalid_argument_types(self):
        """Test handling of invalid argument types."""
        # Test invalid epochs
        result = self.runner.invoke(main_app, ["train", "cnn-lstm", "--epochs", "invalid"])
        assert result.exit_code != 0

        # Test invalid learning rate
        result = self.runner.invoke(main_app, ["train", "cnn-lstm", "--learning-rate", "invalid"])
        assert result.exit_code != 0

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        result = self.runner.invoke(main_app, ["-c", "nonexistent.yaml", "info"])
        assert result.exit_code != 0

    @patch("trading_rl_agent.cli.download_all")
    def test_network_error_handling(self, mock_download):
        """Test handling of network errors."""
        mock_download.side_effect = Exception("Network error")

        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-01-31",
            ],
        )
        assert result.exit_code != 0

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    def test_memory_error_handling(self, mock_train):
        """Test handling of memory errors."""
        mock_train.side_effect = MemoryError("Out of memory")

        result = self.runner.invoke(main_app, ["train", "cnn-lstm", "--epochs", "10"])
        assert result.exit_code != 0


class TestCLISmokeConfiguration:
    """Smoke tests for configuration handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_file_loading(self):
        """Test configuration file loading."""
        # Create a minimal config file
        config_content = """
        data:
          source: "yfinance"
          symbols: ["AAPL", "GOOGL"]
        training:
          epochs: 10
          batch_size: 32
        """
        config_file = Path(self.temp_dir) / "test_config.yaml"
        config_file.write_text(config_content)

        with patch("trading_rl_agent.cli.info") as mock_info:
            mock_info.return_value = None

            result = self.runner.invoke(main_app, ["-c", str(config_file), "info"])
            assert result.exit_code == 0

    def test_env_file_loading(self):
        """Test environment file loading."""
        env_file = Path(self.temp_dir) / ".env"
        env_content = "TRADING_RL_AGENT_DATA_SOURCE=yfinance\n"
        env_file.write_text(env_content)

        with patch("trading_rl_agent.cli.info") as mock_info:
            mock_info.return_value = None

            result = self.runner.invoke(main_app, ["--env-file", str(env_file), "info"])
            assert result.exit_code == 0

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        # Create invalid config file
        invalid_config = Path(self.temp_dir) / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        result = self.runner.invoke(main_app, ["-c", str(invalid_config), "info"])
        assert result.exit_code != 0


class TestCLISmokePerformance:
    """Smoke tests for performance characteristics."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_help_response_time(self):
        """Test that help command responds quickly."""
        import time

        start_time = time.time()
        result = self.runner.invoke(main_app, ["--help"])
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 1.0  # Should respond within 1 second

    def test_version_response_time(self):
        """Test that version command responds quickly."""
        import time

        start_time = time.time()
        result = self.runner.invoke(main_app, ["version"])
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 0.5  # Should respond within 0.5 seconds

    def test_info_response_time(self):
        """Test that info command responds quickly."""
        import time

        start_time = time.time()
        result = self.runner.invoke(main_app, ["info"])
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 2.0  # Should respond within 2 seconds

    @patch("trading_rl_agent.cli.download_all")
    def test_data_command_response_time(self, mock_download):
        """Test that data commands respond within reasonable time."""
        import time

        mock_download.return_value = None

        start_time = time.time()
        result = self.runner.invoke(
            main_app,
            [
                "data",
                "download-all",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2023-01-31",
            ],
        )
        end_time = time.time()

        assert result.exit_code == 0
        assert end_time - start_time < 5.0  # Should respond within 5 seconds


if __name__ == "__main__":
    pytest.main([__file__])
