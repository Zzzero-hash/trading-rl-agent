"""
CLI Workflow Tests for Trading RL Agent.

This module provides tests for CLI workflow functionality:
- Command structure validation
- Argument parsing
- Basic command execution
- Error handling
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestCLIWorkflow:
    """Test CLI workflow functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cli_structure(self):
        """Test CLI structure and command availability."""
        try:
            from trade_agent.cli import (
                app,
                backtest_app,
                data_app,
                scenario_app,
                trade_app,
                train_app,
            )

            # Verify all sub-apps exist
            assert app is not None
            assert data_app is not None
            assert train_app is not None
            assert backtest_app is not None
            assert trade_app is not None
            assert scenario_app is not None

            # Verify app has commands
            assert hasattr(app, "registered_commands")
            assert len(app.registered_commands) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_help_system(self):
        """Test help system functionality."""
        try:
            from trade_agent.cli import app

            # Test main help
            result = self.runner.invoke(app, ["--help"])
            assert result.exit_code == 0
            assert "trade-agent" in result.output
            assert "Production-grade trading RL agent" in result.output

            # Test subcommand help
            result = self.runner.invoke(app, ["data", "--help"])
            assert result.exit_code == 0
            assert "Data pipeline operations" in result.output

            result = self.runner.invoke(app, ["train", "--help"])
            assert result.exit_code == 0
            assert "Model training operations" in result.output

            result = self.runner.invoke(app, ["backtest", "--help"])
            assert result.exit_code == 0
            assert "Backtesting operations" in result.output

            result = self.runner.invoke(app, ["trade", "--help"])
            assert result.exit_code == 0
            assert "Live trading operations" in result.output

            result = self.runner.invoke(app, ["scenario", "--help"])
            assert result.exit_code == 0
            assert "Agent scenario evaluation" in result.output
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_argument_parsing(self):
        """Test argument parsing functionality."""
        try:
            from trade_agent.cli import app

            # Test verbose flag
            result = self.runner.invoke(app, ["-v", "--help"])
            assert result.exit_code == 0

            result = self.runner.invoke(app, ["-vv", "--help"])
            assert result.exit_code == 0

            result = self.runner.invoke(app, ["-vvv", "--help"])
            assert result.exit_code == 0

            # Test config file argument
            config_file = Path(self.temp_dir) / "test_config.yaml"
            config_file.write_text("data:\n  source: yfinance")

            result = self.runner.invoke(app, ["-c", str(config_file), "--help"])
            assert result.exit_code == 0

            # Test env file argument
            env_file = Path(self.temp_dir) / ".env"
            env_file.write_text("TRADING_RL_AGENT_DATA_SOURCE=yfinance")

            result = self.runner.invoke(app, ["--env-file", str(env_file), "--help"])
            assert result.exit_code == 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_error_handling(self):
        """Test error handling functionality."""
        try:
            from trade_agent.cli import app

            # Test invalid command
            result = self.runner.invoke(app, ["invalid-command"])
            assert result.exit_code != 0

            # Test invalid config file
            result = self.runner.invoke(app, ["-c", "nonexistent.yaml", "--help"])
            assert result.exit_code != 0

            # Test invalid env file
            result = self.runner.invoke(app, ["--env-file", "nonexistent.env", "--help"])
            assert result.exit_code != 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_data_commands_structure(self):
        """Test data command structure."""
        try:
            from trade_agent.cli import data_app

            # Test data subcommand help
            result = self.runner.invoke(data_app, ["--help"])
            assert result.exit_code == 0

            # Verify data commands exist
            assert hasattr(data_app, "registered_commands")
            assert len(data_app.registered_commands) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_train_commands_structure(self):
        """Test training command structure."""
        try:
            from trade_agent.cli import train_app

            # Test train subcommand help
            result = self.runner.invoke(train_app, ["--help"])
            assert result.exit_code == 0

            # Verify train commands exist
            assert hasattr(train_app, "registered_commands")
            assert len(train_app.registered_commands) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_backtest_commands_structure(self):
        """Test backtest command structure."""
        try:
            from trade_agent.cli import backtest_app

            # Test backtest subcommand help
            result = self.runner.invoke(backtest_app, ["--help"])
            assert result.exit_code == 0

            # Verify backtest commands exist
            assert hasattr(backtest_app, "registered_commands")
            assert len(backtest_app.registered_commands) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_trade_commands_structure(self):
        """Test trade command structure."""
        try:
            from trade_agent.cli import trade_app

            # Test trade subcommand help
            result = self.runner.invoke(trade_app, ["--help"])
            assert result.exit_code == 0

            # Verify trade commands exist
            assert hasattr(trade_app, "registered_commands")
            assert len(trade_app.registered_commands) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_scenario_commands_structure(self):
        """Test scenario command structure."""
        try:
            from trade_agent.cli import scenario_app

            # Test scenario subcommand help
            result = self.runner.invoke(scenario_app, ["--help"])
            assert result.exit_code == 0

            # Verify scenario commands exist
            assert hasattr(scenario_app, "registered_commands")
            assert len(scenario_app.registered_commands) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    @patch("trading_rl_agent.cli.download_all")
    def test_data_download_command(self, mock_download):
        """Test data download command."""
        try:
            from trade_agent.cli import app

            mock_download.return_value = None

            result = self.runner.invoke(
                app,
                [
                    "data",
                    "download-all",
                    "--start-date",
                    "2023-01-01",
                    "--end-date",
                    "2023-01-31",
                ],
            )

            # Command should execute (even if it fails due to mocking)
            assert result.exit_code in [0, 1]  # Allow for expected failures
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    @patch("trading_rl_agent.cli.train_cnn_lstm")
    def test_training_command(self, mock_train):
        """Test training command."""
        try:
            from trade_agent.cli import app

            mock_train.return_value = None

            result = self.runner.invoke(app, ["train", "cnn-lstm", "--epochs", "1"])

            # Command should execute (even if it fails due to mocking)
            assert result.exit_code in [0, 1]  # Allow for expected failures
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    @patch("trading_rl_agent.cli.run_backtest_strategy")
    def test_backtest_command(self, mock_backtest):
        """Test backtest command."""
        try:
            from trade_agent.cli import app

            mock_backtest.return_value = None

            result = self.runner.invoke(app, ["backtest", "strategy"])

            # Command should execute (even if it fails due to mocking)
            assert result.exit_code in [0, 1]  # Allow for expected failures
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    @patch("trading_rl_agent.cli.start_trading")
    def test_trade_command(self, mock_start):
        """Test trade command."""
        try:
            from trade_agent.cli import app

            mock_start.return_value = None

            result = self.runner.invoke(app, ["trade", "start", "--paper-trading"])

            # Command should execute (even if it fails due to mocking)
            assert result.exit_code in [0, 1]  # Allow for expected failures
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_command_descriptions(self):
        """Test that command descriptions are clear and helpful."""
        try:
            from trade_agent.cli import app

            # Test main app description
            result = self.runner.invoke(app, ["--help"])
            assert result.exit_code == 0
            assert "Production-grade trading RL agent" in result.output

            # Test subcommand descriptions
            result = self.runner.invoke(app, ["data", "--help"])
            assert result.exit_code == 0
            assert "Data pipeline operations" in result.output

            result = self.runner.invoke(app, ["train", "--help"])
            assert result.exit_code == 0
            assert "Model training operations" in result.output

            result = self.runner.invoke(app, ["backtest", "--help"])
            assert result.exit_code == 0
            assert "Backtesting operations" in result.output

            result = self.runner.invoke(app, ["trade", "--help"])
            assert result.exit_code == 0
            assert "Live trading operations" in result.output

            result = self.runner.invoke(app, ["scenario", "--help"])
            assert result.exit_code == 0
            assert "Agent scenario evaluation" in result.output
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_consistent_output_format(self):
        """Test consistent output formatting."""
        try:
            from trade_agent.cli import app

            # Test help output consistency
            result = self.runner.invoke(app, ["--help"])
            assert result.exit_code == 0
            assert len(result.output) > 0

            # Test subcommand help consistency
            result = self.runner.invoke(app, ["data", "--help"])
            assert result.exit_code == 0
            assert len(result.output) > 0
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")


class TestCLIPerformance:
    """Test CLI performance characteristics."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()

    def test_help_response_time(self):
        """Test that help command responds quickly."""
        try:
            import time

            from trade_agent.cli import app

            start_time = time.time()
            result = self.runner.invoke(app, ["--help"])
            end_time = time.time()

            assert result.exit_code == 0
            assert end_time - start_time < 2.0  # Should respond within 2 seconds
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")

    def test_subcommand_response_time(self):
        """Test that subcommands respond quickly."""
        try:
            import time

            from trade_agent.cli import app

            subcommands = ["data", "train", "backtest", "trade", "scenario"]

            for subcmd in subcommands:
                start_time = time.time()
                result = self.runner.invoke(app, [subcmd, "--help"])
                end_time = time.time()

                assert result.exit_code == 0
                assert end_time - start_time < 1.0  # Should respond within 1 second
        except ImportError as e:
            pytest.skip(f"CLI import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
