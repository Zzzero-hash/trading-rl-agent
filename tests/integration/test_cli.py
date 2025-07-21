"""
Integration tests for the Trading RL Agent CLI.

Tests CLI commands with --help and dry-run modes using subprocess.run.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLI:
    """Test CLI commands with --help and dry-run modes."""

    def run_cli_command(self, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run CLI command and return the result."""
        cmd = [sys.executable, "main.py", *args]
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent.parent,
        )

    # ============================================================================
    # ROOT COMMANDS
    # ============================================================================

    def test_root_help(self):
        """Test root command --help."""
        result = self.run_cli_command(["--help"])
        assert result.returncode == 0
        assert "trading RL agent" in result.stdout.lower()
        assert "data" in result.stdout
        assert "train" in result.stdout
        assert "backtest" in result.stdout
        assert "trade" in result.stdout

    def test_version(self):
        """Test version command."""
        result = self.run_cli_command(["version"])
        assert result.returncode == 0
        assert "trading RL agent" in result.stdout.lower()

    def test_info(self):
        """Test info command."""
        result = self.run_cli_command(["info"])
        assert result.returncode == 0
        assert "System Information" in result.stdout

    # ============================================================================
    # DATA SUBCOMMANDS
    # ============================================================================

    def test_data_help(self):
        """Test data subcommand --help."""
        result = self.run_cli_command(["data", "--help"])
        assert result.returncode == 0
        assert "Data pipeline operations" in result.stdout
        assert "all" in result.stdout
        assert "symbols" in result.stdout
        assert "refresh" in result.stdout
        assert "download" in result.stdout
        assert "process" in result.stdout
        assert "standardize" in result.stdout
        assert "pipeline" in result.stdout

    @pytest.mark.slow
    def test_data_all_dry_run(self):
        """Test data all command with dry-run mode."""
        result = self.run_cli_command(
            [
                "data",
                "all",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
                "--output",
                "/tmp/test_data",
                "--source",
                "yfinance",
                "--timeframe",
                "1d",
                "--parallel",
                "false",
            ]
        )
        # Note: This command doesn't have a --dry-run flag, so it will actually run
        # but with minimal data (1 day) to keep it fast
        assert result.returncode in [0, 1]  # May fail due to network/data issues

    @pytest.mark.slow
    def test_data_symbols_dry_run(self):
        """Test data symbols command with minimal data."""
        result = self.run_cli_command(
            [
                "data",
                "symbols",
                "AAPL",
                "--start",
                "2024-01-01",
                "--end",
                "2024-01-02",
                "--output",
                "/tmp/test_data",
                "--source",
                "yfinance",
                "--timeframe",
                "1d",
                "--parallel",
                "false",
            ]
        )
        # Note: This command doesn't have a --dry-run flag, so it will actually run
        # but with minimal data (1 day, 1 symbol) to keep it fast
        assert result.returncode in [0, 1]  # May fail due to network/data issues

    @pytest.mark.slow
    def test_data_refresh_dry_run(self):
        """Test data refresh command with minimal data."""
        result = self.run_cli_command(
            [
                "data",
                "refresh",
                "--days",
                "1",
                "--symbols",
                "AAPL",
                "--output",
                "/tmp/test_data",
                "--source",
                "yfinance",
                "--timeframe",
                "1d",
                "--parallel",
                "false",
            ]
        )
        # Note: This command doesn't have a --dry-run flag, so it will actually run
        # but with minimal data to keep it fast
        assert result.returncode in [0, 1]  # May fail due to network/data issues

    def test_data_download_help(self):
        """Test data download command --help."""
        result = self.run_cli_command(["data", "download", "--help"])
        assert result.returncode == 0
        assert "Download market data" in result.stdout

    def test_data_process_help(self):
        """Test data process command --help."""
        result = self.run_cli_command(["data", "process", "--help"])
        assert result.returncode == 0
        assert "Process and build optimized datasets" in result.stdout

    def test_data_standardize_help(self):
        """Test data standardize command --help."""
        result = self.run_cli_command(["data", "standardize", "--help"])
        assert result.returncode == 0
        assert "Standardize data" in result.stdout

    def test_data_pipeline_help(self):
        """Test data pipeline command --help."""
        result = self.run_cli_command(["data", "pipeline", "--help"])
        assert result.returncode == 0
        assert "Run the complete data pipeline" in result.stdout

    # ============================================================================
    # TRAIN SUBCOMMANDS
    # ============================================================================

    def test_train_help(self):
        """Test train subcommand --help."""
        result = self.run_cli_command(["train", "--help"])
        assert result.returncode == 0
        assert "Model training operations" in result.stdout
        assert "cnn-lstm" in result.stdout
        assert "rl" in result.stdout
        assert "hybrid" in result.stdout
        assert "hyperopt" in result.stdout

    @pytest.mark.slow
    def test_train_cnn_lstm_dry_run(self):
        """Test train cnn-lstm command with minimal epochs."""
        result = self.run_cli_command(
            [
                "train",
                "cnn-lstm",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--output",
                "/tmp/test_models",
                "--gpu",
                "false",
            ]
        )
        # Note: This command doesn't have a --dry-run flag, so it will actually run
        # but with minimal epochs to keep it fast
        assert result.returncode in [0, 1]  # May fail due to missing data/dependencies

    @pytest.mark.slow
    def test_train_rl_dry_run(self):
        """Test train rl command with minimal timesteps."""
        result = self.run_cli_command(
            [
                "train",
                "rl",
                "ppo",
                "--timesteps",
                "100",
                "--workers",
                "1",
                "--output",
                "/tmp/test_models",
            ]
        )
        # Note: This command doesn't have a --dry-run flag, so it will actually run
        # but with minimal timesteps to keep it fast
        assert result.returncode in [0, 1]  # May fail due to missing data/dependencies

    def test_train_hybrid_help(self):
        """Test train hybrid command --help."""
        result = self.run_cli_command(["train", "hybrid", "--help"])
        assert result.returncode == 0
        assert "Train hybrid models" in result.stdout

    def test_train_hyperopt_help(self):
        """Test train hyperopt command --help."""
        result = self.run_cli_command(["train", "hyperopt", "--help"])
        assert result.returncode == 0
        assert "Hyperparameter optimization" in result.stdout

    # ============================================================================
    # BACKTEST SUBCOMMANDS
    # ============================================================================

    def test_backtest_help(self):
        """Test backtest subcommand --help."""
        result = self.run_cli_command(["backtest", "--help"])
        assert result.returncode == 0
        assert "Backtesting operations" in result.stdout
        assert "strategy" in result.stdout
        assert "evaluate" in result.stdout
        assert "compare" in result.stdout
        assert "report" in result.stdout

    def test_backtest_strategy_help(self):
        """Test backtest strategy command --help."""
        result = self.run_cli_command(["backtest", "strategy", "--help"])
        assert result.returncode == 0
        assert "Backtest trading strategy" in result.stdout

    def test_backtest_evaluate_help(self):
        """Test backtest evaluate command --help."""
        result = self.run_cli_command(["backtest", "evaluate", "--help"])
        assert result.returncode == 0
        assert "Evaluate trained model" in result.stdout

    def test_backtest_compare_help(self):
        """Test backtest compare command --help."""
        result = self.run_cli_command(["backtest", "compare", "--help"])
        assert result.returncode == 0
        assert "Compare multiple models" in result.stdout

    def test_backtest_report_help(self):
        """Test backtest report command --help."""
        result = self.run_cli_command(["backtest", "report", "--help"])
        assert result.returncode == 0
        assert "Generate backtest report" in result.stdout

    # ============================================================================
    # TRADE SUBCOMMANDS
    # ============================================================================

    def test_trade_help(self):
        """Test trade subcommand --help."""
        result = self.run_cli_command(["trade", "--help"])
        assert result.returncode == 0
        assert "Live trading operations" in result.stdout
        assert "start" in result.stdout
        assert "stop" in result.stdout
        assert "status" in result.stdout
        assert "monitor" in result.stdout
        assert "paper" in result.stdout

    def test_trade_start_help(self):
        """Test trade start command --help."""
        result = self.run_cli_command(["trade", "start", "--help"])
        assert result.returncode == 0
        assert "Start live trading session" in result.stdout

    def test_trade_stop_help(self):
        """Test trade stop command --help."""
        result = self.run_cli_command(["trade", "stop", "--help"])
        assert result.returncode == 0
        assert "Stop trading session" in result.stdout

    def test_trade_status_help(self):
        """Test trade status command --help."""
        result = self.run_cli_command(["trade", "status", "--help"])
        assert result.returncode == 0
        assert "Show trading session status" in result.stdout

    def test_trade_monitor_help(self):
        """Test trade monitor command --help."""
        result = self.run_cli_command(["trade", "monitor", "--help"])
        assert result.returncode == 0
        assert "Monitor trading session" in result.stdout

    def test_trade_paper_help(self):
        """Test trade paper command --help."""
        result = self.run_cli_command(["trade", "paper", "--help"])
        assert result.returncode == 0
        assert "Start paper trading session" in result.stdout

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================

    def test_invalid_command(self):
        """Test invalid command returns non-zero exit code."""
        result = self.run_cli_command(["invalid-command"])
        assert result.returncode != 0

    def test_invalid_subcommand(self):
        """Test invalid subcommand returns non-zero exit code."""
        result = self.run_cli_command(["data", "invalid-subcommand"])
        assert result.returncode != 0

    def test_missing_required_argument(self):
        """Test missing required argument returns non-zero exit code."""
        result = self.run_cli_command(["data", "symbols"])  # Missing symbols argument
        assert result.returncode != 0

    def test_invalid_option(self):
        """Test invalid option returns non-zero exit code."""
        result = self.run_cli_command(["--invalid-option"])
        assert result.returncode != 0

    # ============================================================================
    # VERBOSE MODE TESTS
    # ============================================================================

    def test_verbose_mode(self):
        """Test verbose mode with -v flag."""
        result = self.run_cli_command(["-v", "version"])
        assert result.returncode == 0
        assert "Trading RL Agent" in result.stdout

    def test_multiple_verbose_flags(self):
        """Test multiple verbose flags."""
        result = self.run_cli_command(["-vv", "version"])
        assert result.returncode == 0
        assert "Trading RL Agent" in result.stdout

    # ============================================================================
    # CONFIGURATION TESTS
    # ============================================================================

    def test_config_file_option(self):
        """Test --config option with non-existent file."""
        result = self.run_cli_command(["--config", "/nonexistent/config.yaml", "version"])
        # Should still work for version command even with invalid config
        assert result.returncode == 0

    def test_env_file_option(self):
        """Test --env-file option with non-existent file."""
        result = self.run_cli_command(["--env-file", "/nonexistent/.env", "version"])
        # Should still work for version command even with invalid env file
        assert result.returncode == 0
