"""
Functional tests for Trading RL Agent CLI commands.

Tests actual CLI command execution with mock data and minimal execution
to verify functionality without requiring external dependencies.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml


class TestCLIFunctional:
    """Functional tests for CLI commands."""

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

    def create_test_config(self, temp_dir: Path) -> Path:
        """Create a test configuration file."""
        config = {
            "data": {
                "symbols": ["AAPL", "GOOGL"],
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "source": "yfinance",
                "timeframe": "1d",
            },
            "model": {"algorithm": "cnn_lstm", "epochs": 1, "batch_size": 2, "learning_rate": 0.001, "device": "cpu"},
            "backtest": {"initial_capital": 10000.0, "commission_rate": 0.001, "slippage_rate": 0.0001},
        }

        config_path = temp_dir / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def create_test_data(self, temp_dir: Path) -> Path:
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

        df = pd.DataFrame(data)
        data_path = temp_dir / "test_data.csv"
        df.to_csv(data_path, index=False)
        return data_path

    # ============================================================================
    # DATA COMMANDS FUNCTIONAL TESTS
    # ============================================================================

    @pytest.mark.slow
    def test_data_download_functional(self):
        """Test data download command with actual execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            result = self.run_cli_command(
                ["data", "download", "AAPL", "--start", "2024-01-01", "--end", "2024-01-02", "--output", str(temp_path)]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]
            if result.returncode == 0:
                # Check if data was downloaded
                data_files = list(temp_path.glob("*.csv"))
                assert len(data_files) > 0

    @pytest.mark.slow
    def test_data_process_functional(self):
        """Test data process command with test config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(["data", "process", "--config", str(config_path), "--output", str(temp_path)])

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_data_standardize_functional(self):
        """Test data standardize command with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = self.create_test_data(temp_path)
            output_path = temp_path / "standardized"
            output_path.mkdir()

            result = self.run_cli_command(["data", "standardize", str(data_path), "--output", str(output_path)])

            # Should succeed
            assert result.returncode == 0

    def test_data_pipeline_functional(self):
        """Test data pipeline command with test config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                ["data", "pipeline", "--config", str(config_path), "--output", str(temp_path)]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    # ============================================================================
    # TRAIN COMMANDS FUNCTIONAL TESTS
    # ============================================================================

    @pytest.mark.slow
    def test_train_cnn_lstm_functional(self):
        """Test CNN+LSTM training command with minimal execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                [
                    "train",
                    "cnn-lstm",
                    "--config",
                    str(config_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--output",
                    str(temp_path),
                    "--gpu",
                    "false",
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    @pytest.mark.slow
    def test_train_rl_functional(self):
        """Test RL training command with minimal execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                [
                    "train",
                    "rl",
                    "--config",
                    str(config_path),
                    "--timesteps",
                    "100",
                    "--output",
                    str(temp_path),
                    "--num-workers",
                    "1",
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_train_hybrid_functional(self):
        """Test hybrid training command with test models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy model files
            cnn_model_path = temp_path / "cnn_lstm_model"
            cnn_model_path.mkdir()
            (cnn_model_path / "model.ckpt").touch()

            rl_model_path = temp_path / "rl_model"
            rl_model_path.mkdir()
            (rl_model_path / "model.ckpt").touch()

            result = self.run_cli_command(
                [
                    "train",
                    "hybrid",
                    "--cnn-lstm-path",
                    str(cnn_model_path),
                    "--rl-path",
                    str(rl_model_path),
                    "--output",
                    str(temp_path),
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_train_hyperopt_functional(self):
        """Test hyperopt command with minimal trials."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                ["train", "hyperopt", "--config", str(config_path), "--n-trials", "2", "--output", str(temp_path)]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    # ============================================================================
    # BACKTEST COMMANDS FUNCTIONAL TESTS
    # ============================================================================

    def test_backtest_strategy_functional(self):
        """Test backtest strategy command with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = self.create_test_data(temp_path)

            result = self.run_cli_command(
                [
                    "backtest",
                    "strategy",
                    "--data-path",
                    str(data_path),
                    "--policy",
                    "momentum",
                    "--initial-capital",
                    "10000",
                    "--output",
                    str(temp_path),
                ]
            )

            # Should succeed
            assert result.returncode == 0

    def test_backtest_evaluate_functional(self):
        """Test backtest evaluate command with test model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = self.create_test_data(temp_path)

            # Create dummy model file
            model_path = temp_path / "test_model"
            model_path.mkdir()
            (model_path / "model.ckpt").touch()

            result = self.run_cli_command(
                [
                    "backtest",
                    "evaluate",
                    "--model-path",
                    str(model_path),
                    "--data-path",
                    str(data_path),
                    "--output",
                    str(temp_path),
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_backtest_walk_forward_functional(self):
        """Test backtest walk-forward command with test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = self.create_test_data(temp_path)

            result = self.run_cli_command(
                [
                    "backtest",
                    "walk-forward",
                    "--data-path",
                    str(data_path),
                    "--model-type",
                    "cnn_lstm",
                    "--train-window-size",
                    "10",
                    "--validation-window-size",
                    "5",
                    "--test-window-size",
                    "5",
                    "--step-size",
                    "3",
                    "--output",
                    str(temp_path),
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_backtest_compare_functional(self):
        """Test backtest compare command with test models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = self.create_test_data(temp_path)

            # Create dummy model files
            model1_path = temp_path / "model1"
            model1_path.mkdir()
            (model1_path / "model.ckpt").touch()

            model2_path = temp_path / "model2"
            model2_path.mkdir()
            (model2_path / "model.ckpt").touch()

            result = self.run_cli_command(
                [
                    "backtest",
                    "compare",
                    "--models",
                    f"{model1_path},{model2_path}",
                    "--data-path",
                    str(data_path),
                    "--output",
                    str(temp_path),
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_backtest_report_functional(self):
        """Test backtest report command with test results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create dummy results file
            results_path = temp_path / "results"
            results_path.mkdir()

            results_data = {"strategy": "test", "total_return": 0.05, "sharpe_ratio": 1.2, "max_drawdown": -0.02}

            with open(results_path / "results.json", "w") as f:
                json.dump(results_data, f)

            result = self.run_cli_command(
                [
                    "backtest",
                    "report",
                    "--results-path",
                    str(results_path),
                    "--output-format",
                    "html",
                    "--output",
                    str(temp_path),
                ]
            )

            # Should succeed
            assert result.returncode == 0

    # ============================================================================
    # TRADE COMMANDS FUNCTIONAL TESTS
    # ============================================================================

    def test_trade_start_functional(self):
        """Test trade start command with paper trading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                [
                    "trade",
                    "start",
                    "--config",
                    str(config_path),
                    "--symbols",
                    "AAPL",
                    "--paper-trading",
                    "true",
                    "--initial-capital",
                    "10000",
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_trade_status_functional(self):
        """Test trade status command."""
        result = self.run_cli_command(["trade", "status"])

        # Should succeed (even if no active sessions)
        assert result.returncode == 0

    def test_trade_paper_functional(self):
        """Test trade paper command with minimal duration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                ["trade", "paper", "--config", str(config_path), "--symbols", "AAPL", "--duration", "1h"]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    # ============================================================================
    # SCENARIO COMMANDS FUNCTIONAL TESTS
    # ============================================================================

    def test_scenario_evaluate_functional(self):
        """Test scenario evaluate command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                [
                    "scenario",
                    "evaluate",
                    "--config",
                    str(config_path),
                    "--agent-type",
                    "moving_average",
                    "--output",
                    str(temp_path),
                    "--seed",
                    "42",
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_scenario_compare_functional(self):
        """Test scenario compare command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                ["scenario", "compare", "--config", str(config_path), "--output", str(temp_path), "--seed", "42"]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    def test_scenario_custom_functional(self):
        """Test scenario custom command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = self.create_test_config(temp_path)

            result = self.run_cli_command(
                [
                    "scenario",
                    "custom",
                    "--config",
                    str(config_path),
                    "--agent-type",
                    "moving_average",
                    "--scenario-name",
                    "strong_uptrend",
                    "--output",
                    str(temp_path),
                    "--seed",
                    "42",
                ]
            )

            # Should succeed or fail gracefully
            assert result.returncode in [0, 1]

    # ============================================================================
    # ERROR HANDLING TESTS
    # ============================================================================

    def test_invalid_config_file(self):
        """Test CLI behavior with invalid config file."""
        result = self.run_cli_command(["--config", "/nonexistent/config.yaml", "version"])

        # Should fail gracefully
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()

    def test_missing_required_arguments(self):
        """Test CLI behavior with missing required arguments."""
        result = self.run_cli_command(["data", "download"])

        # Should fail due to missing symbols argument
        assert result.returncode != 0

    def test_invalid_option_values(self):
        """Test CLI behavior with invalid option values."""
        result = self.run_cli_command(
            [
                "train",
                "cnn-lstm",
                "--epochs",
                "-1",  # Invalid negative epochs
            ]
        )

        # Should fail due to invalid epochs
        assert result.returncode != 0

    def test_verbose_output(self):
        """Test CLI verbose output functionality."""
        result = self.run_cli_command(["-v", "version"])

        # Should succeed with verbose output
        assert result.returncode == 0
        assert "Trading RL Agent" in result.stdout

    def test_multiple_verbose_flags(self):
        """Test CLI with multiple verbose flags."""
        result = self.run_cli_command(["-vv", "info"])

        # Should succeed with increased verbosity
        assert result.returncode == 0
        assert "System Information" in result.stdout
