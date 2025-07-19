"""
Comprehensive CLI Testing Suite for Trading RL Agent.

This module provides extensive testing for:
- All CLI commands and subcommands
- Argument parsing and validation
- Error handling and user feedback
- Integration with core functionality
- Performance under load
- Memory usage during operations
- Error recovery scenarios
- User experience validation
"""

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trading_rl_agent.cli import app as main_app
from trading_rl_agent.cli_backtest import app as backtest_app
from trading_rl_agent.cli_health import app as health_app
from trading_rl_agent.cli_trade import app as trade_app
from trading_rl_agent.cli_train import app as train_app


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = Path(self.temp_dir) / "test_config.yaml"

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_main_app_help(self):
        """Test main app help command."""
        result = self.runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        assert "trading-rl-agent" in result.output
        assert "Production-grade live trading system" in result.output

    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(main_app, ["version"])
        assert result.exit_code == 0
        assert "Trading RL Agent" in result.output

    def test_info_command(self):
        """Test info command."""
        result = self.runner.invoke(main_app, ["info"])
        assert result.exit_code == 0
        assert "System Information" in result.output

    def test_verbose_flag(self):
        """Test verbose flag functionality."""
        result = self.runner.invoke(main_app, ["-v", "info"])
        assert result.exit_code == 0

        result = self.runner.invoke(main_app, ["-vv", "info"])
        assert result.exit_code == 0

        result = self.runner.invoke(main_app, ["-vvv", "info"])
        assert result.exit_code == 0

    def test_config_file_loading(self):
        """Test config file loading."""
        # Create a minimal config file
        config_content = """
        data:
          source: "yfinance"
          symbols: ["AAPL", "GOOGL"]
        training:
          epochs: 10
          batch_size: 32
        """
        self.config_file.write_text(config_content)

        result = self.runner.invoke(main_app, ["-c", str(self.config_file), "info"])
        assert result.exit_code == 0

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        result = self.runner.invoke(main_app, ["-c", "nonexistent.yaml", "info"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_env_file_loading(self):
        """Test environment file loading."""
        env_file = Path(self.temp_dir) / ".env"
        env_content = "TRADING_RL_AGENT_DATA_SOURCE=yfinance\n"
        env_file.write_text(env_content)

        result = self.runner.invoke(main_app, ["--env-file", str(env_file), "info"])
        assert result.exit_code == 0


class TestDataCLICommands:
    """Test data pipeline CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('trading_rl_agent.cli.download_all')
    def test_download_all_command(self, mock_download):
        """Test download all command."""
        mock_download.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "download-all",
            "--start-date", "2023-01-01",
            "--end-date", "2023-12-31",
            "--output-dir", self.temp_dir,
            "--source", "yfinance",
            "--timeframe", "1d",
            "--parallel"
        ])
        
        assert result.exit_code == 0
        mock_download.assert_called_once()

    @patch('trading_rl_agent.cli.download_symbols')
    def test_symbols_command(self, mock_download):
        """Test symbols download command."""
        mock_download.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "symbols",
            "--symbols", "AAPL,GOOGL,MSFT",
            "--start-date", "2023-01-01",
            "--end-date", "2023-12-31",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_download.assert_called_once()

    @patch('trading_rl_agent.cli.refresh_data')
    def test_refresh_command(self, mock_refresh):
        """Test data refresh command."""
        mock_refresh.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "refresh",
            "--days", "7",
            "--symbols", "AAPL,GOOGL",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_refresh.assert_called_once()

    @patch('trading_rl_agent.cli.download_data')
    def test_download_command(self, mock_download):
        """Test single download command."""
        mock_download.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "download",
            "--symbols", "AAPL",
            "--start-date", "2023-01-01",
            "--end-date", "2023-12-31",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_download.assert_called_once()

    @patch('trading_rl_agent.cli.process_data')
    def test_process_command(self, mock_process):
        """Test data processing command."""
        mock_process.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "process",
            "--output-dir", self.temp_dir,
            "--force-rebuild",
            "--parallel"
        ])
        
        assert result.exit_code == 0
        mock_process.assert_called_once()

    @patch('trading_rl_agent.cli.standardize_data')
    def test_standardize_command(self, mock_standardize):
        """Test data standardization command."""
        mock_standardize.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "standardize",
            "--input-path", self.temp_dir,
            "--output-path", f"{self.temp_dir}/standardized",
            "--method", "robust"
        ])
        
        assert result.exit_code == 0
        mock_standardize.assert_called_once()

    @patch('trading_rl_agent.cli.build_pipeline')
    def test_pipeline_command(self, mock_pipeline):
        """Test pipeline building command."""
        mock_pipeline.return_value = None
        
        result = self.runner.invoke(main_app, [
            "data", "pipeline",
            "--config-path", self.temp_dir,
            "--output-dir", f"{self.temp_dir}/pipeline"
        ])
        
        assert result.exit_code == 0
        mock_pipeline.assert_called_once()


class TestTrainingCLICommands:
    """Test model training CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('trading_rl_agent.cli.train_cnn_lstm')
    def test_cnn_lstm_command(self, mock_train):
        """Test CNN+LSTM training command."""
        mock_train.return_value = None
        
        result = self.runner.invoke(main_app, [
            "train", "cnn-lstm",
            "--epochs", "10",
            "--batch-size", "32",
            "--learning-rate", "0.001",
            "--output-dir", self.temp_dir,
            "--gpu",
            "--mixed-precision"
        ])
        
        assert result.exit_code == 0
        mock_train.assert_called_once()

    @patch('trading_rl_agent.cli.train_rl_agent')
    def test_rl_command(self, mock_train):
        """Test RL agent training command."""
        mock_train.return_value = None
        
        result = self.runner.invoke(main_app, [
            "train", "rl",
            "--agent-type", "ppo",
            "--timesteps", "100000",
            "--output-dir", self.temp_dir,
            "--ray-address", "auto",
            "--num-workers", "4"
        ])
        
        assert result.exit_code == 0
        mock_train.assert_called_once()

    @patch('trading_rl_agent.cli.train_hybrid_model')
    def test_hybrid_command(self, mock_train):
        """Test hybrid model training command."""
        mock_train.return_value = None
        
        result = self.runner.invoke(main_app, [
            "train", "hybrid",
            "--cnn-lstm-path", f"{self.temp_dir}/cnn_lstm",
            "--rl-path", f"{self.temp_dir}/rl",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_train.assert_called_once()

    @patch('trading_rl_agent.cli.run_hyperopt')
    def test_hyperopt_command(self, mock_hyperopt):
        """Test hyperparameter optimization command."""
        mock_hyperopt.return_value = None
        
        result = self.runner.invoke(main_app, [
            "train", "hyperopt",
            "--n-trials", "50",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_hyperopt.assert_called_once()


class TestBacktestCLICommands:
    """Test backtesting CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('trading_rl_agent.cli.run_backtest_strategy')
    def test_strategy_command(self, mock_backtest):
        """Test strategy backtesting command."""
        mock_backtest.return_value = None
        
        result = self.runner.invoke(main_app, [
            "backtest", "strategy",
            "--data-path", self.temp_dir,
            "--model-path", f"{self.temp_dir}/model",
            "--policy", "ppo",
            "--initial-capital", "10000",
            "--commission", "0.001",
            "--slippage", "0.0001",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_backtest.assert_called_once()

    @patch('trading_rl_agent.cli.evaluate_model')
    def test_evaluate_command(self, mock_evaluate):
        """Test model evaluation command."""
        mock_evaluate.return_value = None
        
        result = self.runner.invoke(main_app, [
            "backtest", "evaluate",
            "--model-path", f"{self.temp_dir}/model",
            "--data-path", self.temp_dir,
            "--output-dir", self.temp_dir,
            "--initial-capital", "10000"
        ])
        
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()

    @patch('trading_rl_agent.cli.run_walk_forward')
    def test_walk_forward_command(self, mock_walk_forward):
        """Test walk-forward analysis command."""
        mock_walk_forward.return_value = None
        
        result = self.runner.invoke(main_app, [
            "backtest", "walk-forward",
            "--data-path", self.temp_dir,
            "--model-type", "cnn_lstm",
            "--train-window-size", "252",
            "--validation-window-size", "63",
            "--test-window-size", "63",
            "--step-size", "21",
            "--output-dir", self.temp_dir,
            "--confidence-level", "0.95",
            "--generate-plots",
            "--save-results"
        ])
        
        assert result.exit_code == 0
        mock_walk_forward.assert_called_once()

    @patch('trading_rl_agent.cli.compare_models')
    def test_compare_command(self, mock_compare):
        """Test model comparison command."""
        mock_compare.return_value = None
        
        result = self.runner.invoke(main_app, [
            "backtest", "compare",
            "--models", "model1,model2,model3",
            "--data-path", self.temp_dir,
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_compare.assert_called_once()

    @patch('trading_rl_agent.cli.generate_report')
    def test_report_command(self, mock_report):
        """Test report generation command."""
        mock_report.return_value = None
        
        result = self.runner.invoke(main_app, [
            "backtest", "report",
            "--results-path", self.temp_dir,
            "--output-format", "html",
            "--output-dir", self.temp_dir
        ])
        
        assert result.exit_code == 0
        mock_report.assert_called_once()


class TestTradeCLICommands:
    """Test live trading CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('trading_rl_agent.cli.start_trading')
    def test_start_command(self, mock_start):
        """Test start trading command."""
        mock_start.return_value = None
        
        result = self.runner.invoke(main_app, [
            "trade", "start",
            "--symbols", "AAPL,GOOGL",
            "--model-path", f"{self.temp_dir}/model",
            "--paper-trading",
            "--initial-capital", "100000"
        ])
        
        assert result.exit_code == 0
        mock_start.assert_called_once()

    @patch('trading_rl_agent.cli.stop_trading')
    def test_stop_command(self, mock_stop):
        """Test stop trading command."""
        mock_stop.return_value = None
        
        result = self.runner.invoke(main_app, [
            "trade", "stop",
            "--session-id", "test_session",
            "--all-sessions"
        ])
        
        assert result.exit_code == 0
        mock_stop.assert_called_once()

    @patch('trading_rl_agent.cli.get_trading_status')
    def test_status_command(self, mock_status):
        """Test trading status command."""
        mock_status.return_value = {"status": "running", "sessions": 1}
        
        result = self.runner.invoke(main_app, [
            "trade", "status",
            "--session-id", "test_session",
            "--detailed"
        ])
        
        assert result.exit_code == 0
        mock_status.assert_called_once()

    @patch('trading_rl_agent.cli.monitor_trading')
    def test_monitor_command(self, mock_monitor):
        """Test trading monitor command."""
        mock_monitor.return_value = None
        
        result = self.runner.invoke(main_app, [
            "trade", "monitor",
            "--session-id", "test_session",
            "--metrics", "all",
            "--interval", "60"
        ])
        
        assert result.exit_code == 0
        mock_monitor.assert_called_once()

    @patch('trading_rl_agent.cli.start_paper_trading')
    def test_paper_command(self, mock_paper):
        """Test paper trading command."""
        mock_paper.return_value = None
        
        result = self.runner.invoke(main_app, [
            "trade", "paper",
            "--symbols", "AAPL,GOOGL,MSFT",
            "--duration", "1d"
        ])
        
        assert result.exit_code == 0
        mock_paper.assert_called_once()


class TestScenarioCLICommands:
    """Test scenario evaluation CLI commands."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('trading_rl_agent.cli.evaluate_scenario')
    def test_scenario_evaluate_command(self, mock_evaluate):
        """Test scenario evaluation command."""
        mock_evaluate.return_value = None
        
        result = self.runner.invoke(main_app, [
            "scenario", "scenario-evaluate",
            "--agent-type", "moving_average",
            "--output-dir", self.temp_dir,
            "--seed", "42",
            "--save-reports",
            "--save-visualizations"
        ])
        
        assert result.exit_code == 0
        mock_evaluate.assert_called_once()

    @patch('trading_rl_agent.cli.compare_scenarios')
    def test_scenario_compare_command(self, mock_compare):
        """Test scenario comparison command."""
        mock_compare.return_value = None
        
        result = self.runner.invoke(main_app, [
            "scenario", "scenario-compare",
            "--output-dir", self.temp_dir,
            "--seed", "42",
            "--save-reports",
            "--save-visualizations"
        ])
        
        assert result.exit_code == 0
        mock_compare.assert_called_once()

    @patch('trading_rl_agent.cli.run_custom_scenario')
    def test_custom_command(self, mock_custom):
        """Test custom scenario command."""
        mock_custom.return_value = None
        
        result = self.runner.invoke(main_app, [
            "scenario", "custom",
            "--agent-type", "moving_average",
            "--scenario-name", "strong_uptrend",
            "--output-dir", self.temp_dir,
            "--seed", "42",
            "--save-reports"
        ])
        
        assert result.exit_code == 0
        mock_custom.assert_called_once()


class TestCLIErrorHandling:
    """Test CLI error handling and user feedback."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = self.runner.invoke(main_app, ["invalid-command"])
        assert result.exit_code != 0
        assert "No such command" in result.output

    def test_missing_required_arguments(self):
        """Test handling of missing required arguments."""
        result = self.runner.invoke(main_app, ["data", "download"])
        assert result.exit_code != 0

    def test_invalid_argument_types(self):
        """Test handling of invalid argument types."""
        result = self.runner.invoke(main_app, [
            "train", "cnn-lstm",
            "--epochs", "invalid"
        ])
        assert result.exit_code != 0

    def test_invalid_file_paths(self):
        """Test handling of invalid file paths."""
        result = self.runner.invoke(main_app, [
            "-c", "nonexistent.yaml",
            "info"
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_network_errors(self):
        """Test handling of network errors."""
        with patch('trading_rl_agent.cli.download_all', side_effect=Exception("Network error")):
            result = self.runner.invoke(main_app, [
                "data", "download-all",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31"
            ])
            assert result.exit_code != 0

    def test_permission_errors(self):
        """Test handling of permission errors."""
        with patch('trading_rl_agent.cli.download_all', side_effect=PermissionError("Permission denied")):
            result = self.runner.invoke(main_app, [
                "data", "download-all",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31"
            ])
            assert result.exit_code != 0

    def test_memory_errors(self):
        """Test handling of memory errors."""
        with patch('trading_rl_agent.cli.train_cnn_lstm', side_effect=MemoryError("Out of memory")):
            result = self.runner.invoke(main_app, [
                "train", "cnn-lstm",
                "--epochs", "10"
            ])
            assert result.exit_code != 0


class TestCLIPerformance:
    """Test CLI performance under load."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_help_command_performance(self):
        """Test help command performance."""
        start_time = time.time()
        result = self.runner.invoke(main_app, ["--help"])
        end_time = time.time()
        
        assert result.exit_code == 0
        assert end_time - start_time < 1.0  # Should complete within 1 second

    def test_version_command_performance(self):
        """Test version command performance."""
        start_time = time.time()
        result = self.runner.invoke(main_app, ["version"])
        end_time = time.time()
        
        assert result.exit_code == 0
        assert end_time - start_time < 0.5  # Should complete within 0.5 seconds

    def test_info_command_performance(self):
        """Test info command performance."""
        start_time = time.time()
        result = self.runner.invoke(main_app, ["info"])
        end_time = time.time()
        
        assert result.exit_code == 0
        assert end_time - start_time < 2.0  # Should complete within 2 seconds

    @patch('trading_rl_agent.cli.download_all')
    def test_data_command_performance(self, mock_download):
        """Test data command performance."""
        mock_download.return_value = None
        
        start_time = time.time()
        result = self.runner.invoke(main_app, [
            "data", "download-all",
            "--start-date", "2023-01-01",
            "--end-date", "2023-01-31"
        ])
        end_time = time.time()
        
        assert result.exit_code == 0
        assert end_time - start_time < 5.0  # Should complete within 5 seconds

    @patch('trading_rl_agent.cli.train_cnn_lstm')
    def test_training_command_performance(self, mock_train):
        """Test training command performance."""
        mock_train.return_value = None
        
        start_time = time.time()
        result = self.runner.invoke(main_app, [
            "train", "cnn-lstm",
            "--epochs", "1"
        ])
        end_time = time.time()
        
        assert result.exit_code == 0
        assert end_time - start_time < 3.0  # Should complete within 3 seconds


class TestCLIMemoryUsage:
    """Test CLI memory usage during operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_usage_help(self):
        """Test memory usage for help command."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = self.runner.invoke(main_app, ["--help"])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result.exit_code == 0
        assert memory_increase < 50 * 1024 * 1024  # Less than 50MB increase

    def test_memory_usage_version(self):
        """Test memory usage for version command."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = self.runner.invoke(main_app, ["version"])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result.exit_code == 0
        assert memory_increase < 10 * 1024 * 1024  # Less than 10MB increase

    @patch('trading_rl_agent.cli.download_all')
    def test_memory_usage_data_operations(self, mock_download):
        """Test memory usage for data operations."""
        import psutil
        import os
        
        mock_download.return_value = None
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = self.runner.invoke(main_app, [
            "data", "download-all",
            "--start-date", "2023-01-01",
            "--end-date", "2023-01-31"
        ])
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        assert result.exit_code == 0
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase


class TestCLIErrorRecovery:
    """Test CLI error recovery scenarios."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_recovery_from_invalid_config(self):
        """Test recovery from invalid config file."""
        # Create invalid config file
        invalid_config = Path(self.temp_dir) / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")
        
        result = self.runner.invoke(main_app, [
            "-c", str(invalid_config),
            "info"
        ])
        
        # Should handle gracefully
        assert result.exit_code != 0

    def test_recovery_from_network_failure(self):
        """Test recovery from network failure."""
        with patch('trading_rl_agent.cli.download_all', side_effect=ConnectionError("Network failure")):
            result = self.runner.invoke(main_app, [
                "data", "download-all",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31"
            ])
            
            assert result.exit_code != 0
            # Should provide meaningful error message
            assert "error" in result.output.lower() or "failed" in result.output.lower()

    def test_recovery_from_disk_full(self):
        """Test recovery from disk full error."""
        with patch('trading_rl_agent.cli.download_all', side_effect=OSError("No space left on device")):
            result = self.runner.invoke(main_app, [
                "data", "download-all",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31"
            ])
            
            assert result.exit_code != 0

    def test_recovery_from_timeout(self):
        """Test recovery from timeout error."""
        with patch('trading_rl_agent.cli.download_all', side_effect=TimeoutError("Operation timed out")):
            result = self.runner.invoke(main_app, [
                "data", "download-all",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31"
            ])
            
            assert result.exit_code != 0


class TestCLIUserExperience:
    """Test CLI user experience validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_help_text_clarity(self):
        """Test clarity of help text."""
        result = self.runner.invoke(main_app, ["--help"])
        assert result.exit_code == 0
        
        # Check for clear descriptions
        assert "Production-grade live trading system" in result.output
        assert "Data pipeline operations" in result.output
        assert "Model training operations" in result.output
        assert "Backtesting operations" in result.output
        assert "Live trading operations" in result.output

    def test_command_descriptions(self):
        """Test command descriptions are clear and helpful."""
        result = self.runner.invoke(main_app, ["data", "--help"])
        assert result.exit_code == 0
        assert "Data pipeline operations" in result.output

        result = self.runner.invoke(main_app, ["train", "--help"])
        assert result.exit_code == 0
        assert "Model training operations" in result.output

        result = self.runner.invoke(main_app, ["backtest", "--help"])
        assert result.exit_code == 0
        assert "Backtesting operations" in result.output

        result = self.runner.invoke(main_app, ["trade", "--help"])
        assert result.exit_code == 0
        assert "Live trading operations" in result.output

    def test_error_message_clarity(self):
        """Test clarity of error messages."""
        result = self.runner.invoke(main_app, ["invalid-command"])
        assert result.exit_code != 0
        
        # Error message should be clear
        assert "No such command" in result.output or "Error" in result.output

    def test_progress_indication(self):
        """Test progress indication for long-running operations."""
        with patch('trading_rl_agent.cli.download_all') as mock_download:
            mock_download.return_value = None
            
            result = self.runner.invoke(main_app, [
                "data", "download-all",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31"
            ])
            
            assert result.exit_code == 0
            # Should provide some feedback
            assert len(result.output) > 0

    def test_consistent_output_format(self):
        """Test consistent output formatting."""
        result = self.runner.invoke(main_app, ["version"])
        assert result.exit_code == 0
        
        # Output should be consistent
        assert "Trading RL Agent" in result.output
        
        result = self.runner.invoke(main_app, ["info"])
        assert result.exit_code == 0
        
        # Should have consistent structure
        assert "System Information" in result.output


class TestCLIIntegration:
    """Test CLI integration with core functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('trading_rl_agent.cli.download_all')
    @patch('trading_rl_agent.cli.process_data')
    @patch('trading_rl_agent.cli.train_cnn_lstm')
    @patch('trading_rl_agent.cli.run_backtest_strategy')
    def test_end_to_end_workflow(self, mock_backtest, mock_train, mock_process, mock_download):
        """Test end-to-end workflow integration."""
        # Setup mocks
        mock_download.return_value = None
        mock_process.return_value = None
        mock_train.return_value = None
        mock_backtest.return_value = None
        
        # Test data download
        result = self.runner.invoke(main_app, [
            "data", "download-all",
            "--start-date", "2023-01-01",
            "--end-date", "2023-12-31"
        ])
        assert result.exit_code == 0
        mock_download.assert_called_once()
        
        # Test data processing
        result = self.runner.invoke(main_app, [
            "data", "process",
            "--output-dir", self.temp_dir
        ])
        assert result.exit_code == 0
        mock_process.assert_called_once()
        
        # Test model training
        result = self.runner.invoke(main_app, [
            "train", "cnn-lstm",
            "--epochs", "1",
            "--output-dir", self.temp_dir
        ])
        assert result.exit_code == 0
        mock_train.assert_called_once()
        
        # Test backtesting
        result = self.runner.invoke(main_app, [
            "backtest", "strategy",
            "--data-path", self.temp_dir,
            "--model-path", f"{self.temp_dir}/model",
            "--output-dir", self.temp_dir
        ])
        assert result.exit_code == 0
        mock_backtest.assert_called_once()

    def test_config_integration(self):
        """Test configuration integration."""
        # Create test config
        config_content = """
        data:
          source: "yfinance"
          symbols: ["AAPL", "GOOGL"]
        training:
          epochs: 10
          batch_size: 32
        backtest:
          initial_capital: 10000
          commission_rate: 0.001
        """
        config_file = Path(self.temp_dir) / "test_config.yaml"
        config_file.write_text(config_content)
        
        with patch('trading_rl_agent.cli.info') as mock_info:
            mock_info.return_value = None
            
            result = self.runner.invoke(main_app, [
                "-c", str(config_file),
                "info"
            ])
            
            assert result.exit_code == 0
            mock_info.assert_called_once()

    def test_logging_integration(self):
        """Test logging integration."""
        with patch('trading_rl_agent.cli.info') as mock_info:
            mock_info.return_value = None
            
            # Test with different verbosity levels
            result = self.runner.invoke(main_app, ["-v", "info"])
            assert result.exit_code == 0
            
            result = self.runner.invoke(main_app, ["-vv", "info"])
            assert result.exit_code == 0
            
            result = self.runner.invoke(main_app, ["-vvv", "info"])
            assert result.exit_code == 0


if __name__ == "__main__":
    pytest.main([__file__])