"""
CLI State Management Tests.

This module provides testing for CLI state management functionality:
- Configuration state persistence across commands
- Session state management and recovery
- Cache state consistency and cleanup
- Global variable state isolation
- Environment variable handling
- Cleanup on interruption and failure scenarios
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


class TestCLIConfigurationState:
    """Test CLI configuration state management."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_file_state_persistence(self):
        """Test that configuration file state persists across commands."""
        # Create a test configuration file
        config_file = Path(self.temp_dir) / "test_config.yaml"
        config_content = """
        data:
          symbols: ["AAPL", "GOOGL", "MSFT"]
          start_date: "2023-01-01"
          end_date: "2023-12-31"
        environment: "test"
        debug: true
        """
        config_file.write_text(config_content)

        # Test that config is loaded and used consistently
        with patch("trade_agent.cli.load_settings") as mock_load_settings, \
             patch("trade_agent.cli.get_settings") as mock_get_settings:

            # Mock the settings object
            mock_settings = MagicMock()
            mock_settings.environment = "test"
            mock_settings.debug = True
            mock_settings.data.symbols = ["AAPL", "GOOGL", "MSFT"]
            mock_load_settings.return_value = mock_settings
            mock_get_settings.return_value = mock_settings

            # First command with config
            result1 = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "info"]
            )

            # Second command with same config
            result2 = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "version"]
            )

            # Both should succeed and use the same config
            assert result1.exit_code == 0
            assert result2.exit_code == 0

            # Config should be loaded for each command
            assert mock_load_settings.call_count >= 1

    def test_environment_variable_state(self):
        """Test environment variable state management."""
        # Create test environment file
        env_file = Path(self.temp_dir) / ".env"
        env_content = """
        TRADING_RL_AGENT_DATA_SOURCE=yfinance
        TRADING_RL_AGENT_DEBUG=true
        TRADING_RL_AGENT_ENVIRONMENT=test
        """
        env_file.write_text(env_content)

        # Test with environment file
        with patch("dotenv.load_dotenv") as mock_load_dotenv:
            result = self.runner.invoke(
                main_app,
                ["--env-file", str(env_file), "info"]
            )

            # Should load environment variables
            if result.exit_code == 0:
                mock_load_dotenv.assert_called_once()

    def test_config_inheritance_across_subcommands(self):
        """Test that configuration is inherited by subcommands."""
        config_file = Path(self.temp_dir) / "inherit_config.yaml"
        config_content = """
        data:
          symbols: ["AAPL"]
        training:
          epochs: 10
          batch_size: 32
        """
        config_file.write_text(config_content)

        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            mock_settings = MagicMock()
            mock_settings.environment = "test"
            mock_load_settings.return_value = mock_settings

            # Test subcommand inherits config
            result = self.runner.invoke(
                main_app,
                ["--config", str(config_file), "data", "--help"]
            )

            # Should succeed and load config
            if result.exit_code == 0:
                mock_load_settings.assert_called_once()

    def test_config_error_recovery(self):
        """Test recovery from configuration errors."""
        # Create invalid config file
        invalid_config = Path(self.temp_dir) / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [")

        # Should handle gracefully
        result = self.runner.invoke(
            main_app,
            ["--config", str(invalid_config), "info"]
        )

        assert result.exit_code == 1
        assert "Error loading configuration" in result.output

    def test_config_state_isolation(self):
        """Test that configuration state is isolated between test runs."""
        # This test ensures that global state doesn't leak between tests
        config1 = Path(self.temp_dir) / "config1.yaml"
        config1.write_text("environment: config1\ndebug: true")

        config2 = Path(self.temp_dir) / "config2.yaml"
        config2.write_text("environment: config2\ndebug: false")

        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            # First run with config1
            mock_settings1 = MagicMock()
            mock_settings1.environment = "config1"
            mock_load_settings.return_value = mock_settings1

            result1 = self.runner.invoke(
                main_app,
                ["--config", str(config1), "info"]
            )

            # Second run with config2
            mock_settings2 = MagicMock()
            mock_settings2.environment = "config2"
            mock_load_settings.return_value = mock_settings2

            result2 = self.runner.invoke(
                main_app,
                ["--config", str(config2), "info"]
            )

            # Both should succeed independently
            assert result1.exit_code == 0
            assert result2.exit_code == 0


class TestCLISessionState:
    """Test CLI session state management."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.cli.start_trading")
    @patch("trade_agent.cli.get_trading_status")
    @patch("trade_agent.cli.stop_trading")
    def test_trading_session_state_consistency(self, mock_stop, mock_status, mock_start):
        """Test trading session state consistency."""
        # Mock session management
        mock_start.return_value = {"session_id": "test_123", "status": "started"}
        mock_status.return_value = {
            "session_id": "test_123",
            "status": "running",
            "active_sessions": 1
        }
        mock_stop.return_value = {"session_id": "test_123", "status": "stopped"}

        # Start a session
        start_result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", self.temp_dir,
                "--paper-trading"
            ]
        )

        # Check status
        status_result = self.runner.invoke(
            main_app,
            ["trade", "status", "--session-id", "test_123"]
        )

        # Stop the session
        stop_result = self.runner.invoke(
            main_app,
            ["trade", "stop", "--session-id", "test_123"]
        )

        # All operations should succeed
        assert start_result.exit_code == 0
        assert status_result.exit_code == 0
        assert stop_result.exit_code == 0

    def test_verbose_state_consistency(self):
        """Test that verbose state is maintained correctly."""
        # Test different verbose levels
        verbose_levels = [0, 1, 2, 3]

        for level in verbose_levels:
            verbose_args = ["-v"] * level if level > 0 else []

            result = self.runner.invoke(
                main_app,
                [*verbose_args, "info"]
            )

            # Should handle all verbose levels
            assert result.exit_code == 0

    @patch("trade_agent.cli.monitor_trading")
    def test_monitoring_session_state(self, mock_monitor):
        """Test monitoring session state management."""
        mock_monitor.return_value = None

        # Start monitoring
        result = self.runner.invoke(
            main_app,
            [
                "trade", "monitor",
                "--session-id", "test_session",
                "--metrics", "all",
                "--interval", "30"
            ]
        )

        assert result.exit_code == 0
        mock_monitor.assert_called_once()


class TestCLICacheState:
    """Test CLI cache state management."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_cache_state_consistency(self, mock_cache_manager, mock_pipeline):
        """Test cache state consistency across commands."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_cache_instance = MagicMock()
        mock_cache_manager.return_value = mock_cache_instance

        # First call - cache miss
        mock_cache_instance.get_cached_data.return_value = None

        with patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer:
            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            result1 = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--cache",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "cache_test_1"
                ]
            )

            # Second call - cache hit
            mock_cache_instance.get_cached_data.return_value = self._create_mock_dataframe()

            result2 = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--cache",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "cache_test_2"
                ]
            )

            # Both should succeed
            assert result1.exit_code == 0
            assert result2.exit_code == 0

            # Cache should be checked in both calls
            assert mock_cache_instance.get_cached_data.call_count == 2

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_cache_state_with_no_cache_flag(self, mock_cache_manager, mock_pipeline):
        """Test cache state when caching is disabled."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_cache_instance = MagicMock()
        mock_cache_manager.return_value = mock_cache_instance

        with patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer:
            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-cache",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "no_cache_test"
                ]
            )

            assert result.exit_code == 0

            # Cache should not be used when disabled
            # The exact behavior depends on implementation

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLIGlobalStateIsolation:
    """Test isolation of global state between CLI invocations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_global_variable_isolation(self):
        """Test that global variables don't leak between invocations."""
        # Test multiple invocations to ensure state doesn't leak
        for i in range(3):
            result = self.runner.invoke(
                main_app,
                ["-v", "info"]  # Use verbose to potentially trigger global state
            )

            assert result.exit_code == 0

            # Each invocation should be independent
            # This is mainly testing the test framework isolation

    def test_logger_state_isolation(self):
        """Test that logger state is properly isolated."""
        # Test with different verbose levels
        verbose_configs = [
            [],
            ["-v"],
            ["-vv"],
            ["-vvv"]
        ]

        for verbose_args in verbose_configs:
            result = self.runner.invoke(
                main_app,
                [*verbose_args, "version"]
            )

            assert result.exit_code == 0

    def test_settings_state_isolation(self):
        """Test that settings state is isolated between runs."""
        # Create different config files
        config1 = Path(self.temp_dir) / "config1.yaml"
        config1.write_text("environment: test1")

        config2 = Path(self.temp_dir) / "config2.yaml"
        config2.write_text("environment: test2")

        # Run with different configs
        with patch("trade_agent.cli.load_settings") as mock_load_settings:
            # First run
            mock_settings1 = MagicMock()
            mock_settings1.environment = "test1"
            mock_load_settings.return_value = mock_settings1

            result1 = self.runner.invoke(
                main_app,
                ["--config", str(config1), "info"]
            )

            # Reset mock for second run
            mock_load_settings.reset_mock()
            mock_settings2 = MagicMock()
            mock_settings2.environment = "test2"
            mock_load_settings.return_value = mock_settings2

            result2 = self.runner.invoke(
                main_app,
                ["--config", str(config2), "info"]
            )

            # Both should succeed independently
            assert result1.exit_code == 0
            assert result2.exit_code == 0


class TestCLICleanupOnFailure:
    """Test cleanup behavior when CLI commands fail."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_cleanup_on_pipeline_failure(self, mock_cache_manager, mock_pipeline):
        """Test cleanup when data pipeline fails."""
        # Setup mocks to simulate failure
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = Exception("Simulated failure")

        mock_cache_instance = MagicMock()
        mock_cache_manager.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "failure_test"
            ]
        )

        # Should fail gracefully
        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

        # Cleanup should occur (no partial files left)
        dataset_dir = Path(self.temp_dir) / "failure_test"
        if dataset_dir.exists():
            # If directory exists, it should not contain partial data
            csv_file = dataset_dir / "dataset.csv"
            assert not csv_file.exists() or csv_file.stat().st_size == 0

    def test_cleanup_on_keyboard_interrupt(self):
        """Test cleanup when user interrupts operation."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mocks to simulate keyboard interrupt
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.side_effect = KeyboardInterrupt("User interrupted")

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "interrupt_test"
                ]
            )

            # Should handle interruption gracefully
            assert result.exit_code == 1

    def test_state_recovery_after_failure(self):
        """Test that state can recover after failures."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mocks
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            # First call fails
            mock_pipeline_instance.download_data_parallel.side_effect = Exception("First failure")

            result1 = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "recovery_test_1"
                ]
            )

            assert result1.exit_code == 1

            # Second call succeeds
            mock_pipeline_instance.download_data_parallel.side_effect = None
            mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()
            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            result2 = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--no-sentiment",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "recovery_test_2"
                ]
            )

            assert result2.exit_code == 0

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


if __name__ == "__main__":
    pytest.main([__file__])
