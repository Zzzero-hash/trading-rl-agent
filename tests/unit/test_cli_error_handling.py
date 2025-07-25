"""
Comprehensive CLI Error Handling and Edge Case Tests.

This module provides extensive testing for CLI error scenarios:
- Network timeouts and connection failures
- Insufficient resources (memory, disk space, GPU)
- Configuration file corruption and missing dependencies
- Interrupted workflows and recovery mechanisms
- API rate limiting and authentication failures
- Data corruption and validation errors
- System resource exhaustion scenarios
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


class TestCLINetworkErrorHandling:
    """Test CLI handling of network-related errors."""

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
    def test_data_pipeline_network_timeout(self, mock_cache, mock_pipeline):
        """Test data pipeline handling of network timeouts."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = TimeoutError("Network timeout during data download")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_connection_error(self, mock_cache, mock_pipeline):
        """Test data pipeline handling of connection errors."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = ConnectionError("Unable to connect to data source")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.sentiment.SentimentAnalyzer")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_sentiment_analysis_api_failure(self, mock_cache, mock_sentiment, mock_pipeline):
        """Test handling of sentiment analysis API failures."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_sentiment_instance = MagicMock()
        mock_sentiment.return_value = mock_sentiment_instance
        mock_sentiment_instance.get_sentiment_features_parallel.side_effect = ConnectionError("Sentiment API unavailable")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        with patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer:
            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--sentiment",
                    "--output-dir", self.temp_dir
                ]
            )

            # Should complete with fallback sentiment values
            assert result.exit_code == 0
            assert "Sentiment analysis failed" in result.output
            assert "Creating default sentiment features" in result.output

    @patch("trade_agent.cli.start_trading")
    def test_trading_broker_api_failure(self, mock_start_trading):
        """Test handling of broker API connection failures."""
        mock_start_trading.side_effect = ConnectionError("Unable to connect to broker API")

        result = self.runner.invoke(
            main_app,
            [
                "trade", "start",
                "--symbols", "AAPL",
                "--model-path", self.temp_dir,
                "--paper-trading"
            ]
        )

        assert result.exit_code != 0
        # Should provide meaningful error message
        assert "error" in result.output.lower() or "failed" in result.output.lower()

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL"],
            "close": [150.0],
            "volume": [1000000],
            "date": pd.to_datetime(["2023-01-01"])
        })


class TestCLIResourceErrorHandling:
    """Test CLI handling of resource limitation errors."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = self._create_mock_dataset()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_dataset(self):
        """Create a mock dataset file."""
        import pandas as pd

        data_path = Path(self.temp_dir) / "dataset.csv"
        data = {
            "date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "symbol": ["AAPL"] * 50,
            "close": [150.0 + i * 0.1 for i in range(50)],
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        return data_path

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_disk_space_error(self, mock_cache, mock_pipeline):
        """Test data pipeline handling of insufficient disk space."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = OSError("No space left on device")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    def test_cnn_lstm_memory_error(self, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training handling of memory errors."""
        mock_init_ray.return_value = None
        mock_load_data.side_effect = MemoryError("Unable to allocate memory for training data")

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(self.data_path),
                "--epochs", "5",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Error during CNN+LSTM training" in result.output

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    def test_cnn_lstm_gpu_unavailable(self, mock_trainer_class, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training when GPU is requested but unavailable."""
        mock_init_ray.return_value = None
        mock_sequences, mock_targets = self._create_mock_sequences_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train_from_dataset.side_effect = RuntimeError("CUDA out of memory")

        with patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config") as mock_model_config, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config") as mock_training_config:

            mock_model_config.return_value = {"input_dim": 10}
            mock_training_config.return_value = {"epochs": 5}

            result = self.runner.invoke(
                main_app,
                [
                    "train", "cnn-lstm",
                    str(self.data_path),
                    "--gpu",
                    "--epochs", "5",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 1
            assert "Error during CNN+LSTM training" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_permission_error(self, mock_cache, mock_pipeline):
        """Test data pipeline handling of permission errors."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = PermissionError("Permission denied")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--output-dir", "/root/protected_dir"  # Likely to cause permission error
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    def _create_mock_sequences_targets(self):
        """Create mock sequences and targets for training."""
        import numpy as np
        sequences = np.random.rand(50, 30, 10)
        targets = np.random.rand(50, 1)
        return sequences, targets


class TestCLIConfigurationErrorHandling:
    """Test CLI handling of configuration-related errors."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_yaml_config_file(self):
        """Test handling of invalid YAML configuration files."""
        # Create invalid YAML file
        invalid_config = Path(self.temp_dir) / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content: [unclosed")

        result = self.runner.invoke(
            main_app,
            [
                "--config", str(invalid_config),
                "info"
            ]
        )

        assert result.exit_code == 1
        assert "Error loading configuration" in result.output

    def test_missing_config_file(self):
        """Test handling of missing configuration files."""
        nonexistent_config = Path(self.temp_dir) / "nonexistent.yaml"

        result = self.runner.invoke(
            main_app,
            [
                "--config", str(nonexistent_config),
                "info"
            ]
        )

        assert result.exit_code == 1
        assert "Configuration file not found" in result.output

    def test_corrupted_config_file(self):
        """Test handling of corrupted configuration files."""
        # Create corrupted config file
        corrupted_config = Path(self.temp_dir) / "corrupted.yaml"
        corrupted_config.write_bytes(b"\x00\x01\x02\x03\x04\x05")  # Binary data

        result = self.runner.invoke(
            main_app,
            [
                "--config", str(corrupted_config),
                "info"
            ]
        )

        assert result.exit_code == 1
        assert "Error loading configuration" in result.output

    def test_invalid_env_file(self):
        """Test handling of invalid environment files."""
        # Create invalid .env file
        invalid_env = Path(self.temp_dir) / ".env"
        invalid_env.write_text("INVALID_ENV_FORMAT_NO_EQUALS_SIGN")

        result = self.runner.invoke(
            main_app,
            [
                "--env-file", str(invalid_env),
                "info"
            ]
        )

        # Should handle gracefully or provide clear error
        # Exact behavior depends on python-dotenv implementation
        assert result.exit_code in [0, 1]

    def test_missing_env_file(self):
        """Test handling of missing environment files."""
        nonexistent_env = Path(self.temp_dir) / "nonexistent.env"

        result = self.runner.invoke(
            main_app,
            [
                "--env-file", str(nonexistent_env),
                "info"
            ]
        )

        assert result.exit_code == 1
        assert "Environment file not found" in result.output

    def test_config_with_missing_dependencies(self):
        """Test handling of configuration with missing required fields."""
        # Create config with missing required fields
        incomplete_config = Path(self.temp_dir) / "incomplete.yaml"
        incomplete_config.write_text("""
        # Missing required data section
        training:
          epochs: 10
        """)

        result = self.runner.invoke(
            main_app,
            [
                "--config", str(incomplete_config),
                "info"
            ]
        )

        # Should handle gracefully with default values or provide clear error
        # Based on the CLI implementation, it uses minimal settings on error
        assert result.exit_code in [0, 1]


class TestCLIInterruptionHandling:
    """Test CLI handling of interrupted workflows."""

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
    def test_data_pipeline_keyboard_interrupt(self, mock_cache, mock_pipeline):
        """Test data pipeline handling of keyboard interrupts."""
        # Setup mocks
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
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.EnhancedCNNLSTMTrainer")
    def test_training_system_exit(self, mock_trainer_class, mock_load_data, mock_init_ray):
        """Test training handling of system exit signals."""
        mock_init_ray.return_value = None
        mock_sequences, mock_targets = self._create_mock_sequences_targets()
        mock_load_data.return_value = (mock_sequences, mock_targets)

        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train_from_dataset.side_effect = SystemExit("System shutdown")

        with patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_model_config") as mock_model_config, \
             patch("trade_agent.training.train_cnn_lstm_enhanced.create_enhanced_training_config") as mock_training_config:

            mock_model_config.return_value = {"input_dim": 10}
            mock_training_config.return_value = {"epochs": 5}

            data_path = self._create_mock_dataset()

            # SystemExit should propagate and cause test runner to exit
            with pytest.raises(SystemExit):
                self.runner.invoke(
                    main_app,
                    [
                        "train", "cnn-lstm",
                        str(data_path),
                        "--epochs", "5",
                        "--output-dir", self.temp_dir
                    ]
                )

    def _create_mock_dataset(self):
        """Create a mock dataset file."""
        import pandas as pd

        data_path = Path(self.temp_dir) / "dataset.csv"
        data = {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "symbol": ["AAPL"] * 20,
            "close": [150.0 + i * 0.1 for i in range(20)],
        }
        df = pd.DataFrame(data)
        df.to_csv(data_path, index=False)
        return data_path

    def _create_mock_sequences_targets(self):
        """Create mock sequences and targets for training."""
        import numpy as np
        sequences = np.random.rand(20, 10, 5)
        targets = np.random.rand(20, 1)
        return sequences, targets


class TestCLIDataValidationErrors:
    """Test CLI handling of data validation and corruption errors."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    def test_cnn_lstm_corrupted_data_file(self, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training with corrupted data file."""
        mock_init_ray.return_value = None
        mock_load_data.side_effect = pd.errors.ParserError("Error parsing CSV file")

        # Create corrupted CSV file
        corrupted_data = Path(self.temp_dir) / "corrupted.csv"
        corrupted_data.write_text("invalid,csv,format\nwith,missing,columns\n")

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(corrupted_data),
                "--epochs", "5",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Error during CNN+LSTM training" in result.output

    @patch("trade_agent.training.train_cnn_lstm_enhanced.init_ray_cluster")
    @patch("trade_agent.training.train_cnn_lstm_enhanced.load_and_preprocess_csv_data")
    def test_cnn_lstm_empty_data_file(self, mock_load_data, mock_init_ray):
        """Test CNN+LSTM training with empty data file."""
        mock_init_ray.return_value = None
        mock_load_data.side_effect = ValueError("Dataset is empty")

        # Create empty CSV file
        empty_data = Path(self.temp_dir) / "empty.csv"
        empty_data.write_text("")

        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(empty_data),
                "--epochs", "5",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Error during CNN+LSTM training" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_invalid_symbols(self, mock_cache, mock_pipeline):
        """Test data pipeline with invalid stock symbols."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = ValueError("Invalid symbols: INVALID123, FAKE456")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "INVALID123,FAKE456",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_data_pipeline_invalid_date_range(self, mock_cache, mock_pipeline):
        """Test data pipeline with invalid date range."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.side_effect = ValueError("Start date cannot be after end date")

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--start-date", "2023-12-31",
                "--end-date", "2023-01-01",  # End before start
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output


class TestCLIRateLimitingErrors:
    """Test CLI handling of API rate limiting errors."""

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
    def test_data_pipeline_rate_limit_exceeded(self, mock_cache, mock_pipeline):
        """Test data pipeline handling of rate limit errors."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        # Simulate rate limiting error
        rate_limit_error = Exception("Rate limit exceeded. Please try again later.")
        rate_limit_error.response = MagicMock()
        rate_limit_error.response.status_code = 429
        mock_pipeline_instance.download_data_parallel.side_effect = rate_limit_error

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL,MSFT,TSLA,NVDA",  # Many symbols to trigger rate limiting
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "Auto-processing pipeline failed" in result.output

    @patch("trade_agent.data.sentiment.SentimentAnalyzer")
    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_sentiment_analysis_rate_limit(self, mock_cache, mock_pipeline, mock_sentiment):
        """Test sentiment analysis handling of rate limit errors."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_sentiment_instance = MagicMock()
        mock_sentiment.return_value = mock_sentiment_instance

        # Simulate sentiment API rate limiting
        rate_limit_error = Exception("Sentiment API rate limit exceeded")
        rate_limit_error.response = MagicMock()
        rate_limit_error.response.status_code = 429
        mock_sentiment_instance.get_sentiment_features_parallel.side_effect = rate_limit_error

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        with patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer:
            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--sentiment",
                    "--output-dir", self.temp_dir
                ]
            )

            # Should complete with fallback sentiment values
            assert result.exit_code == 0
            assert "Sentiment analysis failed" in result.output

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
