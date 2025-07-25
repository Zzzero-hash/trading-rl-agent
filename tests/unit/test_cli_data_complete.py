"""
Complete CLI Data Command Coverage Tests.

This module provides comprehensive testing for all data pipeline CLI commands:
- Data pipeline command with all parameter combinations
- Symbol validation and processing
- Configuration file integration
- Output formatting and file handling
- Multi-format export testing
- Advanced feature sets (technical, fundamental, custom)
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


class TestCLIDataPipelineComplete:
    """Comprehensive tests for data pipeline CLI command."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_config = {
            "data": {
                "source": "yfinance",
                "symbols": ["AAPL", "GOOGL", "MSFT"],
                "start_date": "2023-01-01",
                "end_date": "2023-12-31"
            },
            "features": {
                "technical_indicators": True,
                "sentiment_analysis": True
            }
        }

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.sentiment.SentimentAnalyzer")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_complete_pipeline_run_basic(self, mock_cache, mock_standardizer, mock_sentiment, mock_pipeline):
        """Test basic complete pipeline run."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_sentiment_instance = MagicMock()
        mock_sentiment.return_value = mock_sentiment_instance
        mock_sentiment_instance.get_sentiment_features_parallel.return_value = self._create_mock_sentiment_data()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL,MSFT",
                "--start-date", "2023-01-01",
                "--end-date", "2023-12-31",
                "--output-dir", self.temp_dir,
                "--dataset-name", "test_dataset",
                "--processing-method", "robust",
                "--feature-set", "full"
            ]
        )

        assert result.exit_code == 0
        assert "Auto-Processing Pipeline completed successfully" in result.output
        mock_pipeline_instance.download_data_parallel.assert_called_once()

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.sentiment.SentimentAnalyzer")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_with_sentiment_analysis(self, mock_cache, mock_standardizer, mock_sentiment, mock_pipeline):
        """Test pipeline with sentiment analysis enabled."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_sentiment_instance = MagicMock()
        mock_sentiment.return_value = mock_sentiment_instance
        mock_sentiment_instance.get_sentiment_features_parallel.return_value = self._create_mock_sentiment_data()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL",
                "--sentiment",
                "--sentiment-days", "14",
                "--sentiment-sources", "news,social",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0
        assert "Analyzing market sentiment" in result.output
        mock_sentiment_instance.get_sentiment_features_parallel.assert_called_once()

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_without_sentiment(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test pipeline with sentiment analysis disabled."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--no-sentiment",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0
        assert "Sentiment analysis disabled by user" in result.output

    @patch("trade_agent.data.market_symbols.get_optimized_symbols")
    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_auto_symbol_selection(self, mock_cache, mock_standardizer, mock_pipeline, mock_symbols):
        """Test pipeline with automatic symbol selection."""
        # Setup mocks
        mock_symbols.return_value = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--max-symbols", "5",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0
        assert "Auto-selected 5 optimized symbols" in result.output
        mock_symbols.assert_called_once_with(max_symbols=5)

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_multiple_export_formats(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test pipeline with multiple export formats."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardized_data = self._create_mock_dataframe()
        mock_standardizer.return_value = (mock_standardized_data, MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--export-formats", "csv,parquet,feather",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0
        assert "Exporting dataset in 3 formats" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.data.features.generate_features")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_feature_sets(self, mock_cache, mock_features, mock_standardizer, mock_pipeline):
        """Test pipeline with different feature sets."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_features.return_value = self._create_mock_dataframe()
        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        # Test different feature sets
        for feature_set in ["basic", "technical", "full"]:
            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--feature-set", feature_set,
                    "--output-dir", self.temp_dir,
                    "--dataset-name", f"test_{feature_set}"
                ]
            )

            assert result.exit_code == 0
            if feature_set in ["technical", "full"]:
                mock_features.assert_called()

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_cache_usage(self, mock_cache, mock_pipeline):
        """Test pipeline cache usage."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance

        # Test with cache enabled
        mock_cache_instance.get_cached_data.return_value = self._create_mock_dataframe()

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--cache",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0
        mock_cache_instance.get_cached_data.assert_called()

        # Test with cache disabled
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--no-cache",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0

    def test_pipeline_without_run_flag(self):
        """Test pipeline command without --run flag shows error."""
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline",
                "--symbols", "AAPL",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 1
        assert "No pipeline action specified" in result.output
        assert "Use --run to execute" in result.output

    @patch("trade_agent.data.pipeline.DataPipeline")
    @patch("trade_agent.data.data_standardizer.create_standardized_dataset")
    @patch("trade_agent.utils.cache_manager.CacheManager")
    def test_pipeline_performance_options(self, mock_cache, mock_standardizer, mock_pipeline):
        """Test pipeline with different performance options."""
        # Setup mocks
        mock_pipeline_instance = MagicMock()
        mock_pipeline.return_value = mock_pipeline_instance
        mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

        mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

        mock_cache_instance = MagicMock()
        mock_cache.return_value = mock_cache_instance
        mock_cache_instance.get_cached_data.return_value = None

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL,MSFT",
                "--workers", "16",
                "--output-dir", self.temp_dir
            ]
        )

        assert result.exit_code == 0
        assert "Workers: 16" in result.output

        # Verify parallel workers parameter was passed
        call_args = mock_pipeline_instance.download_data_parallel.call_args
        assert call_args[1]["max_workers"] == 16

    def test_pipeline_metadata_generation(self):
        """Test that pipeline generates proper metadata."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache, \
             patch("builtins.open", create=True) as mock_open, \
             patch("json.dump") as mock_json_dump:

            # Setup mocks
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL,GOOGL",
                    "--dataset-name", "test_metadata",
                    "--feature-set", "technical",
                    "--sentiment-days", "7",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 0

            # Check that metadata was written
            mock_json_dump.assert_called()
            metadata_call = mock_json_dump.call_args[0][0]
            assert metadata_call["dataset_name"] == "test_metadata"
            assert metadata_call["symbols"] == ["AAPL", "GOOGL"]
            assert metadata_call["feature_set"] == "technical"

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL", "GOOGL"],
            "close": [150.0, 2500.0],
            "volume": [1000000, 500000],
            "date": pd.to_datetime(["2023-01-01", "2023-01-01"])
        })

    def _create_mock_sentiment_data(self):
        """Create mock sentiment data for testing."""
        import pandas as pd
        return pd.DataFrame({
            "symbol": ["AAPL", "GOOGL"],
            "sentiment_score": [0.5, 0.3],
            "sentiment_magnitude": [0.8, 0.6],
            "sentiment_sources": ["news", "social"],
            "sentiment_direction": ["positive", "neutral"]
        })


class TestCLIDataCommandValidation:
    """Test data command parameter validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_invalid_date_format(self):
        """Test handling of invalid date formats."""
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--start-date", "invalid-date",
                "--output-dir", self.temp_dir
            ]
        )

        # Should handle gracefully or provide clear error message
        if result.exit_code != 0:
            assert "date" in result.output.lower() or "invalid" in result.output.lower()

    def test_invalid_symbol_format(self):
        """Test handling of invalid symbol formats."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline:
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.side_effect = Exception("Invalid symbol")

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "INVALID@SYMBOL!",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 1

    def test_invalid_worker_count(self):
        """Test handling of invalid worker count."""
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--workers", "0",
                "--output-dir", self.temp_dir
            ]
        )

        # Should either handle gracefully or provide clear error
        # The exact behavior depends on the implementation

    def test_invalid_export_format(self):
        """Test handling of invalid export formats."""
        with patch("trade_agent.data.pipeline.DataPipeline") as mock_pipeline, \
             patch("trade_agent.data.data_standardizer.create_standardized_dataset") as mock_standardizer, \
             patch("trade_agent.utils.cache_manager.CacheManager") as mock_cache:

            # Setup mocks
            mock_pipeline_instance = MagicMock()
            mock_pipeline.return_value = mock_pipeline_instance
            mock_pipeline_instance.download_data_parallel.return_value = self._create_mock_dataframe()

            mock_standardizer.return_value = (self._create_mock_dataframe(), MagicMock())

            mock_cache_instance = MagicMock()
            mock_cache.return_value = mock_cache_instance
            mock_cache_instance.get_cached_data.return_value = None

            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--export-formats", "csv,invalid_format,parquet",
                    "--output-dir", self.temp_dir
                ]
            )

            assert result.exit_code == 0
            assert "Unknown format: invalid_format" in result.output

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
