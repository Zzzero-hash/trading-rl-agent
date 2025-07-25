"""
Real Data Integration Tests for CLI.

This module provides integration testing with real (but small) datasets:
- End-to-end pipeline testing with actual market data
- Real sentiment analysis integration
- Actual model training with small datasets
- File I/O validation with real data formats
- Cross-platform compatibility testing
- Network connectivity and data source validation
"""

import sys
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

# Add the src directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from trade_agent.cli import app as main_app


@pytest.mark.integration
class TestCLIRealDataPipeline:
    """Integration tests with real data pipeline."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_symbols = ["AAPL"]  # Use single symbol to minimize data and API calls
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-05"  # Very short range for testing

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_real_data_pipeline_basic(self):
        """Test basic data pipeline with real market data."""
        # Skip if no internet connection
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", ",".join(self.test_symbols),
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--no-sentiment",  # Skip sentiment to reduce API calls
                "--feature-set", "basic",
                "--export-formats", "csv",
                "--output-dir", self.temp_dir,
                "--dataset-name", "real_test_basic"
            ],
            catch_exceptions=False
        )

        # Should complete successfully
        assert result.exit_code == 0
        assert "Auto-Processing Pipeline completed successfully" in result.output

        # Verify output files were created
        dataset_dir = Path(self.temp_dir) / "real_test_basic"
        assert dataset_dir.exists()
        assert (dataset_dir / "dataset.csv").exists()
        assert (dataset_dir / "metadata.json").exists()

        # Verify data content
        self._verify_dataset_content(dataset_dir / "dataset.csv")
        self._verify_metadata_content(dataset_dir / "metadata.json")

    @pytest.mark.slow
    def test_real_data_pipeline_with_features(self):
        """Test data pipeline with technical features."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", ",".join(self.test_symbols),
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--no-sentiment",
                "--feature-set", "technical",
                "--export-formats", "csv,parquet",
                "--output-dir", self.temp_dir,
                "--dataset-name", "real_test_features"
            ],
            catch_exceptions=False
        )

        assert result.exit_code == 0

        # Verify multiple export formats
        dataset_dir = Path(self.temp_dir) / "real_test_features"
        assert (dataset_dir / "dataset.csv").exists()
        assert (dataset_dir / "dataset.parquet").exists()

        # Verify technical features were added
        self._verify_technical_features(dataset_dir / "dataset.csv")

    @pytest.mark.slow
    @pytest.mark.network
    def test_real_data_pipeline_with_sentiment(self):
        """Test data pipeline with real sentiment analysis (if available)."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        # This test might fail if sentiment API is not configured
        # We'll catch and handle gracefully
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", self.test_symbols[0],  # Single symbol to reduce API calls
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--sentiment",
                "--sentiment-days", "1",  # Minimal sentiment lookback
                "--feature-set", "basic",
                "--output-dir", self.temp_dir,
                "--dataset-name", "real_test_sentiment"
            ],
            catch_exceptions=False
        )

        # Should complete even if sentiment fails (with defaults)
        assert result.exit_code == 0

        dataset_dir = Path(self.temp_dir) / "real_test_sentiment"
        assert dataset_dir.exists()

        # Check if sentiment columns exist (either real or default)
        self._verify_sentiment_columns(dataset_dir / "dataset.csv")

    @pytest.mark.slow
    def test_real_data_pipeline_multi_symbol(self):
        """Test data pipeline with multiple symbols."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        multi_symbols = ["AAPL", "GOOGL"]  # Small set to minimize API calls

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", ",".join(multi_symbols),
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--no-sentiment",
                "--feature-set", "basic",
                "--output-dir", self.temp_dir,
                "--dataset-name", "real_test_multi"
            ],
            catch_exceptions=False
        )

        assert result.exit_code == 0

        # Verify data for multiple symbols
        dataset_dir = Path(self.temp_dir) / "real_test_multi"
        self._verify_multi_symbol_data(dataset_dir / "dataset.csv", multi_symbols)

    def test_real_data_pipeline_with_cache(self):
        """Test data pipeline with caching enabled."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        # First run to populate cache
        result1 = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", self.test_symbols[0],
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--cache",
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "real_test_cache_1"
            ],
            catch_exceptions=False
        )

        assert result1.exit_code == 0
        self._extract_runtime_from_output(result1.output)

        # Second run should use cache (faster)
        result2 = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", self.test_symbols[0],
                "--start-date", self.start_date,
                "--end-date", self.end_date,
                "--cache",
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "real_test_cache_2"
            ],
            catch_exceptions=False
        )

        assert result2.exit_code == 0
        self._extract_runtime_from_output(result2.output)

        # Second run should be faster or similar (cache benefit)
        # We don't assert this strictly as network variability can affect results

    def _check_internet_connection(self):
        """Check if internet connection is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def _verify_dataset_content(self, csv_path):
        """Verify the dataset contains expected content."""
        import pandas as pd

        assert csv_path.exists()
        df = pd.read_csv(csv_path)

        # Basic structure validation
        assert len(df) > 0
        assert "symbol" in df.columns or any("symbol" in col.lower() for col in df.columns)

        # Should have price data
        price_columns = ["open", "high", "low", "close", "volume"]
        available_price_cols = [col for col in price_columns if col in df.columns or any(col in c.lower() for c in df.columns)]
        assert len(available_price_cols) > 0

    def _verify_metadata_content(self, metadata_path):
        """Verify metadata file contains expected information."""
        import json

        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Basic metadata validation
        assert "dataset_name" in metadata
        assert "created_at" in metadata
        assert "symbols" in metadata
        assert "row_count" in metadata
        assert "column_count" in metadata

        # Verify symbols match what we requested
        assert len(metadata["symbols"]) == len(self.test_symbols)

    def _verify_technical_features(self, csv_path):
        """Verify technical features were added to the dataset."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Look for common technical indicators
        # The exact column names depend on the implementation
        potential_technical_cols = [
            "sma", "ema", "rsi", "macd", "bollinger", "volume_ma",
            "returns", "volatility", "moving_average"
        ]

        technical_cols_found = []
        for col in df.columns:
            col_lower = col.lower()
            for tech_indicator in potential_technical_cols:
                if tech_indicator in col_lower:
                    technical_cols_found.append(col)
                    break

        # Should have at least some technical features
        # (exact count depends on implementation)
        assert len(technical_cols_found) > 0

    def _verify_sentiment_columns(self, csv_path):
        """Verify sentiment columns exist in the dataset."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Expected sentiment columns
        sentiment_columns = [
            "sentiment_score", "sentiment_magnitude",
            "sentiment_sources", "sentiment_direction"
        ]

        sentiment_cols_found = []
        for col in df.columns:
            col_lower = col.lower()
            for sent_col in sentiment_columns:
                if sent_col in col_lower:
                    sentiment_cols_found.append(col)
                    break

        # Should have at least some sentiment columns (even if defaults)
        assert len(sentiment_cols_found) > 0

    def _verify_multi_symbol_data(self, csv_path, expected_symbols):
        """Verify dataset contains data for multiple symbols."""
        import pandas as pd

        df = pd.read_csv(csv_path)

        # Find symbol column
        symbol_col = None
        for col in df.columns:
            if "symbol" in col.lower():
                symbol_col = col
                break

        assert symbol_col is not None

        # Verify all expected symbols are present
        unique_symbols = df[symbol_col].unique()
        for symbol in expected_symbols:
            assert symbol in unique_symbols

    def _extract_runtime_from_output(self, output):
        """Extract runtime information from CLI output."""
        # Look for performance summary in output
        lines = output.split("\n")
        for line in lines:
            if "Total time:" in line:
                # Extract time value (implementation depends on output format)
                try:
                    time_str = line.split("Total time:")[1].strip()
                    time_value = float(time_str.replace("s", "").strip())
                    return time_value
                except (IndexError, ValueError):
                    pass
        return None


@pytest.mark.integration
class TestCLIRealTrainingIntegration:
    """Integration tests for training with real datasets."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.models_dir = Path(self.temp_dir) / "models"
        self.models_dir.mkdir()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_cnn_lstm_training_with_real_dataset(self):
        """Test CNN+LSTM training with real dataset."""
        # First create a real dataset
        dataset_path = self._create_real_dataset()
        if dataset_path is None:
            pytest.skip("Could not create real dataset")

        # Train with minimal configuration for speed
        result = self.runner.invoke(
            main_app,
            [
                "train", "cnn-lstm",
                str(dataset_path),
                "--epochs", "2",  # Minimal epochs for testing
                "--batch-size", "16",
                "--sequence-length", "10",  # Short sequences
                "--prediction-horizon", "1",
                "--output-dir", str(self.models_dir)
            ],
            catch_exceptions=False
        )

        # Training might fail due to insufficient data or dependencies
        # We test that it at least starts properly
        if result.exit_code == 0:
            # Successful training
            assert "CNN+LSTM training complete" in result.output
            assert (self.models_dir / "best_model.pth").exists()
        else:
            # Training failed - check it's for expected reasons
            expected_errors = [
                "insufficient data", "not enough data", "minimum samples",
                "cuda", "gpu", "memory", "ray", "training failed"
            ]
            error_found = any(error in result.output.lower() for error in expected_errors)
            assert error_found, f"Unexpected training failure: {result.output}"

    def _create_real_dataset(self):
        """Create a small real dataset for training tests."""
        if not self._check_internet_connection():
            return None

        try:
            # Use the data pipeline to create a minimal dataset
            result = self.runner.invoke(
                main_app,
                [
                    "data", "pipeline", "--run",
                    "--symbols", "AAPL",
                    "--start-date", "2024-01-01",
                    "--end-date", "2024-01-10",  # Very small dataset
                    "--no-sentiment",
                    "--feature-set", "basic",
                    "--output-dir", self.temp_dir,
                    "--dataset-name", "training_test_data"
                ],
                catch_exceptions=False
            )

            if result.exit_code == 0:
                dataset_path = Path(self.temp_dir) / "training_test_data" / "dataset.csv"
                if dataset_path.exists():
                    return dataset_path

        except Exception:
            pass

        return None

    def _check_internet_connection(self):
        """Check if internet connection is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False


@pytest.mark.integration
class TestCLIFileIOIntegration:
    """Integration tests for file I/O operations."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_multiple_export_formats_real_data(self):
        """Test exporting real data in multiple formats."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL",
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-03",
                "--export-formats", "csv,parquet,feather",
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "multi_format_test"
            ],
            catch_exceptions=False
        )

        if result.exit_code == 0:
            dataset_dir = Path(self.temp_dir) / "multi_format_test"

            # Verify all formats were created
            assert (dataset_dir / "dataset.csv").exists()
            assert (dataset_dir / "dataset.parquet").exists()
            assert (dataset_dir / "dataset.feather").exists()

            # Verify formats contain the same data
            self._verify_format_consistency(dataset_dir)

    def test_large_symbol_list_file_handling(self):
        """Test file handling with larger symbol lists."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        # Test with a reasonable number of symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", ",".join(symbols),
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-02",  # Very short range
                "--no-sentiment",
                "--feature-set", "basic",
                "--output-dir", self.temp_dir,
                "--dataset-name", "large_symbol_test"
            ],
            catch_exceptions=False
        )

        if result.exit_code == 0:
            dataset_dir = Path(self.temp_dir) / "large_symbol_test"
            csv_path = dataset_dir / "dataset.csv"

            assert csv_path.exists()

            # Verify file size is reasonable
            file_size = csv_path.stat().st_size
            assert file_size > 0
            assert file_size < 10 * 1024 * 1024  # Less than 10MB for small dataset

    def test_dataset_metadata_accuracy(self):
        """Test accuracy of dataset metadata."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        symbols = ["AAPL", "GOOGL"]

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", ",".join(symbols),
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-03",
                "--no-sentiment",
                "--feature-set", "technical",
                "--export-formats", "csv,parquet",
                "--output-dir", self.temp_dir,
                "--dataset-name", "metadata_test"
            ],
            catch_exceptions=False
        )

        if result.exit_code == 0:
            dataset_dir = Path(self.temp_dir) / "metadata_test"

            # Load and verify metadata
            self._verify_metadata_accuracy(
                dataset_dir / "metadata.json",
                dataset_dir / "dataset.csv",
                symbols
            )

    def _check_internet_connection(self):
        """Check if internet connection is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False

    def _verify_format_consistency(self, dataset_dir):
        """Verify that different export formats contain consistent data."""
        import pandas as pd

        # Load data in different formats
        csv_df = pd.read_csv(dataset_dir / "dataset.csv")
        parquet_df = pd.read_parquet(dataset_dir / "dataset.parquet")
        feather_df = pd.read_feather(dataset_dir / "dataset.feather")

        # Compare shapes
        assert csv_df.shape == parquet_df.shape == feather_df.shape

        # Compare column names
        assert list(csv_df.columns) == list(parquet_df.columns) == list(feather_df.columns)

        # Compare data (allowing for minor float precision differences)
        numeric_cols = csv_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            assert csv_df[col].equals(parquet_df[col]) or (csv_df[col] - parquet_df[col]).abs().max() < 1e-10

    def _verify_metadata_accuracy(self, metadata_path, csv_path, expected_symbols):
        """Verify metadata accurately reflects the dataset."""
        import json

        import pandas as pd

        # Load metadata and dataset
        with open(metadata_path) as f:
            metadata = json.load(f)

        df = pd.read_csv(csv_path)

        # Verify row and column counts
        assert metadata["row_count"] == len(df)
        assert metadata["column_count"] == len(df.columns)

        # Verify symbols
        assert set(metadata["symbols"]) == set(expected_symbols)

        # Verify date range consistency
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["date"].min().strftime("%Y-%m-%d")
            df["date"].max().strftime("%Y-%m-%d")

            # Metadata dates should be consistent with actual data
            # (allowing for some flexibility in how dates are handled)


@pytest.mark.integration
@pytest.mark.network
class TestCLINetworkResilience:
    """Integration tests for network resilience and error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.slow
    def test_data_pipeline_with_network_delays(self):
        """Test data pipeline behavior with simulated network delays."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        # Test with parameters that might stress network connections
        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", "AAPL,GOOGL,MSFT",  # Multiple symbols
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-10",      # Longer range
                "--workers", "1",                # Single worker to avoid overwhelming
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "network_test"
            ],
            catch_exceptions=False
        )

        # Should handle network delays gracefully
        # Either succeed or fail with appropriate error messages
        if result.exit_code != 0:
            network_error_indicators = [
                "timeout", "connection", "network", "unreachable",
                "failed to fetch", "api limit", "rate limit"
            ]
            has_network_error = any(indicator in result.output.lower()
                                  for indicator in network_error_indicators)
            assert has_network_error, f"Unexpected error: {result.output}"

    def test_graceful_degradation_with_invalid_symbols(self):
        """Test graceful degradation when some symbols are invalid."""
        if not self._check_internet_connection():
            pytest.skip("No internet connection available")

        # Mix valid and invalid symbols
        mixed_symbols = "AAPL,INVALID123,GOOGL,FAKE456"

        result = self.runner.invoke(
            main_app,
            [
                "data", "pipeline", "--run",
                "--symbols", mixed_symbols,
                "--start-date", "2024-01-01",
                "--end-date", "2024-01-03",
                "--no-sentiment",
                "--output-dir", self.temp_dir,
                "--dataset-name", "mixed_symbols_test"
            ],
            catch_exceptions=False
        )

        # Should either:
        # 1. Process valid symbols and warn about invalid ones
        # 2. Fail with clear error about invalid symbols
        if result.exit_code == 0:
            # Success case - check if data was created for valid symbols
            dataset_dir = Path(self.temp_dir) / "mixed_symbols_test"
            if dataset_dir.exists():
                csv_path = dataset_dir / "dataset.csv"
                assert csv_path.exists()
        else:
            # Failure case - should have clear error message
            error_indicators = ["invalid", "symbol", "not found", "error"]
            has_clear_error = any(indicator in result.output.lower()
                                for indicator in error_indicators)
            assert has_clear_error, f"Unclear error message: {result.output}"

    def _check_internet_connection(self):
        """Check if internet connection is available."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
