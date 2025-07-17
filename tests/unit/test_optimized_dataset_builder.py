"""
Comprehensive tests for optimized dataset builder module.

This module tests the OptimizedDatasetBuilder class and related functionality
with focus on small incremental fixes and edge cases.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.data.optimized_dataset_builder import OptimizedDatasetBuilder


class TestOptimizedDatasetBuilder:
    """Test suite for OptimizedDatasetBuilder class."""

    def test_initialization(self):
        """Test OptimizedDatasetBuilder initialization."""
        builder = OptimizedDatasetBuilder()

        assert builder.logger is not None
        assert builder.cache_ttl == 3600  # Default TTL
        assert builder.max_workers == 4  # Default workers
        assert builder.chunk_size == 1000  # Default chunk size

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        builder = OptimizedDatasetBuilder(cache_ttl=7200, max_workers=8, chunk_size=2000)

        assert builder.cache_ttl == 7200
        assert builder.max_workers == 8
        assert builder.chunk_size == 2000

    def test_validate_config(self):
        """Test configuration validation."""
        builder = OptimizedDatasetBuilder()

        # Valid config
        valid_config = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "features": ["close", "volume"],
        }

        assert builder.validate_config(valid_config) is True

        # Invalid config - missing symbols
        invalid_config = {"start_date": "2023-01-01", "end_date": "2023-12-31"}

        with pytest.raises(ValueError):
            builder.validate_config(invalid_config)

    def test_create_cache_key(self):
        """Test cache key creation."""
        builder = OptimizedDatasetBuilder()

        config = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "features": ["close", "volume"],
        }

        cache_key = builder.create_cache_key(config)
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        assert "AAPL" in cache_key
        assert "GOOGL" in cache_key

    def test_is_cache_valid(self):
        """Test cache validity checking."""
        builder = OptimizedDatasetBuilder()

        # Test valid cache
        cache_data = {"timestamp": pd.Timestamp.now(), "data": pd.DataFrame({"close": [100, 101, 102]})}

        assert builder.is_cache_valid(cache_data) is True

        # Test expired cache
        cache_data["timestamp"] = pd.Timestamp.now() - pd.Timedelta(hours=2)
        assert builder.is_cache_valid(cache_data) is False

    def test_fetch_data_parallel(self):
        """Test parallel data fetching."""
        builder = OptimizedDatasetBuilder(max_workers=2)

        symbols = ["AAPL", "GOOGL"]

        # Mock data fetching
        mock_data = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "GOOGL", "GOOGL"],
                "close": [100, 101, 200, 201],
                "volume": [1000, 1100, 2000, 2100],
            }
        )

        with patch.object(builder, "_fetch_symbol_data", return_value=mock_data):
            result = builder.fetch_data_parallel(symbols, "2023-01-01", "2023-01-02")

            assert isinstance(result, dict)
            assert "AAPL" in result
            assert "GOOGL" in result

    def test_fetch_data_parallel_with_failures(self):
        """Test parallel data fetching with some failures."""
        builder = OptimizedDatasetBuilder(max_workers=2)

        symbols = ["AAPL", "INVALID_SYMBOL"]

        def mock_fetch(symbol, start_date, end_date):
            if symbol == "AAPL":
                return pd.DataFrame({"symbol": ["AAPL"], "close": [100], "volume": [1000]})
            raise Exception(f"Failed to fetch {symbol}")

        with patch.object(builder, "_fetch_symbol_data", side_effect=mock_fetch):
            result = builder.fetch_data_parallel(symbols, "2023-01-01", "2023-01-02")

            # Should have data for AAPL but not for INVALID_SYMBOL
            assert "AAPL" in result
            assert "INVALID_SYMBOL" not in result

    def test_process_features(self):
        """Test feature processing functionality."""
        builder = OptimizedDatasetBuilder()

        # Create sample data
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [1000, 1100, 1200, 1300, 1400]})

        processed_data = builder.process_features(data)

        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert len(processed_data.columns) >= len(data.columns)

    def test_normalize_data(self):
        """Test data normalization."""
        builder = OptimizedDatasetBuilder()

        # Create sample data
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [1000, 1100, 1200, 1300, 1400]})

        normalized_data = builder.normalize_data(data)

        assert isinstance(normalized_data, pd.DataFrame)
        assert normalized_data.shape == data.shape

        # Check that data is normalized (mean close to 0, std close to 1)
        assert abs(normalized_data["close"].mean()) < 1e-10
        assert abs(normalized_data["close"].std() - 1.0) < 1e-10

    def test_create_sequences(self):
        """Test sequence creation for time series data."""
        builder = OptimizedDatasetBuilder()

        # Create sample data
        data = pd.DataFrame({"close": list(range(100)), "volume": list(range(100, 200))})

        sequence_length = 10
        sequences = builder.create_sequences(data, sequence_length)

        assert isinstance(sequences, np.ndarray)
        assert len(sequences.shape) == 3
        assert sequences.shape[1] == sequence_length
        assert sequences.shape[2] == len(data.columns)

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        builder = OptimizedDatasetBuilder()

        # Create small dataset
        data = pd.DataFrame({"close": [100, 101, 102], "volume": [1000, 1100, 1200]})

        sequence_length = 10

        with pytest.raises(ValueError):
            builder.create_sequences(data, sequence_length)

    def test_split_data(self):
        """Test data splitting functionality."""
        builder = OptimizedDatasetBuilder()

        # Create sample data
        data = pd.DataFrame({"close": list(range(100)), "volume": list(range(100, 200))})

        train_data, val_data, test_data = builder.split_data(data, train_ratio=0.7, val_ratio=0.15)

        assert isinstance(train_data, pd.DataFrame)
        assert isinstance(val_data, pd.DataFrame)
        assert isinstance(test_data, pd.DataFrame)

        # Check split ratios
        total_len = len(data)
        assert len(train_data) == int(0.7 * total_len)
        assert len(val_data) == int(0.15 * total_len)
        assert len(test_data) == total_len - len(train_data) - len(val_data)

    def test_save_dataset(self):
        """Test dataset saving functionality."""
        builder = OptimizedDatasetBuilder()

        # Create sample dataset
        dataset = {
            "train": pd.DataFrame({"close": [100, 101, 102]}),
            "val": pd.DataFrame({"close": [103, 104]}),
            "test": pd.DataFrame({"close": [105, 106]}),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dataset"

            builder.save_dataset(dataset, output_path)

            # Check that files were created
            assert (output_path / "train.csv").exists()
            assert (output_path / "val.csv").exists()
            assert (output_path / "test.csv").exists()

    def test_load_dataset(self):
        """Test dataset loading functionality."""
        builder = OptimizedDatasetBuilder()

        # Create sample dataset
        dataset = {
            "train": pd.DataFrame({"close": [100, 101, 102]}),
            "val": pd.DataFrame({"close": [103, 104]}),
            "test": pd.DataFrame({"close": [105, 106]}),
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "dataset"

            # Save dataset
            builder.save_dataset(dataset, output_path)

            # Load dataset
            loaded_dataset = builder.load_dataset(output_path)

            assert "train" in loaded_dataset
            assert "val" in loaded_dataset
            assert "test" in loaded_dataset

            # Check data integrity
            pd.testing.assert_frame_equal(dataset["train"], loaded_dataset["train"])
            pd.testing.assert_frame_equal(dataset["val"], loaded_dataset["val"])
            pd.testing.assert_frame_equal(dataset["test"], loaded_dataset["test"])

    def test_build_dataset(self):
        """Test complete dataset building workflow."""
        builder = OptimizedDatasetBuilder()

        config = {
            "symbols": ["AAPL"],
            "start_date": "2023-01-01",
            "end_date": "2023-01-10",
            "features": ["close", "volume"],
            "sequence_length": 5,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        }

        # Mock data fetching
        mock_data = pd.DataFrame({"close": list(range(100, 110)), "volume": list(range(1000, 1010))})

        with patch.object(builder, "_fetch_symbol_data", return_value=mock_data):
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "dataset"

                dataset = builder.build_dataset(config, output_path)

                assert "train" in dataset
                assert "val" in dataset
                assert "test" in dataset
                assert output_path.exists()

    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        builder = OptimizedDatasetBuilder()

        # Test memory usage tracking
        initial_memory = builder.get_memory_usage()
        assert isinstance(initial_memory, float)
        assert initial_memory >= 0

        # Test memory warning
        builder.check_memory_usage(threshold_mb=1000)

    def test_progress_monitoring(self):
        """Test progress monitoring functionality."""
        builder = OptimizedDatasetBuilder()

        # Test progress tracking
        builder.update_progress(0.5, "Processing data")
        progress = builder.get_progress()

        assert progress["percentage"] == 0.5
        assert progress["message"] == "Processing data"

    def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms."""
        builder = OptimizedDatasetBuilder()

        # Test retry mechanism
        def failing_function():
            raise Exception("Temporary failure")

        with pytest.raises(RuntimeError):
            builder.retry_operation(failing_function, max_retries=3)

    def test_data_quality_validation(self):
        """Test data quality validation."""
        builder = OptimizedDatasetBuilder()

        # Valid data
        valid_data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [1000, 1100, 1200, 1300, 1400]})

        assert builder.validate_data_quality(valid_data) is True

        # Invalid data - too many NaN values
        invalid_data = pd.DataFrame(
            {"close": [100, np.nan, np.nan, np.nan, 104], "volume": [1000, 1100, 1200, 1300, 1400]}
        )

        with pytest.raises(ValueError):
            builder.validate_data_quality(invalid_data)

    def test_cache_management(self):
        """Test cache management functionality."""
        builder = OptimizedDatasetBuilder()

        # Test cache storage
        cache_key = "test_key"
        cache_data = pd.DataFrame({"close": [100, 101, 102]})

        builder.store_cache(cache_key, cache_data)

        # Test cache retrieval
        retrieved_data = builder.get_cache(cache_key)
        pd.testing.assert_frame_equal(cache_data, retrieved_data)

        # Test cache clearing
        builder.clear_cache()
        assert builder.get_cache(cache_key) is None

    def test_parallel_processing_configuration(self):
        """Test parallel processing configuration."""
        builder = OptimizedDatasetBuilder(max_workers=4)

        # Test worker configuration
        assert builder.max_workers == 4

        # Test chunk size configuration
        builder.set_chunk_size(500)
        assert builder.chunk_size == 500

    def test_data_transformation_pipeline(self):
        """Test complete data transformation pipeline."""
        builder = OptimizedDatasetBuilder()

        # Create sample raw data
        raw_data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [1000, 1100, 1200, 1300, 1400]})

        # Test transformation pipeline
        transformed_data = builder.transform_data(raw_data)

        assert isinstance(transformed_data, pd.DataFrame)
        assert len(transformed_data) > 0
        assert len(transformed_data.columns) >= len(raw_data.columns)

    def test_feature_engineering_integration(self):
        """Test integration with feature engineering."""
        builder = OptimizedDatasetBuilder()

        # Create sample data
        data = pd.DataFrame({"close": [100, 101, 102, 103, 104], "volume": [1000, 1100, 1200, 1300, 1400]})

        # Test feature engineering
        features = builder.engineer_features(data)

        assert isinstance(features, pd.DataFrame)
        assert len(features.columns) > len(data.columns)  # Should have more features

    def test_performance_optimization(self):
        """Test performance optimization features."""
        builder = OptimizedDatasetBuilder()

        # Test memory optimization
        builder.optimize_memory_usage()

        # Test processing optimization
        builder.optimize_processing()

        # Test caching optimization
        builder.optimize_caching()

    def test_logging_and_monitoring(self):
        """Test logging and monitoring functionality."""
        builder = OptimizedDatasetBuilder()

        # Test logging
        builder.log_info("Test info message")
        builder.log_warning("Test warning message")
        builder.log_error("Test error message")

        # Test metrics collection
        metrics = builder.collect_metrics()
        assert isinstance(metrics, dict)
        assert "processing_time" in metrics
        assert "memory_usage" in metrics

    def test_configuration_management(self):
        """Test configuration management."""
        builder = OptimizedDatasetBuilder()

        # Test config loading
        config = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "features": ["close", "volume"],
            "sequence_length": 10,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        }

        builder.load_config(config)
        assert builder.config == config

        # Test config validation
        assert builder.validate_config(config) is True

    def test_data_source_integration(self):
        """Test integration with different data sources."""
        builder = OptimizedDatasetBuilder()

        # Test yfinance integration
        with patch("yfinance.download") as mock_download:
            mock_download.return_value = pd.DataFrame({"Close": [100, 101, 102], "Volume": [1000, 1100, 1200]})

            data = builder.fetch_from_yfinance("AAPL", "2023-01-01", "2023-01-03")
            assert isinstance(data, pd.DataFrame)
            assert not data.empty

    def test_error_recovery_strategies(self):
        """Test error recovery strategies."""
        builder = OptimizedDatasetBuilder()

        # Test graceful degradation
        result = builder.handle_error(Exception("Test error"), "test_operation")
        assert result is not None

        # Test fallback mechanisms
        fallback_result = builder.use_fallback_strategy("test_operation")
        assert fallback_result is not None


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    def test_invalid_config_parameters(self):
        """Test handling of invalid configuration parameters."""
        builder = OptimizedDatasetBuilder()

        # Test invalid date format
        invalid_config = {"symbols": ["AAPL"], "start_date": "invalid_date", "end_date": "2023-12-31"}

        with pytest.raises(ValueError):
            builder.validate_config(invalid_config)

    def test_network_failures(self):
        """Test handling of network failures."""
        builder = OptimizedDatasetBuilder()

        with patch.object(builder, "_fetch_symbol_data", side_effect=Exception("Network error")):
            with pytest.raises(RuntimeError):
                builder.fetch_data_parallel(["AAPL"], "2023-01-01", "2023-01-02")

    def test_memory_overflow(self):
        """Test handling of memory overflow."""
        builder = OptimizedDatasetBuilder()

        # Create large dataset that might cause memory issues
        large_data = pd.DataFrame({"close": np.random.randn(1000000), "volume": np.random.randn(1000000)})

        # Should handle gracefully
        try:
            builder.process_features(large_data)
        except MemoryError:
            # This is expected for very large datasets
            pass

    def test_data_corruption(self):
        """Test handling of corrupted data."""
        builder = OptimizedDatasetBuilder()

        # Create corrupted data
        corrupted_data = pd.DataFrame(
            {"close": [100, np.inf, 102, -np.inf, 104], "volume": [1000, 1100, 1200, 1300, 1400]}
        )

        with pytest.raises(ValueError):
            builder.validate_data_quality(corrupted_data)

    def test_file_system_errors(self):
        """Test handling of file system errors."""
        builder = OptimizedDatasetBuilder()

        # Test saving to invalid path
        invalid_path = "/invalid/path/that/does/not/exist"

        with pytest.raises(OSError):
            builder.save_dataset({}, invalid_path)


class TestIntegration:
    """Integration tests for optimized dataset builder."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        builder = OptimizedDatasetBuilder()

        config = {
            "symbols": ["AAPL", "GOOGL"],
            "start_date": "2023-01-01",
            "end_date": "2023-01-10",
            "features": ["close", "volume"],
            "sequence_length": 5,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        }

        # Mock data fetching for both symbols
        mock_data = pd.DataFrame({"close": list(range(100, 110)), "volume": list(range(1000, 1010))})

        with patch.object(builder, "_fetch_symbol_data", return_value=mock_data):
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "dataset"

                # Build dataset
                dataset = builder.build_dataset(config, output_path)

                # Verify results
                assert "train" in dataset
                assert "val" in dataset
                assert "test" in dataset
                assert output_path.exists()

                # Test loading the built dataset
                loaded_dataset = builder.load_dataset(output_path)
                assert "train" in loaded_dataset
                assert "val" in loaded_dataset
                assert "test" in loaded_dataset

    def test_large_scale_processing(self):
        """Test large-scale data processing."""
        builder = OptimizedDatasetBuilder(max_workers=4, chunk_size=1000)

        # Create large synthetic dataset
        large_data = pd.DataFrame({"close": np.random.randn(10000), "volume": np.random.randn(10000)})

        # Test processing large dataset
        processed_data = builder.process_features(large_data)
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) == len(large_data)

    def test_multi_symbol_processing(self):
        """Test processing multiple symbols."""
        builder = OptimizedDatasetBuilder(max_workers=4)

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

        # Mock data for all symbols
        mock_data = pd.DataFrame({"close": list(range(100, 110)), "volume": list(range(1000, 1010))})

        with patch.object(builder, "_fetch_symbol_data", return_value=mock_data):
            result = builder.fetch_data_parallel(symbols, "2023-01-01", "2023-01-02")

            # Check all symbols were processed
            for symbol in symbols:
                assert symbol in result
                assert isinstance(result[symbol], pd.DataFrame)

    def test_cache_performance(self):
        """Test cache performance and efficiency."""
        builder = OptimizedDatasetBuilder(cache_ttl=3600)

        # Test cache hit performance
        cache_key = "test_key"
        cache_data = pd.DataFrame({"close": [100, 101, 102]})

        # Store in cache
        builder.store_cache(cache_key, cache_data)

        # Retrieve from cache (should be fast)
        start_time = pd.Timestamp.now()
        retrieved_data = builder.get_cache(cache_key)
        end_time = pd.Timestamp.now()

        # Should be very fast (cache hit)
        assert (end_time - start_time).total_seconds() < 0.1
        pd.testing.assert_frame_equal(cache_data, retrieved_data)

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        builder = OptimizedDatasetBuilder()

        # Create large dataset
        large_data = pd.DataFrame({"close": np.random.randn(100000), "volume": np.random.randn(100000)})

        # Monitor memory usage
        initial_memory = builder.get_memory_usage()

        # Process data
        processed_data = builder.process_features(large_data)

        final_memory = builder.get_memory_usage()

        # Memory increase should be reasonable
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000  # Less than 1GB increase
