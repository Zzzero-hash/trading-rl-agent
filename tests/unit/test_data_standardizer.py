"""
Comprehensive tests for DataStandardizer.

This module tests:
- Feature configuration and validation
- Data transformation pipeline
- Missing value handling strategies
- Scaling and normalization
- Chunked processing for large datasets
- Serialization and deserialization
- Performance benchmarks
- Edge cases and error handling
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.data.data_standardizer import (
    DataStandardizer,
    FeatureConfig,
    LiveDataProcessor,
    create_live_inference_processor,
    create_standardized_dataset,
    load_standardized_dataset,
    process_live_data,
)


class TestFeatureConfig:
    """Test FeatureConfig functionality."""

    def test_feature_config_defaults(self):
        """Test default feature configuration."""
        config = FeatureConfig()

        # Test core price features
        assert "open" in config.price_features
        assert "high" in config.price_features
        assert "low" in config.price_features
        assert "close" in config.price_features
        assert "volume" in config.price_features

        # Test technical indicators
        assert "log_return" in config.technical_indicators
        assert "sma_20" in config.technical_indicators
        assert "rsi_14" in config.technical_indicators
        assert "macd_line" in config.technical_indicators

        # Test candlestick patterns
        assert "doji" in config.candlestick_patterns
        assert "hammer" in config.candlestick_patterns
        assert "bullish_engulfing" in config.candlestick_patterns

        # Test sentiment features
        assert "sentiment" in config.sentiment_features
        assert "sentiment_magnitude" in config.sentiment_features

        # Test time features
        assert "hour" in config.time_features
        assert "day_of_week" in config.time_features
        assert "month" in config.time_features

    def test_get_all_features(self):
        """Test getting all feature names."""
        config = FeatureConfig()
        all_features = config.get_all_features()

        # Should include all feature types
        assert len(all_features) > 0
        assert all(isinstance(feature, str) for feature in all_features)

        # Should include price features
        for feature in config.price_features:
            assert feature in all_features

        # Should include technical indicators
        for feature in config.technical_indicators:
            assert feature in all_features

    def test_get_feature_count(self):
        """Test feature count calculation."""
        config = FeatureConfig()
        count = config.get_feature_count()

        assert count == len(config.get_all_features())
        assert count > 0

    def test_custom_feature_config(self):
        """Test custom feature configuration."""
        custom_config = FeatureConfig(
            price_features=["open", "close"],
            technical_indicators=["sma_5"],
            candlestick_patterns=["doji"],
            sentiment_features=[],
            time_features=["hour"],
        )

        all_features = custom_config.get_all_features()
        assert "open" in all_features
        assert "close" in all_features
        assert "sma_5" in all_features
        assert "doji" in all_features
        assert "hour" in all_features
        assert "sentiment" not in all_features


class TestDataStandardizer:
    """Test DataStandardizer functionality."""

    def test_standardizer_initialization(self):
        """Test DataStandardizer initialization."""
        standardizer = DataStandardizer()

        assert isinstance(standardizer.feature_config, FeatureConfig)
        assert standardizer.scaler is None
        assert isinstance(standardizer.feature_stats, dict)
        assert isinstance(standardizer.missing_value_strategies, dict)
        assert standardizer.logger is not None

    def test_default_missing_strategies(self):
        """Test default missing value strategies."""
        standardizer = DataStandardizer()

        # Price features should use forward_backward
        for feature in standardizer.feature_config.price_features:
            assert standardizer.missing_value_strategies[feature] == "forward_backward"

        # Technical indicators should use forward
        for feature in standardizer.feature_config.technical_indicators:
            assert standardizer.missing_value_strategies[feature] == "forward"

        # Candlestick patterns should use zero
        for feature in standardizer.feature_config.candlestick_patterns:
            assert standardizer.missing_value_strategies[feature] == "zero"

        # Sentiment features should use zero
        for feature in standardizer.feature_config.sentiment_features:
            assert standardizer.missing_value_strategies[feature] == "zero"

    def test_fit_basic(self):
        """Test basic fitting of standardizer."""
        # Create test data
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
                "sma_20": [100.5, 101.5, 102.5],
                "rsi_14": [50.0, 55.0, 60.0],
                "doji": [0, 1, 0],
                "sentiment": [0.1, 0.2, 0.3],
                "hour": [9, 10, 11],
            }
        )

        standardizer = DataStandardizer()
        fitted_standardizer = standardizer.fit(df)

        assert fitted_standardizer is standardizer
        assert len(standardizer.feature_stats) > 0
        assert standardizer.scaler is not None

    def test_fit_feature_stats_calculation(self):
        """Test feature statistics calculation."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Check that stats are calculated for price features
        for feature in ["open", "high", "low", "close", "volume"]:
            assert feature in standardizer.feature_stats
            stats = standardizer.feature_stats[feature]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "median" in stats
            assert "q25" in stats
            assert "q75" in stats

    def test_fit_missing_features(self):
        """Test fitting with missing features."""
        # Create data with only some features
        df = pd.DataFrame(
            {"open": [100.0, 101.0, 102.0], "close": [101.0, 102.0, 103.0], "volume": [1000000, 1100000, 1200000]}
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Should have stats for all features, with defaults for missing ones
        all_features = standardizer.feature_config.get_all_features()
        for feature in all_features:
            assert feature in standardizer.feature_stats

    def test_transform_basic(self):
        """Test basic data transformation."""
        # Create test data
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        result = standardizer.transform(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Should have all required features
        expected_features = standardizer.feature_config.get_all_features()
        for feature in expected_features:
            assert feature in result.columns

    def test_transform_missing_values(self):
        """Test transformation with missing values."""
        df = pd.DataFrame(
            {
                "open": [100.0, np.nan, 102.0],
                "high": [102.0, 103.0, np.nan],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
                "sma_20": [100.5, np.nan, 102.5],
                "doji": [0, 1, np.nan],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        result = standardizer.transform(df)

        # Should not have any NaN values
        assert not result.isnull().any().any()

    def test_transform_chunked_processing(self):
        """Test chunked processing for large datasets."""
        # Create large dataset
        n_rows = 10000
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, n_rows),
                "high": np.random.uniform(100, 200, n_rows),
                "low": np.random.uniform(100, 200, n_rows),
                "close": np.random.uniform(100, 200, n_rows),
                "volume": np.random.randint(1000000, 10000000, n_rows),
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Test with chunked processing
        result = standardizer.transform(df, chunk_size=1000)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == n_rows
        assert not result.isnull().any().any()

    def test_transform_invalid_values(self):
        """Test handling of invalid values."""
        df = pd.DataFrame(
            {
                "open": [100.0, -50.0, np.inf, 102.0],
                "high": [102.0, 103.0, 104.0, -np.inf],
                "low": [99.0, 100.0, 101.0, 101.0],
                "close": [101.0, 102.0, 103.0, 103.0],
                "volume": [1000000, -100000, 1200000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        result = standardizer.transform(df)

        # Should handle invalid values gracefully
        assert not result.isnull().any().any()
        assert not np.isinf(result.select_dtypes(include=[np.number])).any().any()

        # Price values should be non-negative
        price_features = ["open", "high", "low", "close"]
        for feature in price_features:
            if feature in result.columns:
                assert all(result[feature] >= 0)

    def test_transform_feature_order(self):
        """Test that features are in correct order."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        result = standardizer.transform(df)

        # Check that features are in the expected order
        expected_features = standardizer.feature_config.get_all_features()
        actual_features = [col for col in result.columns if col in expected_features]

        assert actual_features == expected_features

    def test_save_and_load(self, tmp_path):
        """Test saving and loading standardizer."""
        # Create and fit standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Save standardizer
        save_path = tmp_path / "standardizer.pkl"
        standardizer.save(str(save_path))

        # Check that files were created
        assert save_path.exists()
        assert (tmp_path / "standardizer.json").exists()

        # Load standardizer
        loaded_standardizer = DataStandardizer.load(str(save_path))

        assert isinstance(loaded_standardizer, DataStandardizer)
        assert loaded_standardizer.feature_stats == standardizer.feature_stats
        assert loaded_standardizer.missing_value_strategies == standardizer.missing_value_strategies

    def test_create_live_data_template(self):
        """Test creation of live data template."""
        standardizer = DataStandardizer()
        template = standardizer.create_live_data_template()

        assert isinstance(template, pd.DataFrame)
        assert len(template) == 1
        assert len(template.columns) == len(standardizer.feature_config.get_all_features())

        # Check that all required features are present
        for feature in standardizer.feature_config.get_all_features():
            assert feature in template.columns

    @pytest.mark.benchmark
    def test_transform_performance(self, benchmark):
        """Benchmark transformation performance."""
        # Create large dataset
        n_rows = 10000
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, n_rows),
                "high": np.random.uniform(100, 200, n_rows),
                "low": np.random.uniform(100, 200, n_rows),
                "close": np.random.uniform(100, 200, n_rows),
                "volume": np.random.randint(1000000, 10000000, n_rows),
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        benchmark(lambda: standardizer.transform(df))


class TestLiveDataProcessor:
    """Test LiveDataProcessor functionality."""

    def test_live_data_processor_initialization(self):
        """Test LiveDataProcessor initialization."""
        standardizer = DataStandardizer()
        processor = LiveDataProcessor(standardizer)

        assert processor.standardizer is standardizer

    def test_process_single_row(self):
        """Test processing single row of live data."""
        # Create and fit standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        processor = LiveDataProcessor(standardizer)

        # Process single row
        live_data = {"open": 105.0, "high": 107.0, "low": 104.0, "close": 106.0, "volume": 1500000}

        result = processor.process_single_row(live_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert len(result.columns) == len(standardizer.feature_config.get_all_features())

    def test_process_batch(self):
        """Test processing batch of live data."""
        # Create and fit standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        processor = LiveDataProcessor(standardizer)

        # Process batch
        live_data_batch = [
            {"open": 105.0, "high": 107.0, "low": 104.0, "close": 106.0, "volume": 1500000},
            {"open": 106.0, "high": 108.0, "low": 105.0, "close": 107.0, "volume": 1600000},
        ]

        result = processor.process_batch(live_data_batch)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert len(result.columns) == len(standardizer.feature_config.get_all_features())

    def test_get_feature_names(self):
        """Test getting feature names from processor."""
        standardizer = DataStandardizer()
        processor = LiveDataProcessor(standardizer)

        feature_names = processor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) == len(standardizer.feature_config.get_all_features())

    def test_get_feature_count(self):
        """Test getting feature count from processor."""
        standardizer = DataStandardizer()
        processor = LiveDataProcessor(standardizer)

        count = processor.get_feature_count()

        assert count == len(standardizer.feature_config.get_all_features())


class TestUtilityFunctions:
    """Test utility functions."""

    def test_create_standardized_dataset(self, tmp_path):
        """Test creating standardized dataset."""
        # Create test data
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        save_path = tmp_path / "dataset.csv"
        standardizer_path = tmp_path / "standardizer.pkl"

        result_df, result_standardizer = create_standardized_dataset(df, str(save_path), None)

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(result_standardizer, DataStandardizer)
        assert save_path.exists()
        assert standardizer_path.exists()

    def test_load_standardized_dataset(self, tmp_path):
        """Test loading standardized dataset."""
        # Create test data and save it
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        data_path = tmp_path / "dataset.csv"
        standardizer_path = tmp_path / "standardizer.pkl"

        # Save dataset
        df.to_csv(data_path, index=False)
        standardizer = DataStandardizer()
        standardizer.fit(df)
        standardizer.save(str(standardizer_path))

        # Load dataset
        result_df, result_standardizer = load_standardized_dataset(str(data_path), str(standardizer_path))

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(result_standardizer, DataStandardizer)
        assert len(result_df) == len(df)

    def test_create_live_inference_processor(self, tmp_path):
        """Test creating live inference processor."""
        # Create and save standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)
        standardizer_path = tmp_path / "standardizer.pkl"
        standardizer.save(str(standardizer_path))

        # Create processor
        processor = create_live_inference_processor(str(standardizer_path))

        assert isinstance(processor, LiveDataProcessor)

    def test_process_live_data(self, tmp_path):
        """Test processing live data."""
        # Create and save standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)
        standardizer_path = tmp_path / "standardizer.pkl"
        standardizer.save(str(standardizer_path))

        # Process live data
        live_data = {"open": 105.0, "high": 107.0, "low": 104.0, "close": 106.0, "volume": 1500000}

        result = process_live_data(live_data, str(standardizer_path))

        assert isinstance(result, np.ndarray)
        assert len(result) == len(standardizer.feature_config.get_all_features())


class TestDataStandardizerEdgeCases:
    """Test edge cases and error scenarios."""

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()

        standardizer = DataStandardizer()

        # Should handle empty dataframe gracefully
        fitted_standardizer = standardizer.fit(df)
        assert fitted_standardizer is standardizer

        result = standardizer.transform(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_row_dataframe(self):
        """Test handling of single row dataframe."""
        df = pd.DataFrame({"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1000000]})

        standardizer = DataStandardizer()
        standardizer.fit(df)

        result = standardizer.transform(df)
        assert len(result) == 1
        assert not result.isnull().any().any()

    def test_all_missing_values(self):
        """Test handling of dataframe with all missing values."""
        df = pd.DataFrame(
            {
                "open": [np.nan, np.nan, np.nan],
                "high": [np.nan, np.nan, np.nan],
                "low": [np.nan, np.nan, np.nan],
                "close": [np.nan, np.nan, np.nan],
                "volume": [np.nan, np.nan, np.nan],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        result = standardizer.transform(df)
        assert not result.isnull().any().any()

    def test_invalid_missing_strategy(self):
        """Test handling of invalid missing value strategy."""
        df = pd.DataFrame(
            {
                "open": [100.0, np.nan, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.missing_value_strategies["open"] = "invalid_strategy"
        standardizer.fit(df)

        # Should handle invalid strategy gracefully
        result = standardizer.transform(df)
        assert not result.isnull().any().any()

    def test_scaler_failure(self):
        """Test handling of scaler fitting failure."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        with patch("sklearn.preprocessing.RobustScaler") as mock_scaler:
            mock_scaler.side_effect = Exception("Scaler error")

            standardizer = DataStandardizer()
            standardizer.fit(df)

            # Should handle scaler failure gracefully
            result = standardizer.transform(df)
            assert isinstance(result, pd.DataFrame)

    def test_memory_usage_optimization(self):
        """Test memory usage optimization for large datasets."""
        # Create large dataset
        n_rows = 50000
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, n_rows),
                "high": np.random.uniform(100, 200, n_rows),
                "low": np.random.uniform(100, 200, n_rows),
                "close": np.random.uniform(100, 200, n_rows),
                "volume": np.random.randint(1000000, 10000000, n_rows),
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Test with different chunk sizes
        chunk_sizes = [1000, 5000, 10000]

        for chunk_size in chunk_sizes:
            result = standardizer.transform(df, chunk_size=chunk_size)
            assert len(result) == n_rows
            assert not result.isnull().any().any()

    def test_concurrent_access(self):
        """Test concurrent access to standardizer."""
        import threading

        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        results = []

        def transform_data():
            result = standardizer.transform(df)
            results.append(result)

        # Create multiple threads
        threads = [threading.Thread(target=transform_data) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all transformations completed successfully
        assert len(results) == 5
        for result in results:
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3


class TestDataStandardizerIntegration:
    """Integration tests for DataStandardizer."""

    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline."""
        # Create training data
        train_df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )

        # Create test data
        test_df = pd.DataFrame(
            {
                "open": [105.0, 106.0],
                "high": [107.0, 108.0],
                "low": [104.0, 105.0],
                "close": [106.0, 107.0],
                "volume": [1500000, 1600000],
            }
        )

        # Fit standardizer on training data
        standardizer = DataStandardizer()
        standardizer.fit(train_df)

        # Transform test data
        transformed_test = standardizer.transform(test_df)

        # Verify results
        assert isinstance(transformed_test, pd.DataFrame)
        assert len(transformed_test) == 2
        assert not transformed_test.isnull().any().any()

        # Check that all required features are present
        expected_features = standardizer.feature_config.get_all_features()
        for feature in expected_features:
            assert feature in transformed_test.columns

    def test_live_inference_workflow(self):
        """Test live inference workflow."""
        # Create and fit standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Create live processor
        processor = LiveDataProcessor(standardizer)

        # Process live data
        live_data = {"open": 105.0, "high": 107.0, "low": 104.0, "close": 106.0, "volume": 1500000}

        result = processor.process_single_row(live_data)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert len(result.columns) == len(standardizer.feature_config.get_all_features())
        assert not result.isnull().any().any()

    def test_persistence_workflow(self, tmp_path):
        """Test complete persistence workflow."""
        # Create and fit standardizer
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Save standardizer
        save_path = tmp_path / "standardizer.pkl"
        standardizer.save(str(save_path))

        # Load standardizer
        loaded_standardizer = DataStandardizer.load(str(save_path))

        # Transform data with loaded standardizer
        result = loaded_standardizer.transform(df)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert not result.isnull().any().any()

    @pytest.mark.benchmark
    def test_large_dataset_performance(self, benchmark):
        """Benchmark performance with large dataset."""
        # Create large dataset
        n_rows = 100000
        df = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, n_rows),
                "high": np.random.uniform(100, 200, n_rows),
                "low": np.random.uniform(100, 200, n_rows),
                "close": np.random.uniform(100, 200, n_rows),
                "volume": np.random.randint(1000000, 10000000, n_rows),
            }
        )

        standardizer = DataStandardizer()
        standardizer.fit(df)

        # Benchmark transformation
        benchmark(lambda: standardizer.transform(df, chunk_size=5000))
