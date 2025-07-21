"""
Comprehensive tests for feature engineering pipeline.

Tests cover:
- Feature computation determinism
- Shape consistency across different timeframes
- Robustness to missing data
- Normalization consistency
- Alternative data integration
- Temporal feature handling
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.features.alternative_data import (
    AlternativeDataConfig,
    AlternativeDataFeatures,
)
from trading_rl_agent.features.normalization import (
    FeatureNormalizer,
    NormalizationConfig,
)
from trading_rl_agent.features.pipeline import FeaturePipeline
from trading_rl_agent.features.technical_indicators import (
    IndicatorConfig,
    TechnicalIndicators,
)


class TestFeatureEngineeringDeterminism:
    """Test that feature engineering produces deterministic results."""

    @pytest.fixture
    def sample_data(self):
        """Create sample trading data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")

        data = []
        for i, date in enumerate(dates):
            base_price = 100 + i * 0.1
            data.append(
                {
                    "timestamp": date,
                    "symbol": "AAPL",
                    "open": base_price + np.random.normal(0, 0.5),
                    "high": base_price + np.random.normal(0.5, 0.3),
                    "low": base_price + np.random.normal(-0.5, 0.3),
                    "close": base_price + np.random.normal(0, 0.5),
                    "volume": np.random.randint(1000, 10000),
                }
            )

        return pd.DataFrame(data)

    def test_technical_indicators_determinism(self, sample_data):
        """Test that technical indicators produce deterministic results."""
        config = IndicatorConfig()
        indicators = TechnicalIndicators(config)

        # Calculate indicators twice
        result1 = indicators.calculate_all_indicators(sample_data.copy())
        result2 = indicators.calculate_all_indicators(sample_data.copy())

        # Check that results are identical
        pd.testing.assert_frame_equal(result1, result2)

        # Check that specific indicators are present and deterministic
        expected_features = ["sma_5", "sma_10", "rsi", "macd", "bb_upper", "atr"]
        for feature in expected_features:
            assert feature in result1.columns
            np.testing.assert_array_equal(result1[feature].values, result2[feature].values)

    def test_alternative_data_determinism(self, sample_data):
        """Test that alternative data features are deterministic."""
        config = AlternativeDataConfig()
        alt_features = AlternativeDataFeatures(config)

        # Add alternative features twice
        result1 = alt_features.add_alternative_features(sample_data.copy(), symbol="AAPL")
        result2 = alt_features.add_alternative_features(sample_data.copy(), symbol="AAPL")

        # Check that results are identical
        pd.testing.assert_frame_equal(result1, result2)

        # Check that sentiment features are present
        sentiment_features = [
            "news_sentiment",
            "news_sentiment_magnitude",
            "news_sentiment_direction",
        ]
        for feature in sentiment_features:
            assert feature in result1.columns

    def test_normalization_determinism(self, sample_data):
        """Test that normalization produces deterministic results."""
        config = NormalizationConfig(method="robust", per_symbol=True)
        normalizer = FeatureNormalizer(config)

        # Fit and transform twice
        result1 = normalizer.fit_transform(sample_data, symbol_column="symbol")
        result2 = normalizer.fit_transform(sample_data, symbol_column="symbol")

        # Check that results are identical
        pd.testing.assert_frame_equal(result1, result2)

        # Check that values are properly scaled
        feature_cols = [col for col in result1.columns if col not in ["timestamp", "symbol"]]
        for col in feature_cols:
            # Check that values are finite
            assert np.all(np.isfinite(result1[col]))
            # Check that values are not all the same (unless original was constant)
            if sample_data[col].std() > 0:
                assert result1[col].std() > 0


class TestFeatureShapeConsistency:
    """Test that features maintain consistent shapes across different timeframes."""

    @pytest.fixture
    def multi_timeframe_data(self):
        """Create data for different timeframes."""
        base_date = datetime(2023, 1, 1)

        # Daily data
        daily_dates = pd.date_range(start=base_date, periods=50, freq="D")
        daily_data = []
        for i, date in enumerate(daily_dates):
            daily_data.append(
                {
                    "timestamp": date,
                    "symbol": "AAPL",
                    "open": 100 + i * 0.1,
                    "high": 101 + i * 0.1,
                    "low": 99 + i * 0.1,
                    "close": 100.5 + i * 0.1,
                    "volume": 1000 + i * 10,
                }
            )

        # Hourly data
        hourly_dates = pd.date_range(start=base_date, periods=200, freq="H")
        hourly_data = []
        for i, date in enumerate(hourly_dates):
            hourly_data.append(
                {
                    "timestamp": date,
                    "symbol": "AAPL",
                    "open": 100 + i * 0.01,
                    "high": 100.5 + i * 0.01,
                    "low": 99.5 + i * 0.01,
                    "close": 100.1 + i * 0.01,
                    "volume": 100 + i,
                }
            )

        return {
            "daily": pd.DataFrame(daily_data),
            "hourly": pd.DataFrame(hourly_data),
        }

    def test_feature_shapes_consistent(self, multi_timeframe_data):
        """Test that feature engineering produces consistent shapes."""
        pipeline = FeaturePipeline()

        daily_result = pipeline.fit_transform(multi_timeframe_data["daily"])
        hourly_result = pipeline.fit_transform(multi_timeframe_data["hourly"])

        # Check that both have the same feature columns (excluding timestamp/symbol)
        daily_features = [col for col in daily_result.columns if col not in ["timestamp", "symbol"]]
        hourly_features = [col for col in hourly_result.columns if col not in ["timestamp", "symbol"]]

        assert set(daily_features) == set(hourly_features)
        assert len(daily_features) > 0

        # Check that temporal features are properly encoded
        temporal_features = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]
        for feature in temporal_features:
            if feature in daily_features:
                assert feature in hourly_features

    def test_sequence_creation_consistency(self, multi_timeframe_data):
        """Test that sequence creation works consistently across timeframes."""
        from trading_rl_agent.data.preprocessing import create_sequences

        pipeline = FeaturePipeline()

        # Process both timeframes
        daily_result = pipeline.fit_transform(multi_timeframe_data["daily"])
        hourly_result = pipeline.fit_transform(multi_timeframe_data["hourly"])

        # Create sequences
        sequence_length = 20

        daily_sequences, daily_targets = create_sequences(daily_result, sequence_length, target_column="close")
        hourly_sequences, hourly_targets = create_sequences(hourly_result, sequence_length, target_column="close")

        # Check that sequences have the expected shape
        assert daily_sequences.shape[1] == sequence_length
        assert hourly_sequences.shape[1] == sequence_length

        # Check that feature dimensions are consistent
        assert daily_sequences.shape[2] == hourly_sequences.shape[2]

        # Check that we have sequences
        assert len(daily_sequences) > 0
        assert len(hourly_sequences) > 0


class TestMissingDataRobustness:
    """Test that feature engineering is robust to missing data."""

    @pytest.fixture
    def data_with_missing(self):
        """Create data with various missing value patterns."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")

        data = []
        for i, date in enumerate(dates):
            # Introduce missing values
            if i % 10 == 0:  # Every 10th row has missing values
                data.append(
                    {
                        "timestamp": date,
                        "symbol": "AAPL",
                        "open": np.nan,
                        "high": 101 + i * 0.1,
                        "low": np.nan,
                        "close": 100.5 + i * 0.1,
                        "volume": 1000 + i * 10,
                    }
                )
            else:
                data.append(
                    {
                        "timestamp": date,
                        "symbol": "AAPL",
                        "open": 100 + i * 0.1,
                        "high": 101 + i * 0.1,
                        "low": 99 + i * 0.1,
                        "close": 100.5 + i * 0.1,
                        "volume": 1000 + i * 10,
                    }
                )

        return pd.DataFrame(data)

    def test_missing_data_handling(self, data_with_missing):
        """Test that missing data is handled gracefully."""
        pipeline = FeaturePipeline()

        # Should not raise an exception
        result = pipeline.fit_transform(data_with_missing)

        # Check that no NaN values remain in feature columns
        feature_cols = [col for col in result.columns if col not in ["timestamp", "symbol"]]
        for col in feature_cols:
            assert not result[col].isnull().any(), f"Column {col} contains NaN values"

    def test_normalization_with_missing(self, data_with_missing):
        """Test that normalization handles missing data properly."""
        config = NormalizationConfig(method="robust", handle_missing=True, missing_strategy="median")
        normalizer = FeatureNormalizer(config)

        # Should not raise an exception
        result = normalizer.fit_transform(data_with_missing, symbol_column="symbol")

        # Check that all values are finite
        feature_cols = [col for col in result.columns if col not in ["timestamp", "symbol"]]
        for col in feature_cols:
            assert np.all(np.isfinite(result[col])), f"Column {col} contains non-finite values"


class TestVaryingTimeframesRobustness:
    """Test that features are robust to varying timeframes."""

    @pytest.fixture
    def varying_timeframe_data(self):
        """Create data with varying timeframes."""
        base_date = datetime(2023, 1, 1)

        # Create data with irregular intervals
        timestamps = []
        current_time = base_date

        for i in range(100):
            # Vary the interval between 1 minute and 1 hour
            interval = np.random.randint(1, 60)
            current_time += timedelta(minutes=interval)
            timestamps.append(current_time)

        data = []
        for i, timestamp in enumerate(timestamps):
            data.append(
                {
                    "timestamp": timestamp,
                    "symbol": "AAPL",
                    "open": 100 + i * 0.1,
                    "high": 101 + i * 0.1,
                    "low": 99 + i * 0.1,
                    "close": 100.5 + i * 0.1,
                    "volume": 1000 + i * 10,
                }
            )

        return pd.DataFrame(data)

    def test_irregular_timeframes(self, varying_timeframe_data):
        """Test that feature engineering works with irregular timeframes."""
        pipeline = FeaturePipeline()

        # Should not raise an exception
        result = pipeline.fit_transform(varying_timeframe_data)

        # Check that temporal features are properly calculated
        temporal_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
        for feature in temporal_features:
            if feature in result.columns:
                # Check that values are in expected ranges
                assert result[feature].min() >= -1
                assert result[feature].max() <= 1

    def test_sequence_creation_irregular(self, varying_timeframe_data):
        """Test that sequence creation works with irregular timeframes."""
        from trading_rl_agent.data.preprocessing import create_sequences

        pipeline = FeaturePipeline()
        result = pipeline.fit_transform(varying_timeframe_data)

        # Create sequences
        sequences, targets = create_sequences(result, sequence_length=20, target_column="close")

        # Should have sequences
        assert len(sequences) > 0
        assert len(targets) > 0

        # Check shapes
        assert sequences.shape[1] == 20  # sequence length
        assert sequences.shape[2] > 0  # number of features


class TestFeatureConsistency:
    """Test that features are consistent across different configurations."""

    @pytest.fixture
    def test_data(self):
        """Create test data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=50, freq="1H")

        data = []
        for i, date in enumerate(dates):
            data.append(
                {
                    "timestamp": date,
                    "symbol": "AAPL",
                    "open": 100 + i * 0.1,
                    "high": 101 + i * 0.1,
                    "low": 99 + i * 0.1,
                    "close": 100.5 + i * 0.1,
                    "volume": 1000 + i * 10,
                }
            )

        return pd.DataFrame(data)

    def test_feature_names_consistency(self, test_data):
        """Test that feature names are consistent."""
        pipeline1 = FeaturePipeline()
        pipeline2 = FeaturePipeline()

        # Get feature names before fitting
        names1 = pipeline1.get_feature_names()
        names2 = pipeline2.get_feature_names()

        # Should be identical
        assert set(names1) == set(names2)

        # Fit and check again
        pipeline1.fit(test_data)
        pipeline2.fit(test_data)

        names1_fitted = pipeline1.get_feature_names()
        names2_fitted = pipeline2.get_feature_names()

        assert set(names1_fitted) == set(names2_fitted)

    def test_normalization_consistency(self, test_data):
        """Test that normalization produces consistent results."""
        config1 = NormalizationConfig(method="robust", per_symbol=True)
        config2 = NormalizationConfig(method="robust", per_symbol=True)

        normalizer1 = FeatureNormalizer(config1)
        normalizer2 = FeatureNormalizer(config2)

        # Fit and transform with same config
        result1 = normalizer1.fit_transform(test_data, symbol_column="symbol")
        result2 = normalizer2.fit_transform(test_data, symbol_column="symbol")

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2)

    def test_pipeline_persistence(self, test_data):
        """Test that pipeline can be saved and loaded consistently."""
        pipeline = FeaturePipeline()
        pipeline.fit(test_data)

        # Save pipeline
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pipeline_path = f.name

        try:
            pipeline.save_pipeline(pipeline_path)

            # Load pipeline
            loaded_pipeline = FeaturePipeline.load_pipeline(pipeline_path)

            # Transform with both pipelines
            result1 = pipeline.transform(test_data)
            result2 = loaded_pipeline.transform(test_data)

            # Results should be identical
            pd.testing.assert_frame_equal(result1, result2)

        finally:
            # Cleanup
            from pathlib import Path

            if os.path.exists(pipeline_path):
                Path(pipeline_path).unlink()
            normalizer_path = pipeline_path.replace(".pkl", "_normalizer.pkl")
            if os.path.exists(normalizer_path):
                Path(normalizer_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
