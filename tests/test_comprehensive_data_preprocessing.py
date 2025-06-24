"""
Comprehensive tests for data preprocessing utilities.
Tests feature engineering, data validation, preprocessing pipelines, and edge cases.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# Import data utilities
from src.data.features import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_macd,
    compute_obv,
    compute_stochastic,
    compute_williams_r,
    detect_doji,
    detect_engulfing,
    detect_hammer,
    detect_shooting_star,
    generate_features,
)
from src.data.preprocessing import create_sequences, standardize_data
from src.data.synthetic import fetch_synthetic_data


class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_technical_indicators_basic(self):
        """Test basic technical indicators with standard data."""
        # Create test data
        prices = [100, 101, 102, 101, 100, 99, 98, 99, 100, 101]
        data = pd.DataFrame(
            {
                "close": prices,
                "high": [p * 1.02 for p in prices],
                "low": [p * 0.98 for p in prices],
                "volume": [1000] * len(prices),
            }
        )

        # Test EMA
        ema = compute_ema(data["close"], window=5)
        assert len(ema) == len(prices)
        assert not np.isnan(ema.iloc[-1])

        # Test MACD
        macd, signal, histogram = compute_macd(data["close"])
        assert len(macd) == len(prices)
        assert len(signal) == len(prices)
        assert len(histogram) == len(prices)

        # Test ATR
        atr = compute_atr(data["high"], data["low"], data["close"])
        assert len(atr) == len(prices)
        assert np.all(atr >= 0)  # ATR should be non-negative

        # Test Bollinger Bands
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(data["close"])
        assert len(bb_upper) == len(prices)
        assert np.all(bb_upper >= bb_middle)  # Upper should be >= middle
        assert np.all(bb_middle >= bb_lower)  # Middle should be >= lower

    def test_technical_indicators_edge_cases(self):
        """Test technical indicators with edge cases."""
        # Constant prices
        constant_prices = [100] * 20
        data = pd.DataFrame(
            {
                "close": constant_prices,
                "high": constant_prices,
                "low": constant_prices,
                "volume": [1000] * len(constant_prices),
            }
        )

        # Should handle constant prices
        ema = compute_ema(data["close"])
        assert not np.isnan(ema.iloc[-1])
        assert ema.iloc[-1] == 100  # Should equal the constant value

        atr = compute_atr(data["high"], data["low"], data["close"])
        assert np.all(atr == 0)  # ATR should be zero for constant prices

        # Single value
        single_data = pd.DataFrame(
            {"close": [100], "high": [100], "low": [100], "volume": [1000]}
        )

        # Should handle single values gracefully
        ema_single = compute_ema(single_data["close"])
        assert len(ema_single) == 1
        assert not np.isnan(ema_single.iloc[0])

    def test_candlestick_patterns(self):
        """Test candlestick pattern detection."""
        # Create test data for doji pattern (open ≈ close)
        doji_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [98, 99, 100],
                "close": [100.1, 101.05, 102.02],  # Very close to open
            }
        )

        doji_pattern = detect_doji(doji_data)
        assert len(doji_pattern) == 3
        assert doji_pattern.iloc[-1] is True  # Last candle should be doji

        # Create test data for hammer pattern
        hammer_data = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [95, 96, 97],  # Long lower shadow
                "close": [100.5, 101.5, 102.5],
            }
        )

        hammer_pattern = detect_hammer(hammer_data)
        assert len(hammer_pattern) == 3

    def test_feature_generation_pipeline(self):
        """Test the complete feature generation pipeline."""
        # Create realistic market data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        base_price = 100
        prices = [base_price]
        for _ in range(99):
            change = np.random.normal(0, 0.02) * prices[-1]
            prices.append(max(1, prices[-1] + change))

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
            }
        )

        # Generate features
        enhanced_data = generate_features(data)

        # Check that features were added
        expected_features = [
            "ema_12",
            "ema_26",
            "macd",
            "macd_signal",
            "atr",
            "bb_upper",
            "bb_lower",
        ]
        for feature in expected_features:
            assert feature in enhanced_data.columns, f"Missing feature: {feature}"

        # Check that data length is preserved
        assert len(enhanced_data) == len(data)

        # Check that features are numeric
        for feature in expected_features:
            assert enhanced_data[feature].dtype in [np.float64, np.float32]


class TestDataValidation:
    """Test data validation and cleaning functions."""

    def test_missing_value_handling(self):
        """Test handling of missing values in data."""
        data_with_nans = pd.DataFrame(
            {
                "close": [100, np.nan, 102, 103, np.nan],
                "high": [101, 102, np.nan, 104, 105],
                "low": [99, 100, 101, np.nan, 103],
                "volume": [1000, 1100, 1200, 1300, np.nan],
            }
        )

        # Test that feature computation handles NaNs
        ema = compute_ema(data_with_nans["close"])
        assert len(ema) == len(data_with_nans)

        # Test that we can identify NaN locations
        nan_mask = data_with_nans.isna()
        assert nan_mask.sum().sum() > 0  # Should have NaNs

    def test_infinite_value_handling(self):
        """Test handling of infinite values in data."""
        data_with_infs = pd.DataFrame(
            {
                "close": [100, float("inf"), 102, -float("inf"), 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        # Should handle infinite values gracefully
        try:
            ema = compute_ema(data_with_infs["close"])
            # If it doesn't raise an error, check that result is reasonable
            finite_ema = ema[np.isfinite(ema)]
            assert len(finite_ema) > 0
        except (ValueError, OverflowError):
            # It's acceptable to raise an error for infinite values
            pass

    def test_data_type_validation(self):
        """Test validation of data types."""
        # Test with string data (should fail or be converted)
        string_data = pd.DataFrame(
            {
                "close": ["100", "101", "102"],
                "high": ["101", "102", "103"],
                "low": ["99", "100", "101"],
                "volume": ["1000", "1100", "1200"],
            }
        )

        # Convert to numeric
        numeric_data = string_data.astype(float)
        ema = compute_ema(numeric_data["close"])
        assert not np.isnan(ema.iloc[-1])

    def test_data_range_validation(self):
        """Test validation of data ranges."""
        # Test with negative prices (should be handled appropriately)
        negative_data = pd.DataFrame(
            {
                "close": [-100, -101, -102],
                "high": [-99, -100, -101],
                "low": [-101, -102, -103],
                "volume": [1000, 1100, 1200],
            }
        )

        # Technical indicators should handle negative prices
        # (though economically they don't make sense)
        ema = compute_ema(negative_data["close"])
        assert len(ema) == 3

        # Test with very large values
        large_data = pd.DataFrame(
            {
                "close": [1e10, 1e10 + 1, 1e10 + 2],
                "high": [1e10 + 1, 1e10 + 2, 1e10 + 3],
                "low": [1e10 - 1, 1e10, 1e10 + 1],
                "volume": [1000, 1100, 1200],
            }
        )

        ema_large = compute_ema(large_data["close"])
        assert np.isfinite(ema_large.iloc[-1])


class TestDataPreprocessing:
    """Test data preprocessing functions."""

    def test_data_standardization(self):
        """Test data standardization functions."""
        # Create test data
        data = np.random.randn(100, 5)

        # Test standardization
        standardized = standardize_data(data)

        # Should have mean ~0 and std ~1
        assert abs(np.mean(standardized)) < 0.1
        assert abs(np.std(standardized) - 1) < 0.1

        # Should preserve shape
        assert standardized.shape == data.shape

    def test_sequence_creation(self):
        """Test sequence creation for time series data."""
        # Create test time series
        data = np.arange(100).reshape(100, 1)

        # Create sequences
        sequences, targets = create_sequences(data, sequence_length=10)

        # Check shapes
        assert sequences.shape == (90, 10, 1)  # 100 - 10 = 90 sequences
        assert targets.shape == (90, 1)

        # Check that sequences are correct
        assert np.array_equal(sequences[0], data[:10])
        assert np.array_equal(targets[0], data[10])

    def test_sequence_creation_edge_cases(self):
        """Test sequence creation with edge cases."""
        # Data shorter than sequence length
        short_data = np.arange(5).reshape(5, 1)

        try:
            sequences, targets = create_sequences(short_data, sequence_length=10)
            assert sequences.shape[0] == 0  # Should have no sequences
        except ValueError:
            # It's acceptable to raise an error
            pass

        # Single sequence
        exact_data = np.arange(11).reshape(11, 1)
        sequences, targets = create_sequences(exact_data, sequence_length=10)
        assert sequences.shape == (1, 10, 1)
        assert targets.shape == (1, 1)


class TestSyntheticDataGeneration:
    """Test synthetic data generation."""

    def test_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        data = fetch_synthetic_data(n_samples=100)

        # Check that data has expected structure
        expected_columns = ["open", "high", "low", "close", "volume"]
        for col in expected_columns:
            assert col in data.columns

        # Check that data has correct length
        assert len(data) == 100

        # Check that OHLC relationships are maintained
        assert (data["high"] >= data["open"]).all()
        assert (data["high"] >= data["close"]).all()
        assert (data["low"] <= data["open"]).all()
        assert (data["low"] <= data["close"]).all()

        # Check that volume is positive
        assert (data["volume"] > 0).all()

    def test_synthetic_data_parameters(self):
        """Test synthetic data generation with different parameters."""
        # Test different sample sizes
        for n_samples in [10, 50, 200]:
            data = fetch_synthetic_data(n_samples=n_samples)
            assert len(data) == n_samples

        # Test different volatility levels
        low_vol_data = fetch_synthetic_data(n_samples=100, volatility=0.001)
        high_vol_data = fetch_synthetic_data(n_samples=100, volatility=0.1)

        # High volatility should have higher price variance
        low_vol_var = np.var(low_vol_data["close"])
        high_vol_var = np.var(high_vol_data["close"])
        assert high_vol_var > low_vol_var

    def test_synthetic_data_reproducibility(self):
        """Test that synthetic data generation is reproducible."""
        np.random.seed(42)
        data1 = fetch_synthetic_data(n_samples=100)

        np.random.seed(42)
        data2 = fetch_synthetic_data(n_samples=100)

        # Should be identical with same seed
        pd.testing.assert_frame_equal(data1, data2)


class TestDataPipelineIntegration:
    """Test integration of data processing components."""

    def test_end_to_end_data_pipeline(self):
        """Test complete data processing pipeline."""
        # Generate synthetic data
        raw_data = fetch_synthetic_data(n_samples=200)

        # Add features
        enhanced_data = generate_features(raw_data)

        # Check that pipeline worked
        assert len(enhanced_data) == len(raw_data)
        assert enhanced_data.shape[1] > raw_data.shape[1]  # More columns

        # Extract features for ML
        feature_columns = [
            col
            for col in enhanced_data.columns
            if col not in ["open", "high", "low", "close", "volume"]
        ]
        features = enhanced_data[feature_columns].values

        # Should be able to standardize
        standardized_features = standardize_data(features)
        assert standardized_features.shape == features.shape

        # Should be able to create sequences
        sequences, targets = create_sequences(standardized_features, sequence_length=10)
        assert sequences.shape[0] > 0
        assert sequences.shape[1] == 10

    def test_pipeline_with_missing_data(self):
        """Test pipeline robustness with missing data."""
        # Generate data with missing values
        raw_data = fetch_synthetic_data(n_samples=100)

        # Introduce missing values
        missing_indices = np.random.choice(100, size=10, replace=False)
        raw_data.loc[missing_indices, "close"] = np.nan

        # Pipeline should handle missing data
        try:
            enhanced_data = generate_features(raw_data)
            # Should complete without error or handle gracefully
            assert len(enhanced_data) == len(raw_data)
        except ValueError as e:
            # Acceptable to raise error for missing data
            assert "nan" in str(e).lower() or "missing" in str(e).lower()

    def test_pipeline_performance(self, benchmark):
        """Benchmark data processing pipeline performance."""

        def process_data():
            raw_data = fetch_synthetic_data(n_samples=1000)
            enhanced_data = generate_features(raw_data)
            features = enhanced_data.select_dtypes(include=[np.number]).values
            standardized = standardize_data(features)
            sequences, targets = create_sequences(standardized, sequence_length=20)
            return sequences, targets

        # Should complete in reasonable time
        result = benchmark(process_data)
        sequences, targets = result
        assert sequences.shape[0] > 0


class TestDataValidationPipeline:
    """Test comprehensive data validation pipeline."""

    def test_data_quality_checks(self):
        """Test comprehensive data quality validation."""
        # Create data with various quality issues
        problematic_data = pd.DataFrame(
            {
                "open": [100, np.nan, 102, float("inf"), 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 0, -100, 1300, 1400],  # Zero and negative volume
            }
        )

        # Validation checks
        quality_issues = {}

        # Check for missing values
        quality_issues["missing_values"] = problematic_data.isna().sum().sum()

        # Check for infinite values
        numeric_cols = problematic_data.select_dtypes(include=[np.number]).columns
        quality_issues["infinite_values"] = sum(
            np.isinf(problematic_data[col]).sum() for col in numeric_cols
        )

        # Check for negative volumes
        if "volume" in problematic_data.columns:
            quality_issues["negative_volume"] = (problematic_data["volume"] < 0).sum()

        # Check for zero volumes
        if "volume" in problematic_data.columns:
            quality_issues["zero_volume"] = (problematic_data["volume"] == 0).sum()

        # Should detect quality issues
        assert quality_issues["missing_values"] > 0
        assert quality_issues["infinite_values"] > 0
        assert quality_issues["negative_volume"] > 0
        assert quality_issues["zero_volume"] > 0

    def test_data_cleaning_pipeline(self):
        """Test automated data cleaning pipeline."""
        # Create dirty data
        dirty_data = pd.DataFrame(
            {
                "open": [100, np.nan, 102, float("inf"), 104],
                "high": [101, 102, np.nan, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000, 0, 1200, 1300, 1400],
            }
        )

        # Clean data
        cleaned_data = dirty_data.copy()

        # Forward fill missing values
        cleaned_data = cleaned_data.fillna(method="ffill")

        # Replace infinite values with NaN and then forward fill
        cleaned_data = cleaned_data.replace([np.inf, -np.inf], np.nan)
        cleaned_data = cleaned_data.fillna(method="ffill")

        # Replace zero volumes with mean volume
        if "volume" in cleaned_data.columns:
            mean_volume = cleaned_data["volume"][cleaned_data["volume"] > 0].mean()
            cleaned_data.loc[cleaned_data["volume"] <= 0, "volume"] = mean_volume

        # Validate cleaning
        assert not cleaned_data.isna().any().any()  # No missing values
        assert (
            not np.isinf(cleaned_data.select_dtypes(include=[np.number])).any().any()
        )  # No infinite values
        if "volume" in cleaned_data.columns:
            assert (cleaned_data["volume"] > 0).all()  # All positive volumes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
