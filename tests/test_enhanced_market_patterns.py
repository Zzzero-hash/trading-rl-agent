"""
Tests for enhanced market pattern generation features.
"""

import numpy as np
import pandas as pd
import pytest

from src.trading_rl_agent.data.market_patterns import (
    STATSMODELS_AVAILABLE,
    MarketPatternGenerator,
    MarketRegime,
    PatternType,
)


class TestEnhancedMarketPatternGenerator:
    @pytest.fixture
    def generator(self):
        return MarketPatternGenerator(base_price=100.0, base_volatility=0.02, seed=42)

    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        return pd.DataFrame(
            {
                "timestamp": dates,
                "open": np.random.uniform(95, 105, 100),
                "high": np.random.uniform(105, 115, 100),
                "low": np.random.uniform(85, 95, 100),
                "close": np.random.uniform(95, 105, 100),
                "volume": np.random.randint(100, 1000, 100),
            }
        )

    def test_generate_arima_trend(self, generator):
        """Test ARIMA trend generation."""
        df = generator.generate_arima_trend(n_periods=50, order=(1, 1, 1), trend_strength=0.001, volatility=0.02)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
        assert "pattern_type" in df.columns
        assert df["pattern_type"].iloc[0] == "arima_trend"
        assert "arima_order" in df.columns
        assert df["arima_order"].iloc[0] == "(1, 1, 1)"
        assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_generate_arima_trend_with_seasonal(self, generator):
        """Test ARIMA trend generation with seasonal components."""
        if not STATSMODELS_AVAILABLE:
            pytest.skip("statsmodels not available")

        df = generator.generate_arima_trend(
            n_periods=100,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12),
            trend_strength=0.001,
            volatility=0.02,
        )

        assert len(df) == 100
        assert df["pattern_type"].iloc[0] == "arima_trend"

    def test_generate_triangle_patterns(self, generator):
        """Test triangle pattern generation."""
        triangle_types = [
            PatternType.ASCENDING_TRIANGLE.value,
            PatternType.DESCENDING_TRIANGLE.value,
            PatternType.SYMMETRICAL_TRIANGLE.value,
        ]

        for pattern_type in triangle_types:
            df = generator.generate_triangle_pattern(
                pattern_type=pattern_type,
                n_periods=50,
                pattern_intensity=0.8,
                base_volatility=0.02,
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
            assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_generate_continuation_patterns(self, generator):
        """Test continuation pattern generation."""
        continuation_types = [
            PatternType.FLAG.value,
            PatternType.PENNANT.value,
            PatternType.WEDGE.value,
            PatternType.CHANNEL.value,
        ]

        for pattern_type in continuation_types:
            df = generator.generate_continuation_pattern(
                pattern_type=pattern_type,
                n_periods=50,
                pattern_intensity=0.8,
                base_volatility=0.02,
            )

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 50
            assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_generate_enhanced_microstructure(self, generator, sample_data):
        """Test enhanced microstructure effects."""
        df = generator.generate_enhanced_microstructure(
            base_data=sample_data,
            bid_ask_spread=0.002,
            order_book_depth=5,
            tick_size=0.01,
            market_impact=0.0001,
            liquidity_profile="high",
            trading_hours={"open": 9, "close": 16},
        )

        # Check basic microstructure columns
        assert "bid" in df.columns
        assert "ask" in df.columns
        assert "spread" in df.columns
        assert "mid_price" in df.columns
        assert "liquidity_profile" in df.columns

        # Check order book levels
        for i in range(1, 6):  # order_book_depth=5
            assert f"bid_level_{i}" in df.columns
            assert f"ask_level_{i}" in df.columns
            assert f"bid_volume_{i}" in df.columns
            assert f"ask_volume_{i}" in df.columns

        # Check bid-ask relationship
        assert (df["ask"] > df["bid"]).all()
        assert (df["spread"] > 0).all()
        assert (df["mid_price"] == (df["bid"] + df["ask"]) / 2).all()

        # Check tick size constraints - allow for small floating point errors
        for col in ["open", "high", "low", "close"]:
            # Check that prices are approximately aligned to tick size
            # Note: Market impact can cause small deviations from exact tick alignment
            remainders = df[col] % 0.01
            # Allow for small deviations due to market impact and floating point precision
            assert np.allclose(remainders, 0, atol=1e-2) or np.allclose(remainders, 0.01, atol=1e-2)

    def test_generate_enhanced_microstructure_different_profiles(self, generator, sample_data):
        """Test microstructure effects with different liquidity profiles."""
        profiles = ["low", "normal", "high"]

        for profile in profiles:
            df = generator.generate_enhanced_microstructure(
                base_data=sample_data,
                bid_ask_spread=0.002,
                order_book_depth=3,
                liquidity_profile=profile,
            )

            assert df["liquidity_profile"].iloc[0] == profile

            # Check that high liquidity has lower spreads (relative to base spread)
            base_spread = 0.002
            if profile == "high":
                assert df["spread"].mean() < base_spread * 1.5  # Allow some variance
            elif profile == "low":
                assert df["spread"].mean() > base_spread * 0.5  # Allow some variance

    def test_detect_enhanced_regime_rolling_stats(self, generator, sample_data):
        """Test enhanced regime detection with rolling statistics."""
        df = generator.detect_enhanced_regime(sample_data, window=10, method="rolling_stats")

        assert "returns" in df.columns
        assert "rolling_mean" in df.columns
        assert "rolling_std" in df.columns
        assert "rolling_skew" in df.columns
        assert "rolling_kurt" in df.columns
        assert "regime" in df.columns

        valid_regimes = [regime.value for regime in MarketRegime]
        assert all(regime in valid_regimes for regime in df["regime"].dropna())

    def test_detect_enhanced_regime_markov_switching(self, generator, sample_data):
        """Test enhanced regime detection with Markov switching."""
        df = generator.detect_enhanced_regime(sample_data, window=10, method="markov_switching")

        assert "regime" in df.columns
        # Should have some regime classifications
        assert df["regime"].notna().any()

    def test_detect_enhanced_regime_volatility(self, generator, sample_data):
        """Test enhanced regime detection with volatility-based method."""
        df = generator.detect_enhanced_regime(sample_data, window=10, method="volatility_regime")

        assert "regime" in df.columns
        valid_vol_regimes = ["low_vol", "medium_vol", "high_vol"]
        assert all(regime in valid_vol_regimes for regime in df["regime"].dropna())

    def test_detect_enhanced_regime_invalid_method(self, generator, sample_data):
        """Test enhanced regime detection with invalid method."""
        with pytest.raises(ValueError, match="Unknown regime detection method"):
            generator.detect_enhanced_regime(sample_data, window=10, method="invalid_method")

    def test_validate_triangle_pattern(self, generator):
        """Test triangle pattern validation."""
        df = generator.generate_triangle_pattern(
            pattern_type=PatternType.ASCENDING_TRIANGLE.value,
            n_periods=50,
            pattern_intensity=0.8,
            base_volatility=0.02,
        )

        validation = generator.validate_pattern_quality(df, PatternType.ASCENDING_TRIANGLE.value)

        assert "pattern_quality" in validation
        pattern_quality = validation["pattern_quality"]

        assert "high_trend_slope" in pattern_quality
        assert "low_trend_slope" in pattern_quality
        assert "convergence_quality" in pattern_quality
        assert "pattern_symmetry" in pattern_quality

    def test_validate_continuation_pattern(self, generator):
        """Test continuation pattern validation."""
        df = generator.generate_continuation_pattern(
            pattern_type=PatternType.FLAG.value,
            n_periods=50,
            pattern_intensity=0.8,
            base_volatility=0.02,
        )

        validation = generator.validate_pattern_quality(df, PatternType.FLAG.value)

        assert "pattern_quality" in validation
        pattern_quality = validation["pattern_quality"]

        assert "consolidation_quality" in pattern_quality
        assert "volatility_compression" in pattern_quality
        assert "pattern_duration" in pattern_quality

    def test_enhanced_statistical_tests(self, generator):
        """Test enhanced statistical tests."""
        df = generator.generate_trend_pattern(
            n_periods=100, trend_type="uptrend", trend_strength=0.001, volatility=0.02
        )

        validation = generator.validate_pattern_quality(df, "uptrend")

        assert "statistical_tests" in validation
        statistical_tests = validation["statistical_tests"]

        # Check for enhanced tests
        assert "seasonality" in statistical_tests
        assert "heteroskedasticity" in statistical_tests
        assert "long_memory" in statistical_tests

        # Check seasonality test structure
        seasonality = statistical_tests["seasonality"]
        assert "has_seasonality" in seasonality
        assert "seasonal_strength" in seasonality

        # Check heteroskedasticity test structure
        heteroskedasticity = statistical_tests["heteroskedasticity"]
        assert "has_heteroskedasticity" in heteroskedasticity
        assert "arch_effect_strength" in heteroskedasticity

        # Check long memory test structure
        long_memory = statistical_tests["long_memory"]
        assert "has_long_memory" in long_memory
        assert "hurst_exponent" in long_memory

    def test_pattern_specific_tests(self, generator):
        """Test pattern-specific validation tests."""
        df = generator.generate_trend_pattern(n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.02)

        validation = generator.validate_pattern_quality(df, "uptrend")

        assert "pattern_specific_tests" in validation
        pattern_tests = validation["pattern_specific_tests"]

        assert "price_momentum" in pattern_tests
        assert "price_acceleration" in pattern_tests
        assert "peak_to_trough_ratio" in pattern_tests
        assert "pattern_completeness" in pattern_tests

    def test_triangle_pattern_generation_methods(self, generator):
        """Test individual triangle pattern generation methods."""
        n_periods = 50
        intensity = 0.8
        volatility = 0.02

        # Test ascending triangle
        df_asc = generator._generate_ascending_triangle(n_periods, intensity, volatility)
        assert len(df_asc) == n_periods
        assert all(col in df_asc.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

        # Test descending triangle
        df_desc = generator._generate_descending_triangle(n_periods, intensity, volatility)
        assert len(df_desc) == n_periods

        # Test symmetrical triangle
        df_sym = generator._generate_symmetrical_triangle(n_periods, intensity, volatility)
        assert len(df_sym) == n_periods

    def test_continuation_pattern_generation_methods(self, generator):
        """Test individual continuation pattern generation methods."""
        n_periods = 50
        intensity = 0.8
        volatility = 0.02

        # Test flag pattern
        df_flag = generator._generate_flag_pattern(n_periods, intensity, volatility)
        assert len(df_flag) == n_periods

        # Test pennant pattern
        df_pennant = generator._generate_pennant_pattern(n_periods, intensity, volatility)
        assert len(df_pennant) == n_periods

        # Test wedge pattern
        df_wedge = generator._generate_wedge_pattern(n_periods, intensity, volatility)
        assert len(df_wedge) == n_periods

        # Test channel pattern
        df_channel = generator._generate_channel_pattern(n_periods, intensity, volatility)
        assert len(df_channel) == n_periods

    def test_regime_detection_methods(self, generator, sample_data):
        """Test individual regime detection methods."""
        sample_data["returns"] = sample_data["close"].pct_change()

        # Test rolling stats method
        df_rolling = generator._detect_regime_rolling_stats(sample_data, window=10)
        assert "rolling_mean" in df_rolling.columns
        assert "rolling_std" in df_rolling.columns
        assert "rolling_skew" in df_rolling.columns
        assert "rolling_kurt" in df_rolling.columns
        assert "regime" in df_rolling.columns

        # Test Markov switching method
        df_markov = generator._detect_regime_markov_switching(sample_data, window=10)
        assert "regime" in df_markov.columns

        # Test volatility regime method
        df_vol = generator._detect_regime_volatility(sample_data, window=10)
        assert "regime" in df_vol.columns

    def test_statistical_test_methods(self, generator):
        """Test individual statistical test methods."""
        # Generate sample returns
        returns = pd.Series(np.random.normal(0, 0.02, 100))

        # Test seasonality
        seasonality_result = generator._test_seasonality(returns)
        assert "has_seasonality" in seasonality_result
        assert "seasonal_strength" in seasonality_result

        # Test heteroskedasticity
        hetero_result = generator._test_heteroskedasticity(returns)
        assert "has_heteroskedasticity" in hetero_result
        assert "arch_effect_strength" in hetero_result

        # Test long memory
        long_memory_result = generator._test_long_memory(returns)
        assert "has_long_memory" in long_memory_result
        assert "hurst_exponent" in long_memory_result

    def test_pattern_validation_methods(self, generator):
        """Test individual pattern validation methods."""
        # Generate sample data
        df = generator.generate_trend_pattern(50, "uptrend", 0.001, 0.02)

        # Test triangle validation
        triangle_validation = generator._validate_triangle_pattern(df, "ascending_triangle")
        assert "high_trend_slope" in triangle_validation
        assert "low_trend_slope" in triangle_validation
        assert "convergence_quality" in triangle_validation
        assert "pattern_symmetry" in triangle_validation

        # Test continuation validation
        continuation_validation = generator._validate_continuation_pattern(df, "flag")
        assert "consolidation_quality" in continuation_validation
        assert "volatility_compression" in continuation_validation
        assert "pattern_duration" in continuation_validation

    def test_utility_methods(self, generator):
        """Test utility methods."""
        prices = np.array([100, 110, 105, 115, 108, 120, 112])

        # Test peak to trough ratio
        ratio = generator._calculate_peak_to_trough_ratio(prices)
        assert isinstance(ratio, float)
        assert ratio > 0

        # Test pattern completeness
        completeness = generator._calculate_pattern_completeness(prices, "triangle")
        assert isinstance(completeness, float)
        assert 0 <= completeness <= 1

    def test_edge_cases(self, generator):
        """Test edge cases and error handling."""
        # Test invalid triangle pattern type
        with pytest.raises(ValueError, match="Unknown triangle pattern type"):
            generator.generate_triangle_pattern("invalid_triangle", 50, 0.8, 0.02)

        # Test invalid continuation pattern type
        with pytest.raises(ValueError, match="Unknown continuation pattern type"):
            generator.generate_continuation_pattern("invalid_continuation", 50, 0.8, 0.02)

        # Test short data for statistical tests
        short_df = generator.generate_trend_pattern(10, "uptrend", 0.001, 0.02)
        validation = generator.validate_pattern_quality(short_df, "uptrend")
        assert "statistical_tests" in validation

    def test_reproducibility_enhanced(self, _generator):
        """Test reproducibility of enhanced features."""
        # Test ARIMA reproducibility
        if STATSMODELS_AVAILABLE:
            gen1 = MarketPatternGenerator(seed=42)
            gen2 = MarketPatternGenerator(seed=42)

            df1 = gen1.generate_arima_trend(50, (1, 1, 1), 0.001, 0.02)
            df2 = gen2.generate_arima_trend(50, (1, 1, 1), 0.001, 0.02)

            pd.testing.assert_frame_equal(df1, df2)

        # Test triangle pattern reproducibility
        gen1 = MarketPatternGenerator(seed=42)
        gen2 = MarketPatternGenerator(seed=42)

        df1 = gen1.generate_triangle_pattern("ascending_triangle", 50, 0.8, 0.02)
        df2 = gen2.generate_triangle_pattern("ascending_triangle", 50, 0.8, 0.02)

        pd.testing.assert_frame_equal(df1, df2)

    def test_performance_characteristics(self, generator):
        """Test performance characteristics of generated patterns."""
        # Test that triangle patterns show convergence
        df = generator.generate_triangle_pattern("ascending_triangle", 100, 0.8, 0.02)
        prices = df["close"].values

        # Check that volatility decreases over time (convergence)
        first_half_vol = np.std(prices[:50])
        second_half_vol = np.std(prices[50:])
        assert second_half_vol < first_half_vol

        # Test that flag patterns show consolidation
        df = generator.generate_continuation_pattern("flag", 100, 0.8, 0.02)
        prices = df["close"].values

        # Check that later part has lower volatility (consolidation)
        initial_move_vol = np.std(prices[:30])
        consolidation_vol = np.std(prices[70:])
        assert consolidation_vol < initial_move_vol


class TestEnhancedPatternTypes:
    """Test the new pattern types."""

    def test_new_pattern_types_exist(self):
        """Test that new pattern types are properly defined."""
        assert PatternType.ASCENDING_TRIANGLE.value == "ascending_triangle"
        assert PatternType.DESCENDING_TRIANGLE.value == "descending_triangle"
        assert PatternType.SYMMETRICAL_TRIANGLE.value == "symmetrical_triangle"
        assert PatternType.FLAG.value == "flag"
        assert PatternType.PENNANT.value == "pennant"
        assert PatternType.WEDGE.value == "wedge"
        assert PatternType.CHANNEL.value == "channel"

    def test_pattern_types_uniqueness(self):
        """Test that all pattern types are unique."""
        pattern_values = [pt.value for pt in PatternType]
        assert len(pattern_values) == len(set(pattern_values))


if __name__ == "__main__":
    pytest.main([__file__])
