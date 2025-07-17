import numpy as np
import pandas as pd
import pytest

from src.trading_rl_agent.data.market_patterns import (
    STATSMODELS_AVAILABLE,
    MarketPatternGenerator,
    MarketRegime,
    PatternType,
)


class TestMarketPatternGenerator:
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

    def test_init(self):
        gen = MarketPatternGenerator()
        assert gen.base_price == 100.0
        assert gen.base_volatility == 0.02
        assert gen.seed is None
        gen = MarketPatternGenerator(base_price=200.0, base_volatility=0.05, seed=123)
        assert gen.base_price == 200.0
        assert gen.base_volatility == 0.05
        assert gen.seed == 123

    def test_generate_trend_pattern_uptrend(self, generator):
        df = generator.generate_trend_pattern(n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.02)
        assert isinstance(df, pd.DataFrame)
        assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
        assert len(df) == 50
        assert df["pattern_type"].iloc[0] == "uptrend"
        assert df["trend_strength"].iloc[0] == 0.001
        assert (df["high"] >= df["low"]).all()
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["low"] <= df["open"]).all()
        assert (df["low"] <= df["close"]).all()
        assert (df["volume"] > 0).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        assert df["close"].iloc[-1] > df["close"].iloc[0]

    def test_generate_trend_pattern_downtrend(self, generator):
        df = generator.generate_trend_pattern(
            n_periods=50, trend_type="downtrend", trend_strength=0.001, volatility=0.02
        )
        assert df["pattern_type"].iloc[0] == "downtrend"
        assert df["close"].iloc[-1] < df["close"].iloc[0]

    def test_generate_trend_pattern_sideways(self, generator):
        df = generator.generate_trend_pattern(
            n_periods=50, trend_type="sideways", trend_strength=0.001, volatility=0.02
        )
        assert df["pattern_type"].iloc[0] == "sideways"

    def test_generate_trend_pattern_with_regime_changes(self, generator):
        regime_changes = [
            {"period": 10, "new_volatility": 2.0, "new_drift": 0.002},
            {"period": 30, "new_volatility": 0.5, "new_drift": -0.001},
        ]
        df = generator.generate_trend_pattern(
            n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.02, regime_changes=regime_changes
        )
        assert len(df) == 50

    def test_generate_reversal_pattern_head_and_shoulders(self, generator):
        df = generator.generate_reversal_pattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS.value,
            n_periods=252,
            pattern_intensity=0.5,
            base_volatility=0.02,
        )
        assert len(df) == 252
        assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_generate_reversal_pattern_inverse_head_and_shoulders(self, generator):
        df = generator.generate_reversal_pattern(
            pattern_type=PatternType.INVERSE_HEAD_AND_SHOULDERS.value,
            n_periods=252,
            pattern_intensity=0.5,
            base_volatility=0.02,
        )
        assert len(df) == 252

    def test_generate_reversal_pattern_double_top(self, generator):
        df = generator.generate_reversal_pattern(
            pattern_type=PatternType.DOUBLE_TOP.value, n_periods=252, pattern_intensity=0.5, base_volatility=0.02
        )
        assert len(df) == 252

    def test_generate_reversal_pattern_double_bottom(self, generator):
        df = generator.generate_reversal_pattern(
            pattern_type=PatternType.DOUBLE_BOTTOM.value, n_periods=252, pattern_intensity=0.5, base_volatility=0.02
        )
        assert len(df) == 252

    def test_generate_reversal_pattern_invalid_type(self, generator):
        with pytest.raises(ValueError, match="Unknown pattern type"):
            generator.generate_reversal_pattern(
                pattern_type="invalid_pattern", n_periods=252, pattern_intensity=0.5, base_volatility=0.02
            )

    def test_generate_volatility_clustering(self, generator):
        volatility_regimes = [
            {"volatility": 0.01, "drift": 0.001, "label": "low_vol"},
            {"volatility": 0.05, "drift": 0.001, "label": "high_vol"},
            {"volatility": 0.02, "drift": -0.001, "label": "medium_vol"},
        ]
        regime_durations = [50, 30, 20]
        df = generator.generate_volatility_clustering(
            n_periods=100, volatility_regimes=volatility_regimes, regime_durations=regime_durations, base_price=100.0
        )
        assert len(df) == 100
        assert "volatility_regime" in df.columns
        assert all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_generate_volatility_clustering_garch(self, generator):
        if not STATSMODELS_AVAILABLE:
            pytest.skip("statsmodels/arch not available")
        volatility_regimes = [
            {"omega": 0.001, "alpha": 0.1, "beta": 0.8, "mu": 0.0, "label": "garch_regime"},
        ]
        regime_durations = [50]
        df = generator.generate_volatility_clustering(
            n_periods=50, volatility_regimes=volatility_regimes, regime_durations=regime_durations, base_price=100.0
        )
        assert len(df) == 50
        assert "volatility_regime" in df.columns

    def test_generate_microstructure_effects(self, generator, sample_data):
        df = generator.generate_microstructure_effects(
            base_data=sample_data, bid_ask_spread=0.001, order_book_depth=10, tick_size=0.01, market_impact=0.0001
        )
        assert "bid" in df.columns
        assert "ask" in df.columns
        assert "order_book_depth" in df.columns
        assert "spread" in df.columns
        assert (df["ask"] > df["bid"]).all()
        assert (df["spread"] > 0).all()
        for col in ["open", "high", "low", "close"]:
            assert np.allclose(df[col] % 0.01, 0, atol=1e-8)

    def test_generate_correlated_assets(self, generator):
        n_assets = 3
        n_periods = 10
        assets_data = generator.generate_correlated_assets(n_assets=n_assets, n_periods=n_periods)
        assert len(assets_data) == n_assets
        for symbol, df in assets_data.items():
            assert len(df) == n_periods
            assert "symbol" in df.columns
            assert df["symbol"].iloc[0] == symbol
        correlation_matrix = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.7],
                [0.3, 0.7, 1.0],
            ]
        )
        assets_data = generator.generate_correlated_assets(
            n_assets=n_assets,
            n_periods=n_periods,
            correlation_matrix=correlation_matrix,
            base_prices=[100, 50, 200],
            volatilities=[0.02, 0.03, 0.025],
        )
        assert len(assets_data) == n_assets

    def test_detect_market_regime(self, generator, sample_data):
        df = generator.detect_market_regime(sample_data, window=10)
        assert "returns" in df.columns
        assert "rolling_mean" in df.columns
        assert "rolling_std" in df.columns
        assert "rolling_skew" in df.columns
        assert "regime" in df.columns
        valid_regimes = [regime.value for regime in MarketRegime]
        assert all(regime in valid_regimes for regime in df["regime"].dropna())

    def test_validate_pattern_quality_trend(self, generator):
        df = generator.generate_trend_pattern(n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.02)
        validation = generator.validate_pattern_quality(df, "uptrend")
        assert "pattern_type" in validation
        assert "data_quality" in validation
        assert "pattern_quality" in validation
        assert "statistical_tests" in validation
        data_quality = validation["data_quality"]
        assert "missing_values" in data_quality
        assert "high_ge_low" in data_quality
        assert "volume_positive" in data_quality
        assert "price_positive" in data_quality
        pattern_quality = validation["pattern_quality"]
        assert "linear_trend" in pattern_quality
        assert "trend_consistency" in pattern_quality
        assert "trend_strength" in pattern_quality
        statistical_tests = validation["statistical_tests"]
        assert "stationarity" in statistical_tests
        assert "normality" in statistical_tests
        assert "autocorrelation" in statistical_tests

    def test_validate_pattern_quality_reversal(self, generator):
        df = generator.generate_reversal_pattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS.value,
            n_periods=252,
            pattern_intensity=0.5,
            base_volatility=0.02,
        )
        validation = generator.validate_pattern_quality(df, PatternType.HEAD_AND_SHOULDERS.value)
        pattern_quality = validation["pattern_quality"]
        assert "peak_count" in pattern_quality
        assert "trough_count" in pattern_quality
        assert "pattern_symmetry" in pattern_quality

    def test_edge_cases(self, generator):
        df = generator.generate_trend_pattern(n_periods=1, trend_type="uptrend")
        assert len(df) == 1
        with pytest.raises(ValueError):
            generator.generate_trend_pattern(n_periods=0, trend_type="uptrend")
        with pytest.raises(ValueError):
            generator.generate_trend_pattern(n_periods=-1, trend_type="uptrend")

    def test_reproducibility(self):
        gen1 = MarketPatternGenerator(seed=42)
        gen2 = MarketPatternGenerator(seed=42)
        df1 = gen1.generate_trend_pattern(n_periods=50, trend_type="uptrend")
        df2 = gen2.generate_trend_pattern(n_periods=50, trend_type="uptrend")
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_results(self):
        gen1 = MarketPatternGenerator(seed=42)
        gen2 = MarketPatternGenerator(seed=123)
        df1 = gen1.generate_trend_pattern(n_periods=50, trend_type="uptrend")
        df2 = gen2.generate_trend_pattern(n_periods=50, trend_type="uptrend")
        assert not df1["close"].equals(df2["close"])

    def test_pattern_intensity_effects(self, generator):
        df_low = generator.generate_reversal_pattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS.value,
            n_periods=252,
            pattern_intensity=0.1,
            base_volatility=0.02,
        )
        df_high = generator.generate_reversal_pattern(
            pattern_type=PatternType.HEAD_AND_SHOULDERS.value,
            n_periods=252,
            pattern_intensity=0.9,
            base_volatility=0.02,
        )
        assert len(df_low) == len(df_high)

    def test_volatility_effects(self, generator):
        df_low_vol = generator.generate_trend_pattern(
            n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.01
        )
        df_high_vol = generator.generate_trend_pattern(
            n_periods=50, trend_type="uptrend", trend_strength=0.001, volatility=0.05
        )
        low_vol_std = df_low_vol["close"].std()
        high_vol_std = df_high_vol["close"].std()
        assert high_vol_std > low_vol_std

    def test_correlation_matrix_validation(self, generator):
        invalid_correlation = np.array([[1.0, 1.1], [1.1, 1.0]])
        with pytest.raises(np.linalg.LinAlgError):
            generator.generate_correlated_assets(n_assets=2, n_periods=50, correlation_matrix=invalid_correlation)

    def test_empty_regime_changes(self, generator):
        df = generator.generate_trend_pattern(n_periods=50, trend_type="uptrend", regime_changes=[])
        assert len(df) == 50
        assert df["pattern_type"].iloc[0] == "uptrend"

    def test_missing_statsmodels_graceful_degradation(self, generator):
        df = generator.generate_volatility_clustering(
            n_periods=50, volatility_regimes=[{"volatility": 0.02, "drift": 0}], regime_durations=[50]
        )
        assert len(df) == 50
        assert "volatility_regime" in df.columns


class TestMarketRegime:
    def test_regime_values(self):
        for regime in MarketRegime:
            assert isinstance(regime.value, str)
            assert len(regime.value) > 0

    def test_regime_uniqueness(self):
        values = [regime.value for regime in MarketRegime]
        assert len(values) == len(set(values))


class TestPatternType:
    def test_pattern_values(self):
        for pattern in PatternType:
            assert isinstance(pattern.value, str)
            assert len(pattern.value) > 0

    def test_pattern_uniqueness(self):
        values = [pattern.value for pattern in PatternType]
        assert len(values) == len(set(values))
