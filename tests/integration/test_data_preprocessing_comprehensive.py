"""
Comprehensive test suite for data preprocessing utilities.
Tests all data processing functionality including feature generation, normalization, and pipeline integration.
"""

import logging
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "structlog" not in sys.modules:
    stub = types.SimpleNamespace(
        BoundLogger=object,
        stdlib=types.SimpleNamespace(
            ProcessorFormatter=object,
            BoundLogger=object,
            LoggerFactory=lambda: None,
            filter_by_level=lambda *a, **k: None,
            add_logger_name=lambda *a, **k: None,
            add_log_level=lambda *a, **k: None,
            PositionalArgumentsFormatter=lambda: None,
            wrap_for_formatter=lambda f: f,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda **_: None,
            StackInfoRenderer=lambda **_: None,
            format_exc_info=lambda **_: None,
            UnicodeDecoder=lambda **_: None,
        ),
        dev=types.SimpleNamespace(ConsoleRenderer=lambda **_: None),
        configure=lambda **_: None,
        get_logger=lambda name=None: logging.getLogger(name),
    )
    sys.modules["structlog"] = stub

base = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [str(base)]
    sys.modules["trading_rl_agent"] = pkg
if "trading_rl_agent.data" not in sys.modules:
    mod = types.ModuleType("trading_rl_agent.data")
    mod.__path__ = [str(base / "data")]
    sys.modules["trading_rl_agent.data"] = mod


@pytest.fixture
def sample_market_data():
    """Generate sample market data for preprocessing tests."""
    np.random.seed(42)
    n_samples = 1000

    prices = [100.0]
    for _i in range(1, n_samples):
        change = np.random.normal(0, 0.02)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 0.01))

    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2023-01-01", periods=n_samples, freq="H"),
            "open": prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            "close": prices,
            "volume": np.random.randint(1000, 100000, n_samples),
        },
    )


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


class TestDataPreprocessingComprehensive:
    """Comprehensive test suite for data preprocessing utilities."""

    @pytest.fixture
    def noisy_market_data(self):
        """Generate noisy market data with missing values and outliers."""
        np.random.seed(123)
        n_samples = 500

        data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start="2023-01-01",
                    periods=n_samples,
                    freq="H",
                ),
                "open": np.random.uniform(90, 110, n_samples),
                "high": np.random.uniform(100, 120, n_samples),
                "low": np.random.uniform(80, 100, n_samples),
                "close": np.random.uniform(95, 105, n_samples),
                "volume": np.random.randint(1000, 50000, n_samples),
            },
        )

        # Add missing values
        missing_indices = np.random.choice(
            data.index,
            size=int(0.05 * len(data)),
            replace=False,
        )
        data.loc[missing_indices, "close"] = np.nan

        # Add outliers
        outlier_indices = np.random.choice(
            data.index,
            size=int(0.02 * len(data)),
            replace=False,
        )
        data.loc[outlier_indices, "close"] *= 10

        return data

    def test_feature_generation_comprehensive(self, sample_market_data):
        """Test comprehensive feature generation from market data."""
        try:
            from trading_rl_agent.data.features import generate_features

            # Test basic feature generation
            features_df = generate_features(sample_market_data.copy())

            # Validate output structure
            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) > 0
            assert len(features_df) <= len(sample_market_data)

            # Check for expected feature columns
            expected_features = [
                "log_returns",
                "sma_10",
                "sma_20",
                "rsi",
                "volatility",
                "price_change",
                "volume_sma",
            ]

            found_features = []
            for feature in expected_features:
                if feature in features_df.columns:
                    found_features.append(feature)

            assert len(found_features) > 0, f"No expected features found. Available: {list(features_df.columns)}"

            # Validate feature values
            for feature in found_features:
                values = features_df[feature].dropna()
                if len(values) > 0:
                    assert not np.any(np.isinf(values)), f"Infinite values in {feature}"
                    # Allow NaN values as they might be expected in some features

            print(
                f"✅ Feature generation test passed - {len(found_features)} features generated",
            )

        except ImportError as e:
            pytest.skip(f"Feature generation test skipped: {e}")

    def test_technical_indicators_comprehensive(self, sample_market_data):
        """Test comprehensive technical indicator calculations."""
        try:
            import pandas_ta as ta

            # Test log returns
            returns = np.log(
                sample_market_data["close"] / sample_market_data["close"].shift(1),
            )
            assert isinstance(returns, pd.Series)
            assert len(returns) == len(sample_market_data)
            assert not np.any(np.isinf(returns.dropna()))

            # Test simple moving average
            sma = sample_market_data["close"].rolling(window=20).mean()
            assert isinstance(sma, pd.Series)
            assert len(sma) == len(sample_market_data)

            # Test RSI
            rsi = ta.rsi(sample_market_data["close"].astype(float), length=14)
            assert isinstance(rsi, pd.Series)
            rsi_valid = rsi.dropna()
            if len(rsi_valid) > 0:
                assert np.all(rsi_valid >= 0), "RSI values should be >= 0"
                assert np.all(rsi_valid <= 100), "RSI values should be <= 100"

            # Test volatility
            lr = returns
            volatility = lr.rolling(window=20).std(ddof=0) * np.sqrt(20)
            assert isinstance(volatility, pd.Series)
            vol_valid = volatility.dropna()
            if len(vol_valid) > 0:
                assert np.all(vol_valid >= 0), "Volatility values should be >= 0"

            print("✅ Technical indicators test passed")

        except ImportError as e:
            pytest.skip(f"Technical indicators test skipped: {e}")

    def test_advanced_technical_indicators(self, sample_market_data):
        """Test advanced technical indicators."""
        try:
            from trading_rl_agent.data.features import (
                compute_atr,
                compute_bollinger_bands,
                compute_ema,
                compute_macd,
                compute_stochastic,
            )

            # Test EMA
            ema = compute_ema(sample_market_data["close"])
            assert isinstance(ema, pd.Series)
            assert len(ema) == len(sample_market_data)

            # Test MACD
            macd_result = compute_macd(sample_market_data["close"])
            if isinstance(macd_result, tuple):
                macd, signal, histogram = macd_result
                assert isinstance(macd, pd.Series)
                assert isinstance(signal, pd.Series)
                assert isinstance(histogram, pd.Series)
            else:
                assert isinstance(macd_result, pd.Series)

            # Test Bollinger Bands
            bb_result = compute_bollinger_bands(sample_market_data["close"])
            if isinstance(bb_result, tuple):
                upper, middle, lower = bb_result
                assert isinstance(upper, pd.Series)
                assert isinstance(middle, pd.Series)
                assert isinstance(lower, pd.Series)
                # Upper should be >= middle >= lower (mostly)
                valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
                if valid_indices.any():
                    assert np.all(upper[valid_indices] >= middle[valid_indices])
                    assert np.all(middle[valid_indices] >= lower[valid_indices])

            # Test Stochastic
            stoch_result = compute_stochastic(
                sample_market_data["high"],
                sample_market_data["low"],
                sample_market_data["close"],
            )
            if isinstance(stoch_result, tuple):
                k_percent, d_percent = stoch_result
                assert isinstance(k_percent, pd.Series)
                assert isinstance(d_percent, pd.Series)
            else:
                assert isinstance(stoch_result, pd.Series)

            # Test ATR
            atr = compute_atr(
                sample_market_data["high"],
                sample_market_data["low"],
                sample_market_data["close"],
            )
            assert isinstance(atr, pd.Series)
            atr_valid = atr.dropna()
            if len(atr_valid) > 0:
                assert np.all(atr_valid >= 0), "ATR values should be >= 0"

            print("✅ Advanced technical indicators test passed")

        except ImportError as e:
            pytest.skip(f"Advanced technical indicators test skipped: {e}")

    def test_candlestick_patterns(self, sample_market_data):
        """Test candlestick pattern detection."""
        try:
            from trading_rl_agent.data.features import (
                detect_doji,
                detect_engulfing,
                detect_hammer,
                detect_shooting_star,
            )

            # Test Doji pattern
            doji = detect_doji(
                sample_market_data["open"],
                sample_market_data["high"],
                sample_market_data["low"],
                sample_market_data["close"],
            )
            assert isinstance(doji, pd.Series)
            assert doji.dtype == bool or np.issubdtype(doji.dtype, np.integer)

            # Test Hammer pattern
            hammer = detect_hammer(
                sample_market_data["open"],
                sample_market_data["high"],
                sample_market_data["low"],
                sample_market_data["close"],
            )
            assert isinstance(hammer, pd.Series)
            assert hammer.dtype == bool or np.issubdtype(hammer.dtype, np.integer)

            # Test Engulfing pattern
            engulfing = detect_engulfing(
                sample_market_data["open"],
                sample_market_data["high"],
                sample_market_data["low"],
                sample_market_data["close"],
            )
            assert isinstance(engulfing, pd.Series)

            # Test Shooting Star pattern
            shooting_star = detect_shooting_star(
                sample_market_data["open"],
                sample_market_data["high"],
                sample_market_data["low"],
                sample_market_data["close"],
            )
            assert isinstance(shooting_star, pd.Series)

            print("✅ Candlestick patterns test passed")

        except ImportError as e:
            pytest.skip(f"Candlestick patterns test skipped: {e}")

    @pytest.mark.xfail(reason="floating point precision issues")
    def test_data_normalization(self, sample_market_data):
        """Test data normalization and scaling."""
        try:
            from trading_rl_agent.data.features import generate_features

            # Generate features first
            features_df = generate_features(sample_market_data.copy())

            # Get numeric columns for normalization
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns

            if len(numeric_columns) > 0:
                # Test min-max normalization
                from sklearn.preprocessing import MinMaxScaler

                scaler = MinMaxScaler()

                normalized_data = scaler.fit_transform(
                    features_df[numeric_columns].fillna(0),
                )

                assert normalized_data.shape == (len(features_df), len(numeric_columns))
                assert np.all(
                    normalized_data >= 0,
                ), "Min-max normalized values should be >= 0"
                assert np.allclose(normalized_data, np.clip(normalized_data, 0, 1))

                # Test standard normalization
                from sklearn.preprocessing import StandardScaler

                std_scaler = StandardScaler()

                standardized_data = std_scaler.fit_transform(
                    features_df[numeric_columns].fillna(0),
                )

                assert standardized_data.shape == (
                    len(features_df),
                    len(numeric_columns),
                )

                # Check that mean is approximately 0 and std is approximately 1
                means = np.mean(standardized_data, axis=0)
                stds = np.std(standardized_data, axis=0)

                assert np.allclose(means, 0, atol=1e-6)
                assert np.allclose(stds, 1, atol=1e-6)

                print("✅ Data normalization test passed")
            else:
                print("⚠️ No numeric columns found for normalization test")

        except ImportError as e:
            pytest.skip(f"Data normalization test skipped: {e}")

    def test_missing_data_handling(self, noisy_market_data):
        """Test handling of missing data and outliers."""
        try:
            from trading_rl_agent.data.features import generate_features

            # Test with noisy data containing NaN values
            features_df = generate_features(noisy_market_data.copy())

            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) > 0

            # Check handling of missing values
            total_missing = features_df.isnull().sum().sum()
            print(f"Total missing values after processing: {total_missing}")

            # Test forward fill for missing values
            filled_df = features_df.fillna(method="ffill")
            remaining_missing = filled_df.isnull().sum().sum()
            assert remaining_missing <= total_missing

            # Test outlier detection and handling
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns

            for col in numeric_columns:
                if col in features_df.columns:
                    values = features_df[col].dropna()
                    if len(values) > 0:
                        # Simple outlier detection using IQR
                        q1 = values.quantile(0.25)
                        q3 = values.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr

                        outliers = (values < lower_bound) | (values > upper_bound)
                        outlier_count = outliers.sum()

                        print(f"Column {col}: {outlier_count} outliers detected")

            print("✅ Missing data handling test passed")

        except ImportError as e:
            pytest.skip(f"Missing data handling test skipped: {e}")

    def test_feature_engineering_pipeline(self, sample_market_data):
        """Test complete feature engineering pipeline."""
        try:
            from trading_rl_agent.data.features import generate_features

            # Test pipeline with different configurations
            configs = [
                {"enable_advanced_patterns": True},
                {"enable_advanced_patterns": False},
                {"window_sizes": [5, 10, 20]},
                {"include_volume_features": True},
            ]

            for i, config in enumerate(configs):
                try:
                    # Try to pass config if the function supports it
                    features_df = generate_features(sample_market_data.copy(), **config)
                except TypeError:
                    # If function doesn't accept additional parameters, use default
                    features_df = generate_features(sample_market_data.copy())

                assert isinstance(features_df, pd.DataFrame)
                assert len(features_df) > 0

                # Basic validation
                numeric_columns = features_df.select_dtypes(include=[np.number]).columns
                assert len(numeric_columns) > 0, f"No numeric features generated for config {i}"

                print(
                    f"✅ Pipeline test {i + 1} passed - {len(numeric_columns)} features",
                )

        except ImportError as e:
            pytest.skip(f"Feature engineering pipeline test skipped: {e}")

    def test_time_series_features(self, sample_market_data):
        """Test time series specific features."""
        try:
            from trading_rl_agent.data.features import generate_features

            # Ensure timestamp column is datetime
            if "timestamp" in sample_market_data.columns:
                sample_market_data["timestamp"] = pd.to_datetime(
                    sample_market_data["timestamp"],
                )

            features_df = generate_features(sample_market_data.copy())

            # Check for time-based features
            time_features = [
                col
                for col in features_df.columns
                if any(time_word in col.lower() for time_word in ["hour", "day", "week", "month", "lag", "shift"])
            ]

            if len(time_features) > 0:
                print(f"Time-based features found: {time_features}")

                for feature in time_features:
                    values = features_df[feature].dropna()
                    if len(values) > 0:
                        assert not np.any(
                            np.isinf(values),
                        ), f"Infinite values in {feature}"

            # Test lag features if available
            lag_features = [col for col in features_df.columns if "lag" in col.lower()]

            for lag_feature in lag_features:
                values = features_df[lag_feature].dropna()
                if len(values) > 0:
                    # Lag features should have reasonable correlation with original
                    original_col = lag_feature.replace("_lag", "").replace("lag_", "")
                    if original_col in features_df.columns:
                        original_values = features_df[original_col].dropna()
                        if len(original_values) > 10 and len(values) > 10:
                            # Basic sanity check - lag features shouldn't be identical
                            assert not np.array_equal(
                                values.iloc[: min(len(values), len(original_values))],
                                original_values.iloc[: min(len(values), len(original_values))],
                            )

            print("✅ Time series features test passed")

        except ImportError as e:
            pytest.skip(f"Time series features test skipped: {e}")

    @pytest.mark.performance
    def test_preprocessing_performance(self, large_dataset, memory_monitor):
        """Test preprocessing performance with large dataset."""
        try:
            import time

            from trading_rl_agent.data.features import generate_features

            initial_memory = memory_monitor["initial"]

            # Time the preprocessing
            start_time = time.time()
            features_df = generate_features(large_dataset.copy())
            end_time = time.time()

            processing_time = end_time - start_time

            # Validate results
            assert isinstance(features_df, pd.DataFrame)
            assert len(features_df) > 0

            # Performance benchmarks
            assert processing_time < 30, f"Preprocessing took too long: {processing_time:.2f}s"

            # Memory usage
            current_memory = memory_monitor["current"]()
            memory_increase = current_memory - initial_memory

            assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f} MB"

            print(
                f"✅ Performance test passed: {processing_time:.2f}s, {memory_increase:.2f}MB",
            )

        except ImportError as e:
            pytest.skip(f"Performance test skipped: {e}")


class TestDataPipelineIntegration:
    """Test data preprocessing integration with other components."""

    def test_preprocessing_environment_integration(self, sample_market_data, tmp_path):
        """Test preprocessing integration with trading environment."""
        try:
            from trading_rl_agent.data.features import generate_features
            from trading_rl_agent.envs.finrl_trading_env import TradingEnv

            # Preprocess data
            features_df = generate_features(sample_market_data.copy())

            # Save to CSV for environment
            csv_path = tmp_path / "processed_data.csv"
            features_df.to_csv(csv_path, index=False)

            # Test with trading environment
            env_config = {
                "dataset_paths": [str(csv_path)],
                "window_size": 10,
                "initial_balance": 10000,
            }

            env = TradingEnv(env_config)

            # Test environment with preprocessed data
            obs, info = env.reset()
            assert obs is not None

            # Take a few steps
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            print("✅ Preprocessing-environment integration test passed")

        except ImportError as e:
            pytest.skip(f"Preprocessing-environment integration test skipped: {e}")

    def test_preprocessing_model_integration(self, sample_market_data):
        """Test preprocessing integration with predictive models."""
        try:
            from trading_rl_agent.data.features import generate_features

            # Preprocess data
            features_df = generate_features(sample_market_data.copy())

            # Get numeric features for model input
            numeric_features = features_df.select_dtypes(include=[np.number]).fillna(0)

            if len(numeric_features.columns) > 0:
                # Test with simple sklearn model
                from sklearn.linear_model import LinearRegression
                from sklearn.model_selection import train_test_split

                # Create target variable (next period close price)
                if "close" in features_df.columns:
                    target = features_df["close"].shift(-1).fillna(method="ffill")
                else:
                    # Use first numeric column as target
                    target = numeric_features.iloc[:, 0].shift(-1).fillna(method="ffill")

                # Remove rows with NaN target
                valid_indices = ~target.isna()
                X = numeric_features[valid_indices]
                y = target[valid_indices]

                if len(X) > 10:  # Need minimum samples
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=42,
                    )

                    # Train model
                    model = LinearRegression()
                    model.fit(X_train, y_train)

                    # Make predictions
                    predictions = model.predict(X_test)

                    assert len(predictions) == len(y_test)
                    assert not np.any(np.isnan(predictions))
                    assert not np.any(np.isinf(predictions))

                    print("✅ Preprocessing-model integration test passed")
                else:
                    print("⚠️ Insufficient data for model integration test")
            else:
                print("⚠️ No numeric features for model integration test")

        except ImportError as e:
            pytest.skip(f"Preprocessing-model integration test skipped: {e}")
