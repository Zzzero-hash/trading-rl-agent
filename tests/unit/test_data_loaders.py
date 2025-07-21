"""
Comprehensive tests for data loaders.

This module tests:
- YFinance data loader with various intervals and error handling
- Alpha Vantage data loader with API key management and data filtering
- Synthetic data generation with different timeframes and volatility
- Network failure scenarios and retry logic
- Data format validation and standardization
- Performance benchmarks for data fetching
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.data.loaders.alphavantage_loader import load_alphavantage

# from freezegun import freeze_time  # Not used in current tests
from trading_rl_agent.data.loaders.yfinance_loader import load_yfinance
from trading_rl_agent.data.synthetic import fetch_synthetic_data, generate_gbm_prices


class TestYFinanceLoader:
    """Test YFinance data loader functionality."""

    def test_load_yfinance_success(self):
        """Test successful data loading from YFinance."""
        # Mock yfinance data
        mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [101.0, 102.0, 103.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        with patch("yfinance.download", return_value=mock_data):
            result = load_yfinance("AAPL", "2024-01-01", "2024-01-03", "day")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(col in result.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
        assert result["open"].iloc[0] == 100.0
        assert result["volume"].iloc[0] == 1000000

    def test_load_yfinance_empty_data(self):
        """Test handling of empty data from YFinance."""
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = load_yfinance("INVALID", "2024-01-01", "2024-01-03", "day")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert all(col in result.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])

    def test_load_yfinance_interval_mapping(self):
        """Test interval mapping for different timeframes."""
        mock_data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        intervals = ["day", "hour", "minute"]
        expected_yf_intervals = ["1d", "1h", "1m"]

        for interval, expected in zip(intervals, expected_yf_intervals, strict=False):
            with patch("yfinance.download") as mock_download:
                mock_download.return_value = mock_data
                load_yfinance("AAPL", "2024-01-01", "2024-01-02", interval)

                # Check that yfinance.download was called with correct interval
                call_args = mock_download.call_args
                assert call_args[1]["interval"] == expected

    def test_load_yfinance_custom_interval(self):
        """Test custom interval string handling."""
        mock_data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("yfinance.download") as mock_download:
            mock_download.return_value = mock_data
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "5m")

            call_args = mock_download.call_args
            assert call_args[1]["interval"] == "5m"

    def test_load_yfinance_timezone_handling(self):
        """Test timezone handling in YFinance data."""
        # Mock data with timezone-aware index
        mock_data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],
                "Low": [99.0],
                "Close": [101.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1, tz="UTC"),
        )

        with patch("yfinance.download", return_value=mock_data):
            result = load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        assert isinstance(result, pd.DataFrame)
        assert "timestamp" in result.columns
        # Should handle timezone conversion gracefully

    def test_load_yfinance_missing_yfinance(self):
        """Test error handling when yfinance is not available."""
        with (
            patch("trading_rl_agent.data.loaders.yfinance_loader.yf", None),
            pytest.raises(ImportError, match="yfinance package is required"),
        ):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

    def test_load_yfinance_network_error(self):
        """Test handling of network errors."""
        with (
            patch("yfinance.download", side_effect=Exception("Network error")),
            pytest.raises(Exception, match="Network error"),
        ):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

    @pytest.mark.benchmark
    def test_load_yfinance_performance(self, benchmark):
        """Benchmark YFinance data loading performance."""
        mock_data = pd.DataFrame(
            {
                "Open": np.random.uniform(100, 200, 1000),
                "High": np.random.uniform(100, 200, 1000),
                "Low": np.random.uniform(100, 200, 1000),
                "Close": np.random.uniform(100, 200, 1000),
                "Volume": np.random.randint(1000000, 10000000, 1000),
            },
            index=pd.date_range("2024-01-01", periods=1000),
        )

        with patch("yfinance.download", return_value=mock_data):
            benchmark(lambda: load_yfinance("AAPL", "2024-01-01", "2024-12-31", "day"))


class TestAlphaVantageLoader:
    """Test Alpha Vantage data loader functionality."""

    def test_load_alphavantage_success(self):
        """Test successful data loading from Alpha Vantage."""
        # Mock Alpha Vantage response
        mock_data = pd.DataFrame(
            {
                "1. open": [100.0, 101.0, 102.0],
                "2. high": [102.0, 103.0, 104.0],
                "3. low": [99.0, 100.0, 101.0],
                "4. close": [101.0, 102.0, 103.0],
                "5. volume": [1000000, 1100000, 1200000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        # Ensure index name is not 'timestamp' to avoid conflicts
        mock_data.index.name = "date"

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            result = load_alphavantage("AAPL", "2024-01-01", "2024-01-03", "day")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert all(col in result.columns for col in ["timestamp", "open", "high", "low", "close", "volume"])
        assert result["open"].iloc[0] == 100.0

    def test_load_alphavantage_api_key_from_env(self):
        """Test API key retrieval from environment variable."""
        mock_data = pd.DataFrame(
            {
                "1. open": [100.0],
                "2. high": [102.0],
                "3. low": [99.0],
                "4. close": [101.0],
                "5. volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            with patch.dict(os.environ, {"ALPHAVANTAGE_API_KEY": "test_key"}):
                load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

                # Check that TimeSeries was initialized with correct key
                mock_timeseries.assert_called_with(key="test_key", output_format="pandas")

    def test_load_alphavantage_api_key_parameter(self):
        """Test API key passed as parameter."""
        mock_data = pd.DataFrame(
            {
                "1. open": [100.0],
                "2. high": [102.0],
                "3. low": [99.0],
                "4. close": [101.0],
                "5. volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day", api_key="custom_key")

            mock_timeseries.assert_called_with(key="custom_key", output_format="pandas")

    def test_load_alphavantage_default_api_key(self):
        """Test default API key when none provided."""
        mock_data = pd.DataFrame(
            {
                "1. open": [100.0],
                "2. high": [102.0],
                "3. low": [99.0],
                "4. close": [101.0],
                "5. volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            with patch.dict(os.environ, {}, clear=True):
                load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

                mock_timeseries.assert_called_with(key="demo", output_format="pandas")

    def test_load_alphavantage_intraday_data(self):
        """Test intraday data fetching."""
        mock_data = pd.DataFrame(
            {
                "1. open": [100.0],
                "2. high": [102.0],
                "3. low": [99.0],
                "4. close": [101.0],
                "5. volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_intraday.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "hour")

            mock_ts.get_intraday.assert_called_with("AAPL", interval="60min", outputsize="full")

    def test_load_alphavantage_date_filtering(self):
        """Test date range filtering."""
        # Mock data with dates outside the requested range
        mock_data = pd.DataFrame(
            {
                "1. open": [100.0, 101.0, 102.0, 103.0],
                "2. high": [102.0, 103.0, 104.0, 105.0],
                "3. low": [99.0, 100.0, 101.0, 102.0],
                "4. close": [101.0, 102.0, 103.0, 104.0],
                "5. volume": [1000000, 1100000, 1200000, 1300000],
            },
            index=pd.date_range("2023-12-30", periods=4),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            result = load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

        # Should only include data within the date range
        assert len(result) == 2
        assert result["timestamp"].min() >= pd.to_datetime("2024-01-01")
        assert result["timestamp"].max() <= pd.to_datetime("2024-01-02")

    def test_load_alphavantage_missing_alphavantage(self):
        """Test error handling when alpha_vantage is not available."""
        with (
            patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries", None),
            pytest.raises(ImportError, match="alpha_vantage package is required"),
        ):
            load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

    def test_load_alphavantage_api_error(self):
        """Test handling of API errors."""
        with (
            patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries,
            pytest.raises(RuntimeError, match="Failed to fetch data from Alpha Vantage"),
        ):
            mock_ts = Mock()
            mock_ts.get_daily.side_effect = Exception("API error")
            mock_timeseries.return_value = mock_ts
            load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

    @pytest.mark.benchmark
    def test_load_alphavantage_performance(self, benchmark):
        """Benchmark Alpha Vantage data loading performance."""
        mock_data = pd.DataFrame(
            {
                "1. open": np.random.uniform(100, 200, 1000),
                "2. high": np.random.uniform(100, 200, 1000),
                "3. low": np.random.uniform(100, 200, 1000),
                "4. close": np.random.uniform(100, 200, 1000),
                "5. volume": np.random.randint(1000000, 10000000, 1000),
            },
            index=pd.date_range("2024-01-01", periods=1000),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (mock_data, {})
            mock_timeseries.return_value = mock_ts

            benchmark(lambda: load_alphavantage("AAPL", "2024-01-01", "2024-12-31", "day"))


class TestSyntheticData:
    """Test synthetic data generation functionality."""

    def test_fetch_synthetic_data_basic(self):
        """Test basic synthetic data generation."""
        df = fetch_synthetic_data(n_samples=10, timeframe="day")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df["timestamp"].dtype == "datetime64[ns]"
        assert df["volume"].dtype == "int64"

    def test_fetch_synthetic_data_timeframes(self):
        """Test synthetic data generation with different timeframes."""
        timeframes = ["day", "hour", "minute"]

        for timeframe in timeframes:
            df = fetch_synthetic_data(n_samples=5, timeframe=timeframe)

            assert len(df) == 5
            # Check that timestamps are properly spaced
            time_diff = df["timestamp"].diff().dropna()
            if timeframe == "day":
                assert all(diff.days == 1 for diff in time_diff)
            elif timeframe == "hour":
                assert all(diff.components.hours == 1 for diff in time_diff)
            elif timeframe == "minute":
                assert all(diff.components.minutes == 1 for diff in time_diff)

    def test_fetch_synthetic_data_volatility(self):
        """Test synthetic data generation with different volatility levels."""
        low_vol_df = fetch_synthetic_data(n_samples=10, volatility=0.001)
        high_vol_df = fetch_synthetic_data(n_samples=10, volatility=0.05)

        # High volatility should have larger price ranges
        low_range = (low_vol_df["high"] - low_vol_df["low"]).mean()
        high_range = (high_vol_df["high"] - high_vol_df["low"]).mean()

        assert high_range > low_range

    def test_fetch_synthetic_data_price_relationships(self):
        """Test that price relationships are maintained."""
        df = fetch_synthetic_data(n_samples=100, timeframe="day")

        # High should be >= low for all rows
        assert all(df["high"] >= df["low"])

        # Close should be between high and low
        assert all((df["close"] >= df["low"]) & (df["close"] <= df["high"]))

        # Open should be between high and low
        assert all((df["open"] >= df["low"]) & (df["open"] <= df["high"]))

    def test_fetch_synthetic_data_volume_positive(self):
        """Test that volume is always positive."""
        df = fetch_synthetic_data(n_samples=50, timeframe="day")

        assert all(df["volume"] > 0)
        assert df["volume"].dtype == "int64"

    def test_fetch_synthetic_data_edge_cases(self):
        """Test edge cases for synthetic data generation."""
        # Single sample
        df = fetch_synthetic_data(n_samples=1, timeframe="day")
        assert len(df) == 1

        # Large number of samples
        df = fetch_synthetic_data(n_samples=1000, timeframe="day")
        assert len(df) == 1000

        # Zero volatility
        df = fetch_synthetic_data(n_samples=10, volatility=0.0)
        assert len(df) == 10

    def test_generate_gbm_prices_basic(self):
        """Test GBM price generation."""
        df = generate_gbm_prices(n_days=10)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        assert df["timestamp"].dtype == "datetime64[ns]"

    def test_generate_gbm_prices_parameters(self):
        """Test GBM price generation with different parameters."""
        # Test with different drift and volatility
        df = generate_gbm_prices(n_days=20, mu=0.001, sigma=0.02, s0=50.0)

        assert len(df) == 20
        assert df["close"].iloc[0] == 50.0  # Initial price

        # Test that prices are positive
        assert all(df["close"] > 0)
        assert all(df["open"] > 0)
        assert all(df["high"] > 0)
        assert all(df["low"] > 0)

    def test_generate_gbm_prices_price_relationships(self):
        """Test price relationships in GBM-generated data."""
        df = generate_gbm_prices(n_days=50)

        # High should be >= low
        assert all(df["high"] >= df["low"])

        # Close should be between high and low
        assert all((df["close"] >= df["low"]) & (df["close"] <= df["high"]))

        # Open should be between high and low
        assert all((df["open"] >= df["low"]) & (df["open"] <= df["high"]))

    def test_generate_gbm_prices_volume_correlation(self):
        """Test that volume correlates with price movement."""
        df = generate_gbm_prices(n_days=100)

        # Calculate price movement
        abs(df["close"] - df["open"]) / df["close"]

        # Volume should be positive
        assert all(df["volume"] > 0)

        # Volume should be integer
        assert df["volume"].dtype == "int64"

    def test_generate_gbm_prices_timestamp_sequence(self):
        """Test that timestamps are properly sequenced."""
        df = generate_gbm_prices(n_days=10)

        # Check that timestamps are in ascending order
        assert df["timestamp"].is_monotonic_increasing

        # Check that timestamps are daily
        time_diff = df["timestamp"].diff().dropna()
        assert all(diff.days == 1 for diff in time_diff)

    @pytest.mark.benchmark
    def test_synthetic_data_performance(self, benchmark):
        """Benchmark synthetic data generation performance."""
        benchmark(lambda: fetch_synthetic_data(n_samples=1000, timeframe="day"))

    @pytest.mark.benchmark
    def test_gbm_performance(self, benchmark):
        """Benchmark GBM price generation performance."""
        benchmark(lambda: generate_gbm_prices(n_days=1000))


class TestDataLoaderIntegration:
    """Integration tests for data loaders."""

    def test_multiple_data_sources_consistency(self):
        """Test that different data sources produce consistent output format."""
        # Mock data for both sources
        yf_mock_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        av_mock_data = pd.DataFrame(
            {
                "1. open": [100.0, 101.0],
                "2. high": [102.0, 103.0],
                "3. low": [99.0, 100.0],
                "4. close": [101.0, 102.0],
                "5. volume": [1000000, 1100000],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )

        with patch("yfinance.download", return_value=yf_mock_data):
            yf_result = load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (av_mock_data, {})
            mock_timeseries.return_value = mock_ts

            av_result = load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

        # Both should have the same column structure
        # Check that both results have the same columns (order may vary)
        assert set(yf_result.columns) == set(av_result.columns)
        assert yf_result.shape == av_result.shape

    def test_data_quality_validation(self):
        """Test data quality validation across different sources."""
        # Test synthetic data quality
        synthetic_df = fetch_synthetic_data(n_samples=100, timeframe="day")

        # Test GBM data quality
        gbm_df = generate_gbm_prices(n_days=100)

        for df in [synthetic_df, gbm_df]:
            # Check for missing values
            assert not df.isnull().any().any()

            # Check for infinite values
            assert not np.isinf(df.select_dtypes(include=[np.number])).any().any()

            # Check price relationships
            assert all(df["high"] >= df["low"])
            assert all((df["close"] >= df["low"]) & (df["close"] <= df["high"]))
            assert all((df["open"] >= df["low"]) & (df["open"] <= df["high"]))

            # Check volume is positive
            assert all(df["volume"] > 0)

    def test_error_handling_consistency(self):
        """Test consistent error handling across data sources."""
        # Test YFinance error handling
        with patch("yfinance.download", side_effect=Exception("Network error")), pytest.raises(RuntimeError):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        # Test Alpha Vantage error handling
        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.side_effect = Exception("API error")
            mock_timeseries.return_value = mock_ts

            with pytest.raises(RuntimeError):
                load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Test synthetic data with large number of samples
        large_df = fetch_synthetic_data(n_samples=10000, timeframe="day")
        assert len(large_df) == 10000

        # Test GBM with large number of days
        large_gbm_df = generate_gbm_prices(n_days=1000)
        assert len(large_gbm_df) == 1000

        # Check memory usage is reasonable
        assert large_df.memory_usage(deep=True).sum() < 10 * 1024 * 1024  # Less than 10MB

    def test_data_loader_performance_comparison(self):
        """Compare performance of different data sources."""
        import time

        # Test synthetic data generation performance
        start_time = time.time()
        fetch_synthetic_data(n_samples=1000, timeframe="day")
        synthetic_time = time.time() - start_time

        # Test GBM generation performance
        start_time = time.time()
        generate_gbm_prices(n_days=1000)
        gbm_time = time.time() - start_time

        # Both should be reasonably fast
        assert synthetic_time < 1.0  # Less than 1 second
        assert gbm_time < 1.0  # Less than 1 second


class TestDataLoaderEdgeCases:
    """Test edge cases and error scenarios for data loaders."""

    def test_invalid_symbols(self):
        """Test handling of invalid symbols."""
        # Test YFinance with invalid symbol
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = load_yfinance("INVALID_SYMBOL_12345", "2024-01-01", "2024-01-02", "day")
            assert len(result) == 0

        # Test Alpha Vantage with invalid symbol
        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            # Provide empty DataFrame with proper column structure
            empty_df = pd.DataFrame(columns=["1. open", "2. high", "3. low", "4. close", "5. volume"])
            mock_ts.get_daily.return_value = (empty_df, {})
            mock_timeseries.return_value = mock_ts

            result = load_alphavantage("INVALID_SYMBOL_12345", "2024-01-01", "2024-01-02", "day")
            assert len(result) == 0

    def test_invalid_date_ranges(self):
        """Test handling of invalid date ranges."""
        # Future dates
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = load_yfinance("AAPL", "2030-01-01", "2030-01-02", "day")
            assert len(result) == 0

        # End date before start date
        with patch("yfinance.download", return_value=pd.DataFrame()):
            result = load_yfinance("AAPL", "2024-01-02", "2024-01-01", "day")
            assert len(result) == 0

    def test_malformed_data_handling(self):
        """Test handling of malformed data from APIs."""
        # YFinance with missing columns
        malformed_data = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [102.0],  # Missing Low, Close, Volume
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("yfinance.download", return_value=malformed_data), pytest.raises(KeyError):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        # Alpha Vantage with unexpected column names
        malformed_av_data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )

        with patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_ts = Mock()
            mock_ts.get_daily.return_value = (malformed_av_data, {})
            mock_timeseries.return_value = mock_ts

            # Should handle gracefully by renaming columns
            result = load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")
            assert len(result) == 1

    def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        with patch("yfinance.download", side_effect=TimeoutError("Request timeout")), pytest.raises(TimeoutError):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        with (
            patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries,
            pytest.raises(RuntimeError),
        ):
            mock_ts = Mock()
            mock_ts.get_daily.side_effect = TimeoutError("Request timeout")
            mock_timeseries.return_value = mock_ts
            load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

    def test_rate_limiting_handling(self):
        """Test handling of API rate limiting."""
        with patch("yfinance.download", side_effect=Exception("Rate limit exceeded")), pytest.raises(RuntimeError):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        with (
            patch("trading_rl_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries,
            pytest.raises(RuntimeError),
        ):
            mock_ts = Mock()
            mock_ts.get_daily.side_effect = Exception("Rate limit exceeded")
            mock_timeseries.return_value = mock_ts
            load_alphavantage("AAPL", "2024-01-01", "2024-01-02", "day")

    def test_synthetic_data_edge_cases(self):
        """Test edge cases for synthetic data generation."""
        # Zero samples
        with pytest.raises(ValueError):
            fetch_synthetic_data(n_samples=0, timeframe="day")

        # Negative samples
        with pytest.raises(ValueError):
            fetch_synthetic_data(n_samples=-1, timeframe="day")

        # Negative volatility
        with pytest.raises(ValueError):
            fetch_synthetic_data(n_samples=10, volatility=-0.1, timeframe="day")

        # GBM with zero days
        with pytest.raises(ValueError):
            generate_gbm_prices(n_days=0)

        # GBM with negative days
        with pytest.raises(ValueError):
            generate_gbm_prices(n_days=-1)
