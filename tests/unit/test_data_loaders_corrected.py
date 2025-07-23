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

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import requests

from trade_agent.data.loaders.alphavantage_loader import load_alphavantage

# from freezegun import freeze_time  # Not used in current tests
from trade_agent.data.loaders.yfinance_loader import load_yfinance
from trade_agent.data.synthetic import fetch_synthetic_data, generate_gbm_prices


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
        """Test graceful failure when yfinance is not installed."""
        with (
            patch(
                "trade_agent.data.loaders.yfinance_loader.yf",
                None,
            ),
            pytest.raises(ImportError, match="yfinance package is required"),
        ):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

    def test_load_yfinance_network_error(self):
        """Test handling of network errors during yfinance data loading."""
        with patch("yfinance.download", side_effect=Exception("Network error")), pytest.raises(
            Exception, match="Network error"
        ):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

    @pytest.mark.benchmark
    def test_load_yfinance_performance(self, benchmark):
        """Benchmark the performance of the yfinance data loader."""
        mock_data = pd.DataFrame(
            {
                "Open": np.random.rand(1000) * 100,
                "High": np.random.rand(1000) * 100 + 1,
                "Low": np.random.rand(1000) * 100 - 1,
                "Close": np.random.rand(1000) * 100,
                "Volume": np.random.randint(100000, 1000000, 1000),
            },
            index=pd.to_datetime(pd.date_range(start="2020-01-01", periods=1000)),
        )

        def run_load_yfinance():
            with patch("yfinance.download", return_value=mock_data):
                load_yfinance("AAPL", "2020-01-01", "2022-09-01", "day")

        benchmark(run_load_yfinance)


class TestAlphaVantageLoader:
    """Tests for the Alpha Vantage data loader."""

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "test_key"}, clear=True)
    def test_load_alphavantage_success(self):
        """Verify successful data loading from Alpha Vantage."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_df = pd.DataFrame(
                {
                    "1. open": [150.0],
                    "2. high": [155.0],
                    "3. low": [148.0],
                    "4. close": [152.0],
                    "5. volume": [12000000],
                },
                index=pd.to_datetime(["2023-01-01"]),
            )
            mock_timeseries.return_value.get_daily.return_value = (
                mock_df,
                {"Time Zone": "US/Eastern"},
            )
            df = load_alphavantage("IBM", "2023-01-01", "2023-01-01", interval="day")
            assert not df.empty
            assert df.iloc[0]["open"] == 150.0

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": ""}, clear=True)
    def test_load_alphavantage_missing_api_key(self):
        """Ensure the function raises an error if the Alpha Vantage API key is missing."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_timeseries.return_value.get_intraday.side_effect = ValueError("The Alpha Vantage API key must be provided")
            with pytest.raises(ValueError, match="API key is required"):
                load_alphavantage("IBM", "2023-01-01", "2023-01-02", "1min")

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "fake_key"}, clear=True)
    def test_load_alphavantage_api_key_from_env(self):
        """Check that the Alpha Vantage API key is read from environment variables."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_data = pd.DataFrame(
                {
                    "1. open": [150.0],
                    "2. high": [155.0],
                    "3. low": [148.0],
                    "4. close": [152.0],
                    "5. volume": [12000000],
                },
                index=pd.to_datetime(["2023-01-01"]),
            )
            mock_timeseries.return_value.get_daily.return_value = (
                mock_data,
                {"Time Zone": "US/Eastern"},
            )
            load_alphavantage("IBM", "2023-01-01", "2023-01-01", interval="day")
            mock_timeseries.assert_called_with(key="fake_key", output_format="pandas")

    def test_load_alphavantage_api_key_parameter(self):
        """Ensure the Alpha Vantage API key can be passed as a parameter."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_data = pd.DataFrame(
                {
                    "1. open": [150.0],
                    "2. high": [155.0],
                    "3. low": [148.0],
                    "4. close": [152.0],
                    "5. volume": [12000000],
                },
                index=pd.to_datetime(["2023-01-01"]),
            )
            mock_timeseries.return_value.get_daily.return_value = (
                mock_data,
                {"Time Zone": "US/Eastern"},
            )
            load_alphavantage("IBM", "2023-01-01", "2023-01-01", interval="day", api_key="param_key")
            mock_timeseries.assert_called_with(key="param_key", output_format="pandas")

    @patch("alpha_vantage.timeseries.TimeSeries.get_intraday", side_effect=ValueError)
    def test_load_alphavantage_invalid_symbol(self, _mock_get_intraday):
        """Test error handling for invalid symbols with Alpha Vantage."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries"), pytest.raises(ValueError):
            load_alphavantage("INVALID_SYMBOL", api_key="test_key")

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "test_key"}, clear=True)
    def test_load_alphavantage_no_data(self):
        """Ensure loading from Alpha Vantage with no data returns an empty DataFrame."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            # Mock the API call to return an empty DataFrame
            mock_timeseries.return_value.get_intraday.return_value = (
                pd.DataFrame(),
                {},
            )
            df = load_alphavantage("IBM", "2023-01-01", "2023-01-02", "1min")
            assert df.empty

    @patch.dict("os.environ", {"ALPHAVANTAGE_API_KEY": "test_key"}, clear=True)
    def test_load_alphavantage_network_error(self):
        """Test handling of network errors during Alpha Vantage data loading."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            # Configure the mock to raise a network-related error
            mock_timeseries.side_effect = requests.exceptions.RequestException("Network error")
            with pytest.raises(requests.exceptions.RequestException, match="Network error"):
                load_alphavantage("IBM", "2023-01-01", "2023-01-02", "1min")


class TestSyntheticData:
    """Tests for synthetic data generation."""

    def test_generate_gbm_prices_basic(self):
        """Verify the basic functionality of GBM price generation."""
        df = generate_gbm_prices()
        assert not df.empty
        assert "close" in df.columns
        assert len(df) == 252

    def test_generate_gbm_prices_parameters(self):
        """Check GBM price generation with custom parameters."""
        df = generate_gbm_prices(n_days=100, s0=50, mu=0.02, sigma=0.05)
        assert len(df) == 100
        assert df["close"].iloc[0] > 40  # Check if the starting price is reasonable

    def test_generate_gbm_prices_price_relationships(self):
        """Ensure that price relationships (high >= low) are consistent."""
        df = generate_gbm_prices(n_days=50)
        assert (df["high"] >= df["low"]).all()
        assert (df["open"] >= df["low"]).all()
        assert (df["close"] <= df["high"]).all()

    def test_generate_gbm_prices_volume_correlation(self):
        """Test the correlation between volume and price movement."""
        df = generate_gbm_prices(n_days=100)
        price_change = (df["close"] - df["open"]).abs()
        # A simple check for correlation; not a strict statistical test
        assert df["volume"][price_change > price_change.median()].mean() > df["volume"][price_change <= price_change.median()].mean()

    def test_generate_gbm_prices_timestamp_sequence(self):
        """Confirm that timestamps are sequential and correctly spaced."""
        df = generate_gbm_prices(n_days=30)
        assert df["timestamp"].is_monotonic_increasing
        assert (df["timestamp"].diff().dropna() == pd.Timedelta(days=1)).all()

    @pytest.mark.benchmark
    def test_gbm_performance(self, benchmark):
        """Benchmark the performance of GBM price generation."""

        def run_generate_gbm():
            generate_gbm_prices(n_days=1000)

        benchmark(run_generate_gbm)

    def test_fetch_synthetic_data_basic(self):
        """Verify the basic functionality of synthetic data fetching."""
        df = fetch_synthetic_data()
        assert not df.empty
        assert "close" in df.columns

    def test_fetch_synthetic_data_timeframes(self):
        """Check synthetic data generation for different timeframes."""
        for timeframe in ["day", "hour", "minute"]:
            df = fetch_synthetic_data(timeframe=timeframe)
            assert not df.empty

    def test_fetch_synthetic_data_volatility(self):
        """Test synthetic data generation with varying volatility."""
        low_vol_df = fetch_synthetic_data(volatility=0.005)
        high_vol_df = fetch_synthetic_data(volatility=0.02)
        assert low_vol_df["close"].std() < high_vol_df["close"].std()

    def test_fetch_synthetic_data_price_relationships(self):
        """Ensure consistent price relationships in synthetic data."""
        df = fetch_synthetic_data(n_samples=200)
        assert (df["high"] >= df["low"]).all()

    def test_fetch_synthetic_data_volume_positive(self):
        """Confirm that volume is always a positive integer."""
        df = fetch_synthetic_data(n_samples=100)
        assert (df["volume"] > 0).all()
        assert df["volume"].dtype == "int64"

    def test_fetch_synthetic_data_edge_cases(self):
        """Test edge cases for synthetic data generation."""
        with pytest.raises(ValueError):
            fetch_synthetic_data(n_samples=0)
        with pytest.raises(ValueError):
            fetch_synthetic_data(n_samples=-5)

    @pytest.mark.benchmark
    def test_synthetic_data_performance(self, benchmark):
        """Benchmark the performance of synthetic data generation."""

        def run_fetch_synthetic():
            fetch_synthetic_data(n_samples=1000)

        benchmark(run_fetch_synthetic)


class TestDataLoaderIntegration:
    """Test suite for data loader integrations."""

    def test_multiple_data_sources_consistency(self):
        """Test that data from different sources is loaded and handled consistently."""
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_av_df = pd.DataFrame(
                {
                    "1. open": [150.0],
                    "2. high": [155.0],
                    "3. low": [148.0],
                    "4. close": [152.0],
                    "5. volume": [12000000],
                },
                index=[pd.to_datetime("2023-01-01 10:00:00-05:00")],
            )
            mock_timeseries.return_value.get_intraday.return_value = (
                mock_av_df,
                {"6. Time Zone": "US/Eastern"},
            )

            with patch("yfinance.download") as mock_yf_download:
                mock_yf_df = pd.DataFrame(
                    {
                        "Open": [150.0],
                        "High": [152.0],
                        "Low": [149.0],
                        "Close": [151.0],
                        "Volume": [1000000],
                    },
                    index=pd.to_datetime(["2023-01-01 10:00:00"]),
                )
                mock_yf_download.return_value = mock_yf_df

                # Load data from both sources
                alphavantage_data = load_alphavantage("IBM", interval="1min", api_key="test")
                yfinance_data = load_yfinance("IBM", "2023-01-01", "2023-01-02", "minute")

                # Check that both DataFrames have the same columns
                assert all(alphavantage_data.columns == yfinance_data.columns)

    def test_error_handling_consistency(self):
        """Test consistent error handling across different data loaders."""
        # Test yfinance with an invalid symbol
        with patch("yfinance.download", side_effect=Exception("Network error")), pytest.raises(
            Exception, match="Network error"
        ):
            load_yfinance("AAPL", "2024-01-01", "2024-01-02", "day")

        # Test Alpha Vantage with an invalid symbol
        with patch("trade_agent.data.loaders.alphavantage_loader.TimeSeries") as mock_timeseries:
            mock_timeseries.return_value.get_intraday.side_effect = ValueError("Invalid API call")
            with pytest.raises(ValueError, match="Invalid API call"):
                load_alphavantage("IBM", "2023-01-01", "2023-01-02", "1min", api_key="test")

    def test_data_quality_validation(self):
        """Test data quality validation across different data sources."""
        # Create a sample DataFrame with potential quality issues
        bad_data = pd.DataFrame(
            {
                "Open": [100.0, np.nan, 102.0],
                "High": [105.0, 106.0, 101.0],  # high < low
                "Low": [98.0, 104.0, 103.0],
                "Close": [102.0, 105.0, np.inf],  # infinite value
                "Volume": [100000, -500, 120000],  # negative volume
            },
            index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        )

        with patch("yfinance.download", return_value=bad_data):
            # The loader should either clean the data or raise an error
            # In this case, we expect it to be handled gracefully
            df = load_yfinance("AAPL", "2023-01-01", "2023-01-03", "day")
            assert not df.empty  # Or check for specific data cleaning logic

    def test_large_dataset_handling(self):
        """Test handling of large datasets from different sources."""
        with patch("yfinance.download") as mock_download:
            large_df = pd.DataFrame(
                np.random.rand(10000, 5),
                columns=["Open", "High", "Low", "Close", "Volume"],
            )
            mock_download.return_value = large_df

            df = load_yfinance("AAPL", "2000-01-01", "2023-01-01", "day")
            assert len(df) == 10000

    @pytest.mark.parametrize(
        ("loader", "params"),
        [
            (
                load_yfinance,
                {"symbol": "AAPL", "start": "2023-01-01", "end": "2023-01-05"},
            ),
            (
                fetch_synthetic_data,
                {"n_samples": 1000, "timeframe": "day"},
            ),
        ],
    )
    def test_data_loader_performance_comparison(self, benchmark, loader, params):
        """Compare the performance of different data loaders."""
        with patch("yfinance.download", return_value=pd.DataFrame(np.random.rand(1000, 5), columns=["Open", "High", "Low", "Close", "Volume"])):
            benchmark(loader, **params)
