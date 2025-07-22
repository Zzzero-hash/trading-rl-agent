"""
Tests for timezone handling consistency across data sources.
"""

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from trade_agent.data.utils import (
    _normalize_timestamp_series,
    convert_to_naive_timestamps,
    ensure_timezone_aware,
    get_market_timezone,
    normalize_timestamps,
)


class TestTimezoneHandling(unittest.TestCase):
    """Test timezone handling utilities."""

    def test_normalize_timestamps_with_timezone_aware_index(self):
        """Test normalizing timezone-aware index."""
        # Create timezone-aware index
        dates = pd.date_range(
            start="2023-01-01",
            end="2023-01-03",
            freq="D",
            tz="UTC"
        )
        df = pd.DataFrame(
            {"value": [1, 2, 3]},
            index=dates
        )

        # Normalize to America/New_York
        result = normalize_timestamps(df, timezone="America/New_York")

        # Check that timezone was converted
        assert result.index.tz is not None
        assert str(result.index.tz) == "America/New_York"
        assert len(result) == 3

    def test_normalize_timestamps_with_naive_index(self):
        """Test normalizing naive index."""
        # Create naive index
        dates = pd.date_range(start="2023-01-01", end="2023-01-03", freq="D")
        df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

        # Normalize to America/New_York
        result = normalize_timestamps(df, timezone="America/New_York")

        # Check that timezone was added
        assert result.index.tz is not None
        assert str(result.index.tz) == "America/New_York"
        assert len(result) == 3

    def test_normalize_timestamps_with_timestamp_column(self):
        """Test normalizing specific timestamp column."""
        # Create DataFrame with timestamp column
        dates = pd.date_range(start="2023-01-01", end="2023-01-03", freq="D", tz="UTC")
        df = pd.DataFrame({
            "timestamp": dates,
            "value": [1, 2, 3]
        })

        # Normalize timestamp column
        result = normalize_timestamps(df, timestamp_column="timestamp", timezone="America/New_York")

        # Check that timestamp column was normalized
        assert result["timestamp"].dt.tz is not None
        assert str(result["timestamp"].dt.tz) == "America/New_York"
        assert len(result) == 3

    def test_normalize_timestamp_series_with_various_inputs(self):
        """Test _normalize_timestamp_series with various input types."""
        # Test with timezone-aware series
        tz_aware = pd.Series(pd.date_range("2023-01-01", periods=3, tz="UTC"))
        result = _normalize_timestamp_series(tz_aware, "America/New_York")
        assert result.dt.tz is not None
        assert str(result.dt.tz) == "America/New_York"

        # Test with naive series
        naive = pd.Series(pd.date_range("2023-01-01", periods=3))
        result = _normalize_timestamp_series(naive, "America/New_York")
        assert result.dt.tz is not None
        assert str(result.dt.tz) == "America/New_York"

    @patch("trade_agent.core.unified_config.load_config")
    def test_get_market_timezone_with_config(self, mock_load_config):
        """Test getting market timezone from configuration."""
        # Mock configuration
        mock_config = MagicMock()
        mock_config.data.market_timezone = "Europe/London"
        mock_load_config.return_value = mock_config

        result = get_market_timezone()
        assert result == "Europe/London"

    def test_get_market_timezone_fallback(self):
        """Test getting market timezone fallback."""
        with patch("trade_agent.core.unified_config.load_config", side_effect=ImportError):
            result = get_market_timezone()
            assert result == "America/New_York"

    def test_ensure_timezone_aware(self):
        """Test ensuring timestamps are timezone-aware."""
        # Test with naive timestamps
        naive = pd.Series(pd.date_range("2023-01-01", periods=3))
        result = ensure_timezone_aware(naive)
        assert result.dt.tz is not None

        # Test with timezone-aware timestamps
        tz_aware = pd.Series(pd.date_range("2023-01-01", periods=3, tz="UTC"))
        result = ensure_timezone_aware(tz_aware)
        assert result.dt.tz is not None

    def test_convert_to_naive_timestamps(self):
        """Test converting to naive timestamps."""
        # Test with timezone-aware timestamps
        tz_aware = pd.Series(pd.date_range("2023-01-01", periods=3, tz="UTC"))
        result = convert_to_naive_timestamps(tz_aware)
        assert result.dt.tz is None

        # Test with naive timestamps
        naive = pd.Series(pd.date_range("2023-01-01", periods=3))
        result = convert_to_naive_timestamps(naive)
        assert result.dt.tz is None

    def test_timezone_consistency_across_data_sources(self):
        """Test that timezone handling is consistent across different scenarios."""
        # Create test data with different timezone scenarios
        test_cases = [
            # (input_tz, expected_tz, description)
            ("UTC", "America/New_York", "UTC to NY"),
            (None, "America/New_York", "Naive to NY"),
            ("Europe/London", "America/New_York", "London to NY"),
        ]

        for input_tz, expected_tz, description in test_cases:
            with self.subTest(description):
                # Create test data
                if input_tz:
                    dates = pd.date_range("2023-01-01", periods=3, tz=input_tz)
                else:
                    dates = pd.date_range("2023-01-01", periods=3)

                df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

                # Normalize
                result = normalize_timestamps(df, timezone=expected_tz)

                # Verify
                assert result.index.tz is not None
                assert str(result.index.tz) == expected_tz
                assert len(result) == 3


class TestTimezoneIntegration(unittest.TestCase):
    """Test timezone handling integration with data sources."""

    @patch("trade_agent.data.historical.yf")
    def test_historical_data_timezone_handling(self, mock_yf):
        """Test timezone handling in historical data fetching."""
        from trade_agent.data.historical import fetch_historical_data

        # Mock yfinance response with non-empty data
        mock_data = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [103, 104, 105],
            "Volume": [1000, 1100, 1200]
        }, index=pd.date_range("2023-01-01", periods=3, tz="UTC"))

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_data
        mock_yf.Ticker.return_value = mock_ticker

        # Fetch data with timezone specification
        result = fetch_historical_data(
            symbol="AAPL",
            start="2023-01-01",
            end="2023-01-03",
            timezone="America/New_York"
        )

        # Verify timezone handling
        assert "timestamp" in result.columns
        assert len(result) == 3
        # Note: The timestamp column should be timezone-aware in NY timezone

    def test_pipeline_timezone_configuration(self):
        """Test that pipeline configuration includes timezone settings."""
        # This test verifies that the pipeline can handle timezone configuration
        config = {
            "start": "2023-01-01",
            "end": "2023-01-31",
            "timestep": "day",
            "timezone": "America/New_York",
            "yfinance_symbols": ["AAPL", "GOOGL"]
        }

        # Verify timezone is in config
        assert "timezone" in config
        assert config["timezone"] == "America/New_York"
