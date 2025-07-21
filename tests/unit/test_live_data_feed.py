"""
Comprehensive tests for live data feed module.

This module tests the LiveDataFeed class and related functionality
with focus on small incremental fixes and edge cases.
"""

import asyncio
import contextlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from trade_agent.data.live_feed import LiveDataFeed


class TestLiveDataFeed:
    """Test suite for LiveDataFeed class."""

    def test_initialization(self):
        """Test LiveDataFeed initialization."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        feed = LiveDataFeed(symbols, data_source="yfinance")

        assert feed.symbols == symbols
        assert feed.data_source == "yfinance"
        assert feed.connected is False
        assert feed.update_interval == 60
        assert isinstance(feed.price_cache, dict)
        assert isinstance(feed.feature_cache, dict)
        assert isinstance(feed.last_update, dict)

    def test_initialization_with_single_symbol(self):
        """Test initialization with single symbol string."""
        feed = LiveDataFeed("AAPL", data_source="yfinance")
        assert feed.symbols == ["AAPL"]

    def test_initialization_with_empty_symbols(self):
        """Test initialization with empty symbols list."""
        with pytest.raises(ValueError):
            LiveDataFeed([], data_source="yfinance")

    def test_initialization_with_invalid_data_source(self):
        """Test initialization with invalid data source."""
        with pytest.raises(ValueError):
            LiveDataFeed(["AAPL"], data_source="invalid_source")

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection to data source."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        await feed.connect()
        assert feed.connected is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure handling."""
        feed = LiveDataFeed(["AAPL"], data_source="invalid_source")

        with pytest.raises(ValueError):
            await feed.connect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnection functionality."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")
        feed.connected = True

        await feed.disconnect()
        assert feed.connected is False

    @pytest.mark.asyncio
    async def test_fetch_latest_data_success(self):
        """Test successful data fetching."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock yfinance data
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        with patch.object(yf.Ticker, "history", return_value=mock_data):
            data = await feed.fetch_latest_data("AAPL")

            assert isinstance(data, pd.DataFrame)
            assert not data.empty
            assert "Open" in data.columns
            assert "High" in data.columns
            assert "Low" in data.columns
            assert "Close" in data.columns
            assert "Volume" in data.columns

    @pytest.mark.asyncio
    async def test_fetch_latest_data_failure(self):
        """Test data fetching failure handling."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with patch.object(yf.Ticker, "history", side_effect=Exception("Network error")), pytest.raises(RuntimeError):
            await feed.fetch_latest_data("AAPL")

    @pytest.mark.asyncio
    async def test_fetch_latest_data_empty_result(self):
        """Test handling of empty data results."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with patch.object(yf.Ticker, "history", return_value=pd.DataFrame()):
            data = await feed.fetch_latest_data("AAPL")
            assert data.empty

    @pytest.mark.asyncio
    async def test_update_all_symbols(self):
        """Test updating data for all symbols."""
        symbols = ["AAPL", "GOOGL"]
        feed = LiveDataFeed(symbols, data_source="yfinance")

        # Mock data for both symbols
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        with patch.object(yf.Ticker, "history", return_value=mock_data):
            await feed.update_all_symbols()

            # Check that data was cached for both symbols
            for symbol in symbols:
                assert symbol in feed.price_cache
                assert not feed.price_cache[symbol].empty
                assert symbol in feed.last_update

    @pytest.mark.asyncio
    async def test_update_all_symbols_partial_failure(self):
        """Test handling of partial failures during update."""
        symbols = ["AAPL", "INVALID_SYMBOL"]
        feed = LiveDataFeed(symbols, data_source="yfinance")

        # Mock successful data for AAPL, failure for INVALID_SYMBOL
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        def mock_history(*args, **_):
            if "AAPL" in str(args):
                return mock_data
            raise Exception("Symbol not found")

        with patch.object(yf.Ticker, "history", side_effect=mock_history):
            await feed.update_all_symbols()

            # AAPL should be updated, INVALID_SYMBOL should not
            assert "AAPL" in feed.price_cache
            assert "INVALID_SYMBOL" not in feed.price_cache

    def test_get_latest_price(self):
        """Test getting latest price for a symbol."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock cached data
        mock_data = pd.DataFrame(
            {"Close": [153.0, 154.0, 155.0]},
            index=[
                datetime.now() - timedelta(minutes=2),
                datetime.now() - timedelta(minutes=1),
                datetime.now(),
            ],
        )

        feed.price_cache["AAPL"] = mock_data

        price = feed.get_latest_price("AAPL")
        assert price == 155.0

    def test_get_latest_price_no_data(self):
        """Test getting latest price when no data is available."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with pytest.raises(ValueError):
            feed.get_latest_price("AAPL")

    def test_get_latest_price_invalid_symbol(self):
        """Test getting latest price for invalid symbol."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with pytest.raises(ValueError):
            feed.get_latest_price("INVALID_SYMBOL")

    def test_get_price_history(self):
        """Test getting price history for a symbol."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock cached data
        mock_data = pd.DataFrame(
            {"Close": [150.0, 151.0, 152.0, 153.0, 154.0]},
            index=pd.date_range(start="2023-01-01", periods=5, freq="D"),
        )

        feed.price_cache["AAPL"] = mock_data

        history = feed.get_price_history("AAPL", periods=3)
        assert len(history) == 3
        assert history.iloc[-1]["Close"] == 154.0

    def test_get_price_history_invalid_periods(self):
        """Test getting price history with invalid periods."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock cached data
        mock_data = pd.DataFrame(
            {"Close": [150.0, 151.0, 152.0]},
            index=pd.date_range(start="2023-01-01", periods=3, freq="D"),
        )

        feed.price_cache["AAPL"] = mock_data

        # Request more periods than available
        history = feed.get_price_history("AAPL", periods=10)
        assert len(history) == 3  # Should return all available data

    def test_calculate_features(self):
        """Test feature calculation functionality."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock cached data
        mock_data = pd.DataFrame(
            {
                "Open": [150.0, 151.0, 152.0, 153.0, 154.0],
                "High": [155.0, 156.0, 157.0, 158.0, 159.0],
                "Low": [149.0, 150.0, 151.0, 152.0, 153.0],
                "Close": [151.0, 152.0, 153.0, 154.0, 155.0],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            },
            index=pd.date_range(start="2023-01-01", periods=5, freq="D"),
        )

        feed.price_cache["AAPL"] = mock_data

        features = feed.calculate_features("AAPL")
        assert isinstance(features, np.ndarray)
        assert features.shape[0] > 0  # Should have some features

    def test_calculate_features_no_data(self):
        """Test feature calculation when no data is available."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with pytest.raises(ValueError):
            feed.calculate_features("AAPL")

    def test_get_cached_features(self):
        """Test getting cached features."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock cached features
        mock_features = np.random.randn(10, 5)
        feed.feature_cache["AAPL"] = mock_features
        feed.last_update["AAPL"] = datetime.now()

        features = feed.get_cached_features("AAPL")
        assert np.array_equal(features, mock_features)

    def test_get_cached_features_expired(self):
        """Test getting cached features when cache is expired."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock cached features with old timestamp
        mock_features = np.random.randn(10, 5)
        feed.feature_cache["AAPL"] = mock_features
        feed.last_update["AAPL"] = datetime.now() - timedelta(hours=2)

        with pytest.raises(ValueError):
            feed.get_cached_features("AAPL")

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Add some cached data
        feed.price_cache["AAPL"] = pd.DataFrame({"Close": [150.0]})
        feed.feature_cache["AAPL"] = np.random.randn(5, 3)
        feed.last_update["AAPL"] = datetime.now()

        feed.clear_cache()

        assert len(feed.price_cache) == 0
        assert len(feed.feature_cache) == 0
        assert len(feed.last_update) == 0

    def test_clear_cache_specific_symbol(self):
        """Test clearing cache for specific symbol."""
        feed = LiveDataFeed(["AAPL", "GOOGL"], data_source="yfinance")

        # Add cached data for both symbols
        feed.price_cache["AAPL"] = pd.DataFrame({"Close": [150.0]})
        feed.price_cache["GOOGL"] = pd.DataFrame({"Close": [2500.0]})
        feed.feature_cache["AAPL"] = np.random.randn(5, 3)
        feed.feature_cache["GOOGL"] = np.random.randn(5, 3)

        feed.clear_cache("AAPL")

        assert "AAPL" not in feed.price_cache
        assert "AAPL" not in feed.feature_cache
        assert "GOOGL" in feed.price_cache
        assert "GOOGL" in feed.feature_cache

    @pytest.mark.asyncio
    async def test_start_continuous_update(self):
        """Test starting continuous update loop."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")
        feed.update_interval = 0.1  # Short interval for testing

        # Mock update method
        feed.update_all_symbols = AsyncMock()

        # Start continuous update
        task = asyncio.create_task(feed.start_continuous_update())

        # Wait a bit for the first update
        await asyncio.sleep(0.2)

        # Cancel the task
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Check that update was called
        assert feed.update_all_symbols.called

    def test_get_system_status(self):
        """Test getting system status information."""
        feed = LiveDataFeed(["AAPL", "GOOGL"], data_source="yfinance")

        # Add some cached data
        feed.price_cache["AAPL"] = pd.DataFrame({"Close": [150.0]})
        feed.last_update["AAPL"] = datetime.now()

        status = feed.get_system_status()

        assert "connected" in status
        assert "symbols" in status
        assert "cached_symbols" in status
        assert "last_update" in status
        assert status["connected"] is False
        assert len(status["symbols"]) == 2
        assert len(status["cached_symbols"]) == 1

    def test_validate_symbol(self):
        """Test symbol validation."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Valid symbol
        assert feed.validate_symbol("AAPL") is True

        # Invalid symbol
        assert feed.validate_symbol("INVALID_SYMBOL") is False

        # Empty symbol
        assert feed.validate_symbol("") is False

        # None symbol
        assert feed.validate_symbol(None) is False

    def test_set_update_interval(self):
        """Test setting update interval."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Test valid interval
        feed.set_update_interval(30)
        assert feed.update_interval == 30

        # Test invalid interval
        with pytest.raises(ValueError):
            feed.set_update_interval(-1)

        with pytest.raises(ValueError):
            feed.set_update_interval(0)

    def test_add_symbol(self):
        """Test adding new symbol to feed."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        feed.add_symbol("GOOGL")
        assert "GOOGL" in feed.symbols
        assert len(feed.symbols) == 2

    def test_add_symbol_duplicate(self):
        """Test adding duplicate symbol."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        feed.add_symbol("AAPL")  # Should not add duplicate
        assert len(feed.symbols) == 1

    def test_remove_symbol(self):
        """Test removing symbol from feed."""
        feed = LiveDataFeed(["AAPL", "GOOGL"], data_source="yfinance")

        # Add some cached data
        feed.price_cache["AAPL"] = pd.DataFrame({"Close": [150.0]})
        feed.feature_cache["AAPL"] = np.random.randn(5, 3)
        feed.last_update["AAPL"] = datetime.now()

        feed.remove_symbol("AAPL")

        assert "AAPL" not in feed.symbols
        assert "AAPL" not in feed.price_cache
        assert "AAPL" not in feed.feature_cache
        assert "AAPL" not in feed.last_update
        assert "GOOGL" in feed.symbols

    def test_remove_symbol_not_found(self):
        """Test removing non-existent symbol."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with pytest.raises(ValueError):
            feed.remove_symbol("INVALID_SYMBOL")


class TestErrorHandling:
    """Test suite for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with (
            patch.object(yf.Ticker, "history", side_effect=TimeoutError("Network timeout")),
            pytest.raises(TimeoutError),
        ):
            await feed.fetch_latest_data("AAPL")

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test handling of rate limiting."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        with (
            patch.object(yf.Ticker, "history", side_effect=Exception("Rate limit exceeded")),
            pytest.raises(RuntimeError),
        ):
            await feed.fetch_latest_data("AAPL")

    def test_invalid_data_format(self):
        """Test handling of invalid data format."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock data with missing required columns
        invalid_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                # Missing Low, Close, Volume
            }
        )

        feed.price_cache["AAPL"] = invalid_data

        with pytest.raises(ValueError):
            feed.calculate_features("AAPL")

    def test_memory_cleanup(self):
        """Test memory cleanup functionality."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Add large cached data
        large_data = pd.DataFrame({"Close": np.random.randn(10000)})
        feed.price_cache["AAPL"] = large_data

        len(feed.price_cache["AAPL"])

        # Simulate memory cleanup
        feed.clear_cache()

        assert len(feed.price_cache) == 0


class TestIntegration:
    """Integration tests for live data feed."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from connection to data retrieval."""
        feed = LiveDataFeed(["AAPL"], data_source="yfinance")

        # Mock data
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        with patch.object(yf.Ticker, "history", return_value=mock_data):
            # Connect
            await feed.connect()
            assert feed.connected is True

            # Update data
            await feed.update_all_symbols()
            assert "AAPL" in feed.price_cache

            # Get latest price
            price = feed.get_latest_price("AAPL")
            assert price == 153.0

            # Calculate features
            features = feed.calculate_features("AAPL")
            assert isinstance(features, np.ndarray)

            # Disconnect
            await feed.disconnect()
            assert feed.connected is False

    @pytest.mark.asyncio
    async def test_multiple_symbols_workflow(self):
        """Test workflow with multiple symbols."""
        symbols = ["AAPL", "GOOGL", "MSFT"]
        feed = LiveDataFeed(symbols, data_source="yfinance")

        # Mock data for all symbols
        mock_data = pd.DataFrame(
            {
                "Open": [150.0],
                "High": [155.0],
                "Low": [149.0],
                "Close": [153.0],
                "Volume": [1000000],
            },
            index=[datetime.now()],
        )

        with patch.object(yf.Ticker, "history", return_value=mock_data):
            await feed.connect()
            await feed.update_all_symbols()

            # Check all symbols have data
            for symbol in symbols:
                assert symbol in feed.price_cache
                assert not feed.price_cache[symbol].empty

                price = feed.get_latest_price(symbol)
                assert price == 153.0

            await feed.disconnect()
