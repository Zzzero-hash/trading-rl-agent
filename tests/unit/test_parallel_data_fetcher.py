"""
Comprehensive tests for parallel data fetcher.

This module tests:
- Ray remote ParallelDataFetcher class
- Caching system with TTL
- Memory-mapped datasets
- Parallel data management
- Retry logic and error handling
- Performance benchmarks
- Edge cases and memory optimization
"""

import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import ray

from trading_rl_agent.data.parallel_data_fetcher import (
    MemoryMappedDataset,
    ParallelDataFetcher,
    ParallelDataManager,
    SmartDataCache,
    fetch_data_parallel,
    fetch_data_with_retry,
)


@pytest.fixture(scope="module")
def ray_init():
    """Initialize Ray for testing."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestParallelDataFetcher:
    """Test Ray remote ParallelDataFetcher class."""

    def test_fetcher_initialization(self, tmp_path):
        """Test ParallelDataFetcher initialization."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        # Check that cache directory was created
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_fetch_symbol_data_success(self, tmp_path):
        """Test successful data fetching."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-03", "1d"))

        assert result["success"] is True
        assert result["symbol"] == "AAPL"
        assert result["source"] == "api"
        assert isinstance(result["data"], pd.DataFrame)
        assert len(result["data"]) == 3

    def test_fetch_symbol_data_cache_hit(self, tmp_path):
        """Test cache hit scenario."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        # Create cached data
        cached_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        cache_key = "AAPL_2024-01-01_2024-01-03_1d.parquet"
        cache_path = cache_dir / cache_key
        cached_data.to_parquet(cache_path)

        # Fetch data (should hit cache)
        result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-03", "1d"))

        assert result["success"] is True
        assert result["source"] == "cache"
        assert len(result["data"]) == 3

    def test_fetch_symbol_data_cache_expired(self, tmp_path):
        """Test cache expiration scenario."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=1)

        # Create old cached data
        cached_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        cache_key = "AAPL_2024-01-01_2024-01-03_1d.parquet"
        cache_path = cache_dir / cache_key
        cached_data.to_parquet(cache_path)

        # Make file old
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(cache_path, (old_time, old_time))

        # Mock fresh data
        mock_data = pd.DataFrame(
            {
                "Open": [105.0, 106.0, 107.0],
                "High": [107.0, 108.0, 109.0],
                "Low": [104.0, 105.0, 106.0],
                "Close": [106.0, 107.0, 108.0],
                "Volume": [1500000, 1600000, 1700000],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-03", "1d"))

        assert result["success"] is True
        assert result["source"] == "api"  # Should fetch fresh data

    def test_fetch_symbol_data_retry_logic(self, tmp_path):
        """Test retry logic for failed requests."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            # Fail first two attempts, succeed on third
            mock_ticker_instance.history.side_effect = [
                Exception("Network error"),
                Exception("Timeout"),
                pd.DataFrame(
                    {
                        "Open": [100.0],
                        "High": [102.0],
                        "Low": [99.0],
                        "Close": [101.0],
                        "Volume": [1000000],
                    },
                    index=pd.date_range("2024-01-01", periods=1),
                ),
            ]
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-02", "1d", max_retries=3))

        assert result["success"] is True
        assert result["source"] == "api"

    def test_fetch_symbol_data_max_retries_exceeded(self, tmp_path):
        """Test behavior when max retries are exceeded."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = Exception("Persistent error")
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-02", "1d", max_retries=2))

        assert result["success"] is False
        assert result["source"] == "api"
        assert "error" in result

    def test_fetch_symbol_data_empty_response(self, tmp_path):
        """Test handling of empty data response."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("INVALID", "2024-01-01", "2024-01-02", "1d"))

        assert result["success"] is False
        assert result["error"] == "Empty data returned"

    def test_fetch_symbol_data_interval_mapping(self, tmp_path):
        """Test interval mapping for different timeframes."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

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

        intervals = ["1d", "1h", "5m", "15m", "30m"]

        for interval in intervals:
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = mock_data
                mock_ticker.return_value = mock_ticker_instance

                result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-02", interval))

                assert result["success"] is True


class TestSmartDataCache:
    """Test SmartDataCache functionality."""

    def test_cache_initialization(self, tmp_path):
        """Test SmartDataCache initialization."""
        cache_dir = tmp_path / "cache"
        cache = SmartDataCache(str(cache_dir), ttl_hours=24)

        assert cache.cache_dir == cache_dir
        assert cache.ttl == 24 * 3600
        assert cache.cache_stats["hits"] == 0
        assert cache.cache_stats["misses"] == 0

    def test_cache_hit_scenario(self, tmp_path):
        """Test cache hit scenario."""
        cache_dir = tmp_path / "cache"
        cache = SmartDataCache(str(cache_dir), ttl_hours=24)

        # Create cached data
        cached_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        cache_key = "AAPL_2024-01-01_2024-01-03_1d.parquet"
        cache_path = cache_dir / cache_key
        cached_data.to_parquet(cache_path)

        # Get data (should hit cache)
        result = cache.get_or_fetch("AAPL", "2024-01-01", "2024-01-03", "1d")

        assert len(result) == 3
        assert cache.cache_stats["hits"] == 1
        assert cache.cache_stats["misses"] == 0

    def test_cache_miss_scenario(self, tmp_path):
        """Test cache miss scenario."""
        cache_dir = tmp_path / "cache"
        cache = SmartDataCache(str(cache_dir), ttl_hours=24)

        # Get data (should miss cache)
        result = cache.get_or_fetch("AAPL", "2024-01-01", "2024-01-03", "1d")

        assert len(result) == 0  # Empty DataFrame for cache miss
        assert cache.cache_stats["hits"] == 0
        assert cache.cache_stats["misses"] == 1

    def test_cache_expiration(self, tmp_path):
        """Test cache expiration."""
        cache_dir = tmp_path / "cache"
        cache = SmartDataCache(str(cache_dir), ttl_hours=1)

        # Create old cached data
        cached_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "open": [100.0, 101.0, 102.0],
                "high": [102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0],
                "close": [101.0, 102.0, 103.0],
                "volume": [1000000, 1100000, 1200000],
            }
        )

        cache_key = "AAPL_2024-01-01_2024-01-03_1d.parquet"
        cache_path = cache_dir / cache_key
        cached_data.to_parquet(cache_path)

        # Make file old
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(cache_path, (old_time, old_time))

        # Get data (should miss cache due to expiration)
        result = cache.get_or_fetch("AAPL", "2024-01-01", "2024-01-03", "1d")

        assert len(result) == 0
        assert cache.cache_stats["misses"] == 1

    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache_dir = tmp_path / "cache"
        cache = SmartDataCache(str(cache_dir), ttl_hours=24)

        # Simulate some hits and misses
        cache.cache_stats["hits"] = 5
        cache.cache_stats["misses"] = 3

        stats = cache.get_stats()

        assert stats["hits"] == 5
        assert stats["misses"] == 3
        assert stats["total"] == 8
        assert stats["hit_rate"] == 5 / 8

    def test_cache_clear(self, tmp_path):
        """Test cache clearing."""
        cache_dir = tmp_path / "cache"
        cache = SmartDataCache(str(cache_dir), ttl_hours=24)

        # Create some cached files
        cached_data = pd.DataFrame({"test": [1, 2, 3]})
        cache_path1 = cache_dir / "test1.parquet"
        cache_path2 = cache_dir / "test2.parquet"
        cached_data.to_parquet(cache_path1)
        cached_data.to_parquet(cache_path2)

        assert cache_path1.exists()
        assert cache_path2.exists()

        # Clear cache
        cache.clear_cache()

        assert not cache_path1.exists()
        assert not cache_path2.exists()


class TestMemoryMappedDataset:
    """Test MemoryMappedDataset functionality."""

    def test_dataset_initialization(self, tmp_path):
        """Test MemoryMappedDataset initialization."""
        # Create test data
        data = np.random.randn(1000, 10)
        data_path = tmp_path / "test_data.npy"
        np.save(data_path, data)

        dataset = MemoryMappedDataset(str(data_path), chunk_size=100)

        assert dataset.file_path == data_path
        assert dataset.chunk_size == 100
        assert dataset.mmap is not None

    def test_get_batch(self, tmp_path):
        """Test getting batches from dataset."""
        # Create test data
        data = np.random.randn(1000, 10)
        data_path = tmp_path / "test_data.npy"
        np.save(data_path, data)

        dataset = MemoryMappedDataset(str(data_path), chunk_size=100)

        # Get first batch
        batch = dataset.get_batch(0, 50)
        assert batch.shape == (50, 10)
        assert np.array_equal(batch, data[:50])

        # Get middle batch
        batch = dataset.get_batch(500, 100)
        assert batch.shape == (100, 10)
        assert np.array_equal(batch, data[500:600])

    def test_dataset_length(self, tmp_path):
        """Test dataset length."""
        # Create test data
        data = np.random.randn(1000, 10)
        data_path = tmp_path / "test_data.npy"
        np.save(data_path, data)

        dataset = MemoryMappedDataset(str(data_path), chunk_size=100)

        assert len(dataset) == 1000

    def test_dataset_close(self, tmp_path):
        """Test dataset closing."""
        # Create test data
        data = np.random.randn(100, 10)
        data_path = tmp_path / "test_data.npy"
        np.save(data_path, data)

        dataset = MemoryMappedDataset(str(data_path), chunk_size=50)

        # Close dataset
        dataset.close()

        # Should handle close gracefully
        assert dataset.mmap is None


class TestParallelDataManager:
    """Test ParallelDataManager functionality."""

    def test_manager_initialization(self):
        """Test ParallelDataManager initialization."""
        manager = ParallelDataManager(cache_dir="test_cache", ttl_hours=24, max_workers=4)

        assert manager.cache_dir == Path("test_cache")
        assert manager.ttl == 24 * 3600
        assert manager.max_workers == 4

    def test_fetch_multiple_symbols(self, tmp_path):
        """Test fetching data for multiple symbols."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        symbols = ["AAPL", "GOOGL", "MSFT"]

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            results = manager.fetch_multiple_symbols(symbols, "2024-01-01", "2024-01-03", "1d")

        assert len(results) == 3
        for symbol in symbols:
            assert symbol in results
            assert isinstance(results[symbol], pd.DataFrame)
            assert len(results[symbol]) == 3

    def test_fetch_with_retry(self, tmp_path):
        """Test fetching with retry logic."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        symbols = ["AAPL", "GOOGL"]

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            results = manager.fetch_with_retry(symbols, "2024-01-01", "2024-01-03", "1d", max_retries=3)

        assert len(results) == 2
        for symbol in symbols:
            assert symbol in results
            assert isinstance(results[symbol], pd.DataFrame)

    def test_create_memory_mapped_dataset(self, tmp_path):
        """Test creating memory-mapped dataset."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        # Create test data
        data_dict = {
            "AAPL": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=100),
                    "open": np.random.uniform(100, 200, 100),
                    "high": np.random.uniform(100, 200, 100),
                    "low": np.random.uniform(100, 200, 100),
                    "close": np.random.uniform(100, 200, 100),
                    "volume": np.random.randint(1000000, 10000000, 100),
                }
            ),
            "GOOGL": pd.DataFrame(
                {
                    "timestamp": pd.date_range("2024-01-01", periods=100),
                    "open": np.random.uniform(100, 200, 100),
                    "high": np.random.uniform(100, 200, 100),
                    "low": np.random.uniform(100, 200, 100),
                    "close": np.random.uniform(100, 200, 100),
                    "volume": np.random.randint(1000000, 10000000, 100),
                }
            ),
        }

        output_path = tmp_path / "dataset.npy"
        dataset = manager.create_memory_mapped_dataset(data_dict, str(output_path))

        assert isinstance(dataset, MemoryMappedDataset)
        assert output_path.exists()

    def test_get_cache_stats(self, tmp_path):
        """Test getting cache statistics."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        stats = manager.get_cache_stats()

        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "total" in stats
        assert "hit_rate" in stats

    def test_clear_cache(self, tmp_path):
        """Test clearing cache."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        # Create some cached files
        cached_data = pd.DataFrame({"test": [1, 2, 3]})
        cache_path1 = cache_dir / "test1.parquet"
        cache_path2 = cache_dir / "test2.parquet"
        cached_data.to_parquet(cache_path1)
        cached_data.to_parquet(cache_path2)

        assert cache_path1.exists()
        assert cache_path2.exists()

        # Clear cache
        manager.clear_cache()

        assert not cache_path1.exists()
        assert not cache_path2.exists()


class TestUtilityFunctions:
    """Test utility functions."""

    def test_fetch_data_parallel(self, tmp_path):
        """Test fetch_data_parallel function."""
        symbols = ["AAPL", "GOOGL"]

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            results = fetch_data_parallel(
                symbols,
                "2024-01-01",
                "2024-01-03",
                "1d",
                cache_dir=str(tmp_path),
                max_workers=2,
            )

        assert len(results) == 2
        for symbol in symbols:
            assert symbol in results
            assert isinstance(results[symbol], pd.DataFrame)

    def test_fetch_data_with_retry(self, tmp_path):
        """Test fetch_data_with_retry function."""
        symbols = ["AAPL", "GOOGL"]

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            results = fetch_data_with_retry(
                symbols,
                "2024-01-01",
                "2024-01-03",
                "1d",
                cache_dir=str(tmp_path),
                max_retries=3,
            )

        assert len(results) == 2
        for symbol in symbols:
            assert symbol in results
            assert isinstance(results[symbol], pd.DataFrame)


class TestParallelDataFetcherEdgeCases:
    """Test edge cases and error scenarios."""

    def test_missing_yfinance(self, tmp_path):
        """Test handling when yfinance is not available."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        with patch("yfinance", None):
            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-02", "1d"))

        assert result["success"] is False
        assert "ImportError" in result["error"]

    def test_network_failures(self, tmp_path):
        """Test handling of network failures."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.side_effect = [
                Exception("Connection timeout"),
                Exception("DNS resolution failed"),
                Exception("Server error"),
            ]
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-02", "1d", max_retries=3))

        assert result["success"] is False
        assert "error" in result

    def test_invalid_symbols(self, tmp_path):
        """Test handling of invalid symbols."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("INVALID_SYMBOL_12345", "2024-01-01", "2024-01-02", "1d"))

        assert result["success"] is False
        assert result["error"] == "Empty data returned"

    def test_malformed_data(self, tmp_path):
        """Test handling of malformed data."""
        cache_dir = tmp_path / "cache"
        fetcher = ParallelDataFetcher.remote(str(cache_dir), ttl_hours=24)

        # Mock data with missing columns
        malformed_data = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [102.0, 103.0, 104.0],
                # Missing Low, Close, Volume
            },
            index=pd.date_range("2024-01-01", periods=3),
        )

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = malformed_data
            mock_ticker.return_value = mock_ticker_instance

            result = ray.get(fetcher.fetch_symbol_data.remote("AAPL", "2024-01-01", "2024-01-02", "1d"))

        # Should handle malformed data gracefully
        assert result["success"] is True
        assert isinstance(result["data"], pd.DataFrame)

    def test_memory_optimization(self, tmp_path):
        """Test memory optimization for large datasets."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        # Create large dataset
        large_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10000),
                "open": np.random.uniform(100, 200, 10000),
                "high": np.random.uniform(100, 200, 10000),
                "low": np.random.uniform(100, 200, 10000),
                "close": np.random.uniform(100, 200, 10000),
                "volume": np.random.randint(1000000, 10000000, 10000),
            }
        )

        data_dict = {"AAPL": large_data}
        output_path = tmp_path / "large_dataset.npy"

        # Should handle large dataset without memory issues
        dataset = manager.create_memory_mapped_dataset(data_dict, str(output_path))

        assert isinstance(dataset, MemoryMappedDataset)
        assert len(dataset) == 10000

    def test_concurrent_access(self, tmp_path):
        """Test concurrent access to parallel data fetcher."""
        import threading

        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=4)

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]

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

        results = []

        def fetch_data():
            with patch("yfinance.Ticker") as mock_ticker:
                mock_ticker_instance = Mock()
                mock_ticker_instance.history.return_value = mock_data
                mock_ticker.return_value = mock_ticker_instance

                result = manager.fetch_multiple_symbols(symbols[:2], "2024-01-01", "2024-01-03", "1d")
                results.append(result)

        # Create multiple threads
        threads = [threading.Thread(target=fetch_data) for _ in range(3)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all fetches completed successfully
        assert len(results) == 3
        for result in results:
            assert len(result) == 2


class TestParallelDataFetcherIntegration:
    """Integration tests for parallel data fetcher."""

    def test_end_to_end_workflow(self, tmp_path):
        """Test complete end-to-end workflow."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        symbols = ["AAPL", "GOOGL", "MSFT"]

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            # Fetch data
            results = manager.fetch_multiple_symbols(symbols, "2024-01-01", "2024-01-03", "1d")

            # Create memory-mapped dataset
            output_path = tmp_path / "dataset.npy"
            dataset = manager.create_memory_mapped_dataset(results, str(output_path))

            # Verify results
            assert len(results) == 3
            for symbol in symbols:
                assert symbol in results
                assert isinstance(results[symbol], pd.DataFrame)
                assert len(results[symbol]) == 3

            assert isinstance(dataset, MemoryMappedDataset)
            assert output_path.exists()

    def test_caching_workflow(self, tmp_path):
        """Test complete caching workflow."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        symbols = ["AAPL"]

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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            # First fetch (should hit API)
            results1 = manager.fetch_multiple_symbols(symbols, "2024-01-01", "2024-01-03", "1d")

            # Second fetch (should hit cache)
            results2 = manager.fetch_multiple_symbols(symbols, "2024-01-01", "2024-01-03", "1d")

            # Verify results are the same
            assert len(results1) == len(results2)
            for symbol in symbols:
                assert symbol in results1
                assert symbol in results2
                pd.testing.assert_frame_equal(results1[symbol], results2[symbol])

    @pytest.mark.benchmark
    def test_parallel_performance(self, benchmark, tmp_path):
        """Benchmark parallel data fetching performance."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=4)

        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        # Mock yfinance data
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

        with patch("yfinance.Ticker") as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker_instance.history.return_value = mock_data
            mock_ticker.return_value = mock_ticker_instance

            benchmark(lambda: manager.fetch_multiple_symbols(symbols, "2024-01-01", "2024-12-31", "1d"))

    @pytest.mark.benchmark
    def test_memory_mapped_performance(self, benchmark, tmp_path):
        """Benchmark memory-mapped dataset performance."""
        cache_dir = tmp_path / "cache"
        manager = ParallelDataManager(str(cache_dir), ttl_hours=24, max_workers=2)

        # Create large dataset
        large_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50000),
                "open": np.random.uniform(100, 200, 50000),
                "high": np.random.uniform(100, 200, 50000),
                "low": np.random.uniform(100, 200, 50000),
                "close": np.random.uniform(100, 200, 50000),
                "volume": np.random.randint(1000000, 10000000, 50000),
            }
        )

        data_dict = {"AAPL": large_data}
        output_path = tmp_path / "benchmark_dataset.npy"

        # Create dataset once
        dataset = manager.create_memory_mapped_dataset(data_dict, str(output_path))

        # Benchmark batch access
        benchmark(lambda: dataset.get_batch(0, 1000))
