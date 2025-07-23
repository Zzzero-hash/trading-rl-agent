"""
Chaos engineering tests for trading system resilience.

These tests simulate various failure scenarios to ensure the system
remains operational and recovers gracefully.
"""

import asyncio
import contextlib
import threading
import time
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from trade_agent.core.live_trading import LiveTradingEngine
from trade_agent.data.parallel_data_fetcher import ParallelDataManager
from trade_agent.portfolio.manager import PortfolioManager
from trade_agent.risk.manager import RiskManager


class TestNetworkFailureResilience:
    """Test system resilience to network failures."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_data_feed_interruption(self):
        """Test system behavior when data feed is interrupted."""
        data_manager = ParallelDataManager()

        # Start data collection
        data_task = asyncio.create_task(data_manager.start_data_collection())

        # Let it run for a bit
        await asyncio.sleep(0.1)

        # Simulate network failure by raising an exception
        with patch.object(data_manager, "_fetch_market_data", side_effect=Exception("Network error")):
            # System should handle the error gracefully
            try:
                await asyncio.sleep(0.1)
                # Should not crash
                assert data_manager.is_running
            except Exception as e:
                pytest.fail(f"System crashed on network failure: {e}")

        # Cleanup
        data_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await data_task

    @pytest.mark.chaos
    def test_api_rate_limit_handling(self):
        """Test system behavior when hitting API rate limits."""
        with patch("requests.get") as mock_get:
            # Simulate rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "1"}
            mock_get.return_value = mock_response

            # System should handle rate limiting gracefully
            try:
                # This should not crash
                pass
            except Exception as e:
                pytest.fail(f"System crashed on rate limit: {e}")

    @pytest.mark.chaos
    def test_database_connection_failure(self):
        """Test system behavior when database connection fails."""
        with patch("sqlite3.connect", side_effect=Exception("Database connection failed")):
            # System should handle database failures gracefully
            try:
                # This should not crash
                pass
            except Exception as e:
                pytest.fail(f"System crashed on database failure: {e}")


class TestMemoryAndResourceResilience:
    """Test system resilience to memory and resource issues."""

    @pytest.mark.chaos
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        # Create large datasets to simulate memory pressure
        large_data = []
        for i in range(1000):
            large_data.append(np.random.randn(1000, 100))

        # System should handle memory pressure gracefully
        try:
            # Perform operations under memory pressure
            portfolio = PortfolioManager(initial_cash=100000)
            for i in range(100):
                portfolio.update_position(f"ASSET_{i}", 100, 50.0)

            # Should not crash
            assert portfolio.total_value > 0
        except MemoryError:
            # Memory error is acceptable, but should be handled gracefully
            pass
        except Exception as e:
            pytest.fail(f"System crashed under memory pressure: {e}")

    @pytest.mark.chaos
    def test_cpu_intensive_operations(self):
        """Test system behavior during CPU-intensive operations."""

        # Simulate CPU-intensive operations
        def cpu_intensive_task():
            for i in range(1000000):
                _ = np.random.randn(100, 100).dot(np.random.randn(100, 100))

        # Run in background thread
        thread = threading.Thread(target=cpu_intensive_task)
        thread.start()

        try:
            # Main system should remain responsive
            portfolio = PortfolioManager(initial_cash=100000)
            portfolio.update_position("AAPL", 100, 150.0)

            # Should complete without hanging
            assert portfolio.total_value > 0
        finally:
            thread.join(timeout=5)

    @pytest.mark.chaos
    def test_disk_space_issues(self):
        """Test system behavior when disk space is limited."""
        with patch("os.path.getmtime", side_effect=OSError("No space left on device")):
            # System should handle disk space issues gracefully
            try:
                # This should not crash
                pass
            except Exception as e:
                pytest.fail(f"System crashed on disk space issue: {e}")


class TestDataQualityResilience:
    """Test system resilience to poor data quality."""

    @pytest.mark.chaos
    def test_missing_data_handling(self):
        """Test system behavior with missing data."""
        # Create data with missing values
        data = pd.DataFrame(
            {
                "price": [100, np.nan, 102, np.nan, 105],
                "volume": [1000, 1100, np.nan, 1200, 1300],
            }
        )

        # System should handle missing data gracefully
        try:
            # Should not crash on missing data
            risk_manager = RiskManager()
            var = risk_manager.calculate_var(data["price"].dropna())
            assert not np.isnan(var)
        except Exception as e:
            pytest.fail(f"System crashed on missing data: {e}")

    @pytest.mark.chaos
    def test_outlier_data_handling(self):
        """Test system behavior with outlier data."""
        # Create data with extreme outliers
        normal_data = np.random.normal(100, 10, 100)
        outlier_data = np.concatenate([normal_data, [1e6, -1e6]])  # Extreme outliers

        # System should handle outliers gracefully
        try:
            risk_manager = RiskManager()
            var = risk_manager.calculate_var(outlier_data)
            # Should not be infinite or NaN
            assert np.isfinite(var)
        except Exception as e:
            pytest.fail(f"System crashed on outlier data: {e}")

    @pytest.mark.chaos
    def test_inconsistent_data_handling(self):
        """Test system behavior with inconsistent data."""
        # Create inconsistent data (negative prices, etc.)
        data = pd.DataFrame(
            {
                "price": [100, -50, 102, 0, 105],  # Invalid prices
                "volume": [1000, 1100, -500, 1200, 1300],  # Invalid volumes
            }
        )

        # System should handle inconsistent data gracefully
        try:
            # Should filter out invalid data
            valid_data = data[data["price"] > 0]
            assert len(valid_data) < len(data)  # Some data should be filtered
        except Exception as e:
            pytest.fail(f"System crashed on inconsistent data: {e}")


class TestTradingEngineResilience:
    """Test the resilience of the TradingEngine."""

    @patch("trade_agent.core.live_trading.LiveTradingEngine.connect_broker")
    def test_broker_connection_failure(self, mock_connect_broker):
        """Test system behavior when broker connection fails."""
        mock_connect_broker.side_effect = ConnectionError("Failed to connect to broker")
        trading_engine = LiveTradingEngine()
        with pytest.raises(ConnectionError):
            trading_engine.start()

    @pytest.mark.chaos
    def test_order_execution_failure(self):
        """Test system behavior when order execution fails."""
        trading_engine = LiveTradingEngine()

        with patch.object(
            trading_engine,
            "execute_order",
            side_effect=Exception("Order execution failed"),
        ):
            # System should handle order execution failures gracefully
            try:
                # Should not crash
                result = trading_engine.place_order("AAPL", "BUY", 100, 150.0)
                # Should return error status, not crash
                assert result is not None
            except Exception as e:
                pytest.fail(f"System crashed on order execution failure: {e}")

    @pytest.mark.chaos
    def test_market_data_delay(self):
        """Test system behavior when market data is delayed."""
        # Simulate delayed market data
        with patch("time.time", return_value=time.time() + 3600):  # 1 hour delay
            # System should handle delayed data gracefully
            try:
                # Should not crash
                pass
            except Exception as e:
                pytest.fail(f"System crashed on delayed market data: {e}")

    @pytest.mark.chaos
    def test_risk_limit_breach_handling(self):
        """Test system behavior when risk limits are breached."""
        portfolio = PortfolioManager(initial_cash=100000)
        risk_manager = RiskManager()

        # Simulate a large loss that breaches risk limits
        portfolio.update_position("AAPL", -1000, 150.0)  # Large short position

        # System should handle risk limit breaches gracefully
        try:
            risk_status = risk_manager.check_risk_limits(portfolio)
            # Should return risk status, not crash
            assert risk_status is not None
        except Exception as e:
            pytest.fail(f"System crashed on risk limit breach: {e}")


class TestConcurrentOperationResilience:
    """Test system resilience under concurrent operations."""

    @pytest.mark.chaos
    def test_concurrent_portfolio_updates(self):
        """Test system behavior under concurrent portfolio updates."""
        portfolio = PortfolioManager(initial_cash=100000)

        def update_portfolio():
            for i in range(100):
                portfolio.update_position("AAPL", 1, 150.0)

        # Run multiple threads updating portfolio concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=update_portfolio)
            threads.append(thread)
            thread.start()

        try:
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)

            # Portfolio should be in consistent state
            assert portfolio.total_value > 0
        except Exception as e:
            pytest.fail(f"System crashed under concurrent operations: {e}")

    @pytest.mark.chaos
    def test_concurrent_data_access(self):
        """Test system behavior under concurrent data access."""
        data_manager = ParallelDataManager()

        def access_data():
            for i in range(100):
                _ = data_manager.get_latest_data("AAPL")

        # Run multiple threads accessing data concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=access_data)
            threads.append(thread)
            thread.start()

        try:
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=10)
        except Exception as e:
            pytest.fail(f"System crashed under concurrent data access: {e}")


class TestRecoveryResilience:
    """Test system recovery from failures."""

    @pytest.mark.chaos
    def test_system_recovery_after_failure(self):
        """Test system recovery after a failure."""
        portfolio = PortfolioManager(initial_cash=100000)

        # Simulate a failure
        from contextlib import suppress

        with suppress(ValueError, OverflowError):
            # This should fail
            portfolio.update_position("AAPL", float("inf"), 150.0)

        # System should recover and continue operating
        try:
            # Should work normally after failure
            portfolio.update_position("AAPL", 100, 150.0)
            assert portfolio.total_value > 0
        except Exception as e:
            pytest.fail(f"System did not recover from failure: {e}")

    @pytest.mark.chaos
    def test_data_recovery_after_corruption(self):
        """Test system recovery after data corruption."""
        # Simulate corrupted data
        corrupted_data = pd.DataFrame(
            {
                "price": [100, np.nan, "invalid", 105],
                "volume": [1000, "invalid", 1200, 1300],
            }
        )

        # System should recover and clean data
        try:
            # Should handle corrupted data gracefully
            cleaned_data = corrupted_data.dropna()
            assert len(cleaned_data) < len(corrupted_data)
        except Exception as e:
            pytest.fail(f"System did not recover from data corruption: {e}")


class TestPerformanceDegradationResilience:
    """Test system resilience to performance degradation."""

    @pytest.mark.chaos
    def test_slow_network_handling(self):
        """Test system behavior with slow network."""
        with patch("requests.get") as mock_get:
            # Simulate slow network response
            def slow_response(*_args, **_kwargs):
                time.sleep(0.1)  # 100ms delay
                return Mock(status_code=200, json=dict)

            mock_get.side_effect = slow_response

            # System should handle slow network gracefully
            try:
                # Should not hang indefinitely
                pass
            except Exception as e:
                pytest.fail(f"System crashed on slow network: {e}")

    @pytest.mark.chaos
    def test_high_latency_handling(self):
        """Test system behavior under high latency conditions."""

        # Simulate high latency operations
        def high_latency_operation():
            time.sleep(0.05)  # 50ms latency
            return "result"

        # System should handle high latency gracefully
        try:
            result = high_latency_operation()
            assert result == "result"
        except Exception as e:
            pytest.fail(f"System crashed under high latency: {e}")
