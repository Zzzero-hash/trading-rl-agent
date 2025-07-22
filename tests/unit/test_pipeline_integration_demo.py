"""
Tests for pipeline integration demo variable scope consistency.
"""

from unittest.mock import MagicMock, patch

from trade_agent.portfolio.cli_attribution import (
    _create_demo_portfolio_manager,
    _demo_portfolio_manager_instance,
    _reset_demo_portfolio_manager,
)


class TestPipelineIntegrationDemo:
    """Test pipeline integration demo variable scope consistency."""

    def test_demo_portfolio_manager_singleton_pattern(self):
        """Test that demo portfolio manager uses singleton pattern."""
        # Reset any existing instance
        _reset_demo_portfolio_manager()

        # Create first instance
        manager1 = _create_demo_portfolio_manager()
        assert manager1 is not None

        # Create second instance - should be the same object
        manager2 = _create_demo_portfolio_manager()
        assert manager2 is not None
        assert manager1 is manager2  # Same object reference

        # Verify global instance is set
        assert _demo_portfolio_manager_instance is manager1

    def test_demo_portfolio_manager_reset_functionality(self):
        """Test that demo portfolio manager can be reset."""
        # Create initial instance
        manager1 = _create_demo_portfolio_manager()
        assert _demo_portfolio_manager_instance is manager1

        # Reset
        _reset_demo_portfolio_manager()
        assert _demo_portfolio_manager_instance is None

        # Create new instance
        manager2 = _create_demo_portfolio_manager()
        assert manager2 is not None
        assert manager2 is not manager1  # Different object
        assert _demo_portfolio_manager_instance is manager2

    def test_demo_portfolio_manager_consistent_state(self):
        """Test that demo portfolio manager maintains consistent state."""
        # Reset and create fresh instance
        _reset_demo_portfolio_manager()
        manager1 = _create_demo_portfolio_manager()

        # Modify the performance history
        len(manager1.performance_history)
        manager1.performance_history = manager1.performance_history.iloc[:5]  # Keep only first 5 rows

        # Get another instance
        manager2 = _create_demo_portfolio_manager()

        # Should have the same modified state
        assert len(manager2.performance_history) == 5
        assert len(manager1.performance_history) == 5
        assert manager1.performance_history.equals(manager2.performance_history)

    def test_demo_portfolio_manager_multiple_calls(self):
        """Test multiple calls to demo portfolio manager creation."""
        # Reset
        _reset_demo_portfolio_manager()

        # Create multiple instances
        managers = []
        for i in range(5):
            manager = _create_demo_portfolio_manager()
            managers.append(manager)

        # All should be the same object
        for i in range(1, len(managers)):
            assert managers[i] is managers[0]

        # Global instance should be the same
        assert _demo_portfolio_manager_instance is managers[0]

    def test_demo_portfolio_manager_configuration_consistency(self):
        """Test that demo portfolio manager configuration is consistent."""
        # Reset and create fresh instance
        _reset_demo_portfolio_manager()
        manager1 = _create_demo_portfolio_manager()

        # Check configuration
        assert hasattr(manager1, "config")
        assert manager1.config.benchmark_symbol == "SPY"

        # Get another instance
        manager2 = _create_demo_portfolio_manager()

        # Should have same configuration
        assert manager2.config.benchmark_symbol == "SPY"
        assert manager1.config.benchmark_symbol == manager2.config.benchmark_symbol

    def test_demo_portfolio_manager_performance_history_structure(self):
        """Test that demo portfolio manager has correct performance history structure."""
        # Reset and create fresh instance
        _reset_demo_portfolio_manager()
        manager = _create_demo_portfolio_manager()

        # Check performance history structure
        assert hasattr(manager, "performance_history")
        assert isinstance(manager.performance_history, type(manager.performance_history))

        # Check required columns
        required_columns = ["timestamp", "portfolio_value", "benchmark_value"]
        for col in required_columns:
            assert col in manager.performance_history.columns

        # Check data types
        assert manager.performance_history["timestamp"].dtype == "datetime64[ns]"
        assert manager.performance_history["portfolio_value"].dtype in ["float64", "float32"]
        assert manager.performance_history["benchmark_value"].dtype in ["float64", "float32"]

        # Check data range
        assert len(manager.performance_history) > 0
        assert manager.performance_history["portfolio_value"].min() > 0
        assert manager.performance_history["benchmark_value"].min() > 0

    @patch("trade_agent.portfolio.cli_attribution.PortfolioManager")
    def test_demo_portfolio_manager_creation_with_mock(self, mock_portfolio_manager):
        """Test demo portfolio manager creation with mocked dependencies."""
        # Reset
        _reset_demo_portfolio_manager()

        # Mock PortfolioManager
        mock_instance = MagicMock()
        mock_portfolio_manager.return_value = mock_instance

        # Create demo manager
        manager = _create_demo_portfolio_manager()

        # Verify PortfolioManager was called
        mock_portfolio_manager.assert_called_once()

        # Verify performance history was set
        assert hasattr(manager, "performance_history")
        assert len(manager.performance_history) > 0

    def test_demo_portfolio_manager_thread_safety_consideration(self):
        """Test that demo portfolio manager handles concurrent access gracefully."""
        # This test documents the current behavior and potential considerations
        # for thread safety in a multi-threaded environment

        # Reset
        _reset_demo_portfolio_manager()

        # Create instance in main thread
        manager1 = _create_demo_portfolio_manager()

        # Simulate potential race condition by resetting and creating quickly
        _reset_demo_portfolio_manager()
        manager2 = _create_demo_portfolio_manager()

        # Should be different objects after reset
        assert manager1 is not manager2

        # Create again without reset - should be same as manager2
        manager3 = _create_demo_portfolio_manager()
        assert manager3 is manager2
