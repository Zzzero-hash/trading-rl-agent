#!/usr/bin/env python3
"""
Basic Alpaca Integration Test

Simple test script to verify the Alpaca integration works without requiring API credentials.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports() -> bool:
    """Test that all imports work correctly."""
    print("Testing imports...")

    try:
        from trading_rl_agent.data.alpaca_integration import (
            AlpacaIntegration,
            AlpacaOrderError,
            MarketData,
            OrderRequest,
            OrderSide,
            OrderType,
            PortfolioPosition,
            create_alpaca_config_from_env,
        )

        # Test that classes can be instantiated (basic import test)
        _ = AlpacaIntegration
        _ = AlpacaOrderError
        _ = MarketData
        _ = OrderRequest
        _ = OrderSide
        _ = OrderType
        _ = PortfolioPosition
        _ = create_alpaca_config_from_env

        print("✓ All Alpaca integration imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    try:
        from trading_rl_agent.configs.alpaca_config import (
            AlpacaConfigManager,
            AlpacaConfigModel,
        )

        # Test that classes can be referenced (basic import test)
        _ = AlpacaConfigManager
        _ = AlpacaConfigModel

        print("✓ All Alpaca config imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

    return True


def test_config_creation() -> bool:
    """Test configuration creation."""
    print("\nTesting configuration creation...")

    try:
        from trading_rl_agent.data.alpaca_integration import AlpacaConfig

        # Test basic config creation
        config = AlpacaConfig(api_key="test_key", secret_key="test_secret", paper_trading=True)
        print("✓ AlpacaConfig creation successful")
        print(f"  - API Key: {config.api_key}")
        print(f"  - Paper Trading: {config.paper_trading}")
        print(f"  - Base URL: {config.base_url}")

        return True
    except Exception as e:
        print(f"✗ Config creation error: {e}")
        return False


def test_data_structures() -> bool:
    """Test data structure creation."""
    print("\nTesting data structures...")

    try:
        from datetime import datetime

        from trading_rl_agent.data.alpaca_integration import (
            MarketData,
            OrderRequest,
            OrderSide,
            OrderType,
            PortfolioPosition,
        )

        # Test OrderRequest
        order_request = OrderRequest(
            symbol="AAPL", qty=10.0, side=OrderSide.BUY, order_type=OrderType.MARKET, time_in_force="day"
        )
        print("✓ OrderRequest creation successful")
        print(f"  - Symbol: {order_request.symbol}")
        print(f"  - Side: {order_request.side}")
        print(f"  - Type: {order_request.order_type}")

        # Test MarketData
        market_data = MarketData(
            symbol="AAPL", timestamp=datetime.now(), open=150.00, high=155.00, low=149.00, close=154.00, volume=1000000
        )
        print("✓ MarketData creation successful")
        print(f"  - Symbol: {market_data.symbol}")
        print(f"  - Close: ${market_data.close:.2f}")
        print(f"  - Volume: {market_data.volume:,}")

        # Test PortfolioPosition
        position = PortfolioPosition(
            symbol="AAPL",
            qty=10.0,
            avg_entry_price=150.00,
            current_price=155.00,
            market_value=1550.00,
            unrealized_pl=50.00,
            unrealized_plpc=0.0333,
            side="long",
            timestamp=datetime.now(),
        )
        print("✓ PortfolioPosition creation successful")
        print(f"  - Symbol: {position.symbol}")
        print(f"  - Quantity: {position.qty}")
        print(f"  - P&L: ${position.unrealized_pl:.2f}")

        return True
    except Exception as e:
        print(f"✗ Data structure error: {e}")
        return False


def test_config_validation() -> bool:
    """Test configuration validation."""
    print("\nTesting configuration validation...")

    try:
        from trading_rl_agent.configs.alpaca_config import AlpacaConfigModel

        # Test valid config
        config = AlpacaConfigModel(api_key="test_key", secret_key="test_secret")
        print("✓ Valid configuration creation successful")

        # Test invalid config (should raise error)
        try:
            invalid_config = AlpacaConfigModel(api_key="", secret_key="test_secret")
            print("✗ Invalid config should have raised error")
            return False
        except ValueError:
            print("✓ Invalid configuration properly rejected")

        return True
    except Exception as e:
        print(f"✗ Config validation error: {e}")
        return False


def test_exceptions() -> bool:
    """Test custom exceptions."""
    print("\nTesting custom exceptions...")

    try:
        from trading_rl_agent.data.alpaca_integration import (
            AlpacaConnectionError,
            AlpacaDataError,
            AlpacaError,
            AlpacaOrderError,
        )

        # Test exception hierarchy
        assert issubclass(AlpacaConnectionError, AlpacaError)
        assert issubclass(AlpacaOrderError, AlpacaError)
        assert issubclass(AlpacaDataError, AlpacaError)
        print("✓ Exception hierarchy correct")

        # Test exception creation
        conn_error = AlpacaConnectionError("Connection failed")
        order_error = AlpacaOrderError("Order failed")
        data_error = AlpacaDataError("Data retrieval failed")

        print("✓ Custom exceptions creation successful")
        print(f"  - Connection Error: {conn_error}")
        print(f"  - Order Error: {order_error}")
        print(f"  - Data Error: {data_error}")

        return True
    except Exception as e:
        print(f"✗ Exception test error: {e}")
        return False


def main() -> int:
    """Run all tests."""
    print("Alpaca Integration Basic Test")
    print("=" * 40)

    tests = [
        test_imports,
        test_config_creation,
        test_data_structures,
        test_config_validation,
        test_exceptions,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Alpaca integration is working correctly.")
        return 0
    print("❌ Some tests failed. Please check the errors above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
