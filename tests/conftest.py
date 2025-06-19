"""
pytest configuration and fixtures for test data management.
"""

import logging

import pandas as pd
import pytest

from tests.test_data_utils import TestDataManager, get_dynamic_test_config

# Import legacy function for backward compatibility
try:
    from generate_sample_data import generate_sample_price_data
except ImportError:
    # Fallback if generate_sample_data is not available
    def generate_sample_price_data(
        symbol="TEST", days=30, start_price=100.0, volatility=0.01
    ):
        from datetime import datetime, timedelta

        import numpy as np

        np.random.seed(42)
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(days)]
        prices = [start_price]
        for _ in range(days - 1):
            change = np.random.normal(0, volatility * start_price)
            prices.append(max(0.01, prices[-1] + change))

        return pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": [symbol] * days,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 10000, days),
            }
        )


# Configure logging for tests
logging.basicConfig(level=logging.INFO)

# Global test data manager
_test_manager = None


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Auto-setup test data for entire test session."""
    global _test_manager
    _test_manager = TestDataManager()
    yield _test_manager
    # Cleanup after all tests
    if _test_manager:
        _test_manager.cleanup()


@pytest.fixture
def test_dataset():
    """Provide a test dataset for individual tests."""
    global _test_manager
    if _test_manager is None:
        _test_manager = TestDataManager()

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    dataset_path = _test_manager.get_or_create_test_dataset(required_columns)
    return dataset_path


@pytest.fixture
def dynamic_test_config():
    """Provide dynamic test configuration."""
    return get_dynamic_test_config()


@pytest.fixture(scope="session")
def sample_csv_path(tmp_path_factory):
    """Legacy fixture for backward compatibility."""
    data_dir = tmp_path_factory.mktemp("data")
    file_path = data_dir / "sample.csv"
    df = generate_sample_price_data(
        symbol="TEST", days=30, start_price=100.0, volatility=0.01
    )
    df = df.drop(columns=["timestamp", "symbol"])
    df.to_csv(file_path, index=False)
    return str(file_path)
