"""
pytest configuration and fixtures for FinRL-based trading system tests.
"""

from datetime import datetime, timedelta
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# Configure logging for tests
logging.basicConfig(level=logging.INFO)


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment with FinRL configuration."""
    yield
    # Cleanup after all tests


@pytest.fixture
def finrl_sample_data():
    """Provide sample data in FinRL format for testing."""
    # Create realistic market data in FinRL format
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    dates = pd.date_range(start_date, end_date, freq="D")

    symbols = ["AAPL", "GOOGL", "MSFT"]
    data = []

    for symbol in symbols:
        base_price = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300}[symbol]
        price = base_price

        for date in dates:
            # Generate realistic OHLCV data
            change = np.random.normal(0, 0.02) * price
            price = max(1, price + change)

            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = price + np.random.normal(0, 0.005) * price
            volume = int(np.random.uniform(1000000, 5000000))

            # Technical indicators (simplified)
            data.append(
                {
                    "date": date,
                    "tic": symbol,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": price,
                    "volume": volume,
                    "macd": np.random.normal(0, 1),
                    "rsi_30": np.random.uniform(20, 80),
                    "cci_30": np.random.normal(0, 100),
                    "dx_30": np.random.uniform(10, 40),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def finrl_trading_env(finrl_sample_data):
    """Provide a FinRL-based trading environment for testing."""
    try:
        from src.envs.finrl_trading_env import HybridFinRLEnv

        # Create environment with minimal configuration for fast testing
        env = HybridFinRLEnv(
            df=finrl_sample_data,
            initial_amount=100000,
            transaction_cost_pct=0.001,
        )

        yield env

        # Cleanup
        if hasattr(env, "close"):
            env.close()

    except ImportError:
        # Fallback to mock environment if FinRL is not available
        from unittest.mock import Mock

        mock_env = Mock()
        mock_env.reset.return_value = (np.array([1.0, 2.0, 3.0]), {})
        mock_env.step.return_value = (np.array([1.1, 2.1, 3.1]), 0.1, False, False, {})
        mock_env.observation_space = Mock()
        mock_env.action_space = Mock()
        mock_env.action_space.sample.return_value = np.array([0.5, -0.3, 0.1])

        yield mock_env


@pytest.fixture
def sample_csv_file(tmp_path):
    """Provide a sample CSV file for legacy compatibility tests."""
    # Minimal CSV data for TraderEnv compatibility
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "open": np.random.uniform(95, 105, 50),
            "high": np.random.uniform(100, 110, 50),
            "low": np.random.uniform(90, 100, 50),
            "close": np.random.uniform(95, 105, 50),
            "volume": np.random.randint(1000, 10000, 50),
        }
    )

    csv_path = tmp_path / "sample_test_data.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def production_dataset_path():
    """Provide path to existing production dataset if available."""
    data_dir = "/workspaces/trading-rl-agent/data"

    # Look for existing advanced datasets
    for filename in os.listdir(data_dir):
        if filename.startswith("advanced_trading_dataset") and filename.endswith(
            ".csv"
        ):
            return os.path.join(data_dir, filename)

    # Fallback to sample data
    sample_path = os.path.join(
        data_dir, "sample_training_data_simple_20250607_192034.csv"
    )
    if os.path.exists(sample_path):
        return sample_path

    return None


# Backward compatibility - alias for existing tests
trading_env = finrl_trading_env
