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
        # Fallback to simple TradingEnv using synthetic data
        from src.envs.finrl_trading_env import TradingEnv
        from tests.test_data_utils import get_dynamic_test_config

        cfg = get_dynamic_test_config(
            {
                "include_features": False,
                "continuous_actions": False,
                "reward_type": "profit",
            }
        )
        env = TradingEnv(cfg)

        yield env

        manager = cfg.get("_test_manager")
        if manager:
            manager.cleanup()
        if hasattr(env, "close"):
            env.close()


@pytest.fixture
def sample_csv_file(tmp_path):
    """Provide a sample CSV file using ``TestDataManager`` utilities."""
    import pandas as pd

    from tests.test_data_utils import TestDataManager

    manager = TestDataManager(str(tmp_path))
    df = manager.generate_synthetic_dataset()
    df = df.drop(columns=["timestamp"])  # TraderEnv expects numeric columns
    dataset_path = tmp_path / "sample_test_data.csv"
    df.to_csv(dataset_path, index=False)
    manager.temp_files.append(dataset_path)

    yield str(dataset_path)
    manager.cleanup()


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
