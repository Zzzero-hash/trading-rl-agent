"""
Robust pytest configuration and fixtures for Trading RL Agent tests.

Provides:
1. Test isolation and reproducibility
2. Environment-agnostic test configurations
3. Comprehensive test data management
4. Proper cleanup and resource management
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _pytest.config import Config

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def _setup_test_environment():
    """Setup robust test environment with proper isolation."""
    # Set test environment variables
    os.environ["TRADING_RL_AGENT_ENVIRONMENT"] = "test"
    os.environ["TRADING_RL_AGENT_DEBUG"] = "false"
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Set thread limits for consistent performance
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Create test directories
    test_dirs = ["test_data", "test_logs", "test_models", "test_output"]
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)

    logger.info("Test environment setup complete")

    yield

    # Cleanup after all tests
    logger.info("Cleaning up test environment")
    for dir_name in test_dirs:
        test_dir = Path(dir_name)
        if test_dir.exists():
            for file_path in test_dir.rglob("*"):
                if file_path.is_file():
                    file_path.unlink()
            for dir_path in reversed(list(test_dir.rglob("*"))):
                if dir_path.is_dir():
                    dir_path.rmdir()
            test_dir.rmdir()


@pytest.fixture(scope="session")
def test_data_manager():
    """Provide a robust test data manager for consistent test data."""
    from tests.unit.test_data_utils import TestDataManager

    # Use temporary directory for test data
    temp_dir = tempfile.mkdtemp(prefix="trading_rl_test_")
    manager = TestDataManager(temp_dir)

    yield manager

    # Cleanup
    manager.cleanup()
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def finrl_sample_data(_test_data_manager):
    """Provide consistent sample data in FinRL format for testing."""
    # Use fixed seed for reproducibility
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
            # Generate realistic OHLCV data with fixed seed
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
                },
            )

    return pd.DataFrame(data)


@pytest.fixture
def finrl_trading_env(finrl_sample_data):
    """Provide a robust FinRL-based trading environment for testing."""
    try:
        from trading_rl_agent.envs.finrl_trading_env import HybridFinRLEnv

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
        from tests.unit.test_data_utils import get_dynamic_test_config
        from trading_rl_agent.envs.finrl_trading_env import TradingEnv

        cfg = get_dynamic_test_config(
            {
                "include_features": False,
                "continuous_actions": False,
                "reward_type": "profit",
            },
        )
        env = TradingEnv(cfg)

        yield env

        manager = cfg.get("_test_manager")
        if manager:
            manager.cleanup()
        if hasattr(env, "close"):
            env.close()


@pytest.fixture
def sample_csv_file(test_data_manager):
    """Provide a consistent sample CSV file using TestDataManager utilities."""
    df = test_data_manager.generate_synthetic_dataset()
    df = df.drop(columns=["timestamp"])  # TraderEnv expects numeric columns
    dataset_path = Path(test_data_manager.data_dir) / "sample_test_data.csv"
    df.to_csv(dataset_path, index=False)
    test_data_manager.temp_files.append(dataset_path)

    return str(dataset_path)


@pytest.fixture
def production_dataset_path():
    """Provide path to existing production dataset if available."""
    data_dir = Path("/workspaces/trading-rl-agent/data")

    # Look for existing advanced datasets
    for p in data_dir.iterdir():
        if p.name.startswith("advanced_trading_dataset") and p.name.endswith(".csv"):
            return str(p)

    # Fallback to sample data
    sample_path = data_dir / "sample_training_data_simple_20250607_192034.csv"
    if sample_path.exists():
        return str(sample_path)

    return None


@pytest.fixture
def basic_trading_env(sample_csv_file):
    """Return a minimal TradingEnv using sample_csv_file."""
    from trading_rl_agent.envs.finrl_trading_env import TradingEnv

    cfg = {"dataset_paths": sample_csv_file, "reward_type": "profit"}
    env = TradingEnv(cfg)
    yield env
    if hasattr(env, "close"):
        env.close()


# Backward compatibility - alias for existing tests
trading_env = finrl_trading_env


@pytest.fixture
def mock_config():
    """Provide a mock configuration for testing."""
    return {
        "environment": "test",
        "debug": False,
        "data": {
            "primary_source": "synthetic",
            "symbols": ["AAPL", "GOOGL", "MSFT"],
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
        },
        "model": {
            "type": "cnn_lstm",
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "agent": {
            "agent_type": "sac",
            "gamma": 0.99,
            "tau": 0.005,
        },
        "risk": {
            "max_position_size": 0.1,
            "max_drawdown": 0.2,
        },
        "execution": {
            "broker": "mock",
            "paper_trading": True,
        },
        "monitoring": {
            "log_level": "INFO",
            "save_models": False,
        },
    }


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="trading_rl_test_")
    yield temp_dir
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_api_responses():
    """Provide mock API responses for testing."""
    return {
        "alpaca": {
            "account": {"cash": "100000", "buying_power": "100000"},
            "positions": [],
            "orders": [],
        },
        "alphavantage": {
            "Time Series (Daily)": {
                "2024-01-01": {
                    "1. open": "150.00",
                    "2. high": "155.00",
                    "3. low": "148.00",
                    "4. close": "152.00",
                    "5. volume": "1000000",
                }
            }
        },
    }


def pytest_configure(config: Config):
    """Configure pytest with custom markers and settings."""
    # Register custom markers
    markers = [
        "fast: mark as fast test (<1 second)",
        "slow: mark as slow test (>1 second)",
        "very_slow: mark as very slow test (>5 seconds)",
        "core: mark as core infrastructure test",
        "data: mark as data pipeline test",
        "model: mark as model architecture test",
        "training: mark as training pipeline test",
        "risk: mark as risk management test",
        "portfolio: mark as portfolio management test",
        "cli: mark as CLI interface test",
    ]

    for marker in markers:
        config.addinivalue_line("markers", marker)


def pytest_collection_modifyitems(_config: Config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add default markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.fast)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.slow)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.very_slow)
        elif "smoke" in str(item.fspath):
            item.add_marker(pytest.mark.smoke)
            item.add_marker(pytest.mark.fast)
