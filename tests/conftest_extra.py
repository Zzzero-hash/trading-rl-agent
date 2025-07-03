"""
Comprehensive pytest configuration and fixtures for trading RL agent testing.
Provides fixtures for data management, environment setup, and test utilities.
"""

import asyncio
from collections.abc import Generator
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch
import warnings

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

# Import test utilities
from tests.test_data_utils import TestDataManager, get_dynamic_test_config

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Legacy compatibility import
try:
    from generate_sample_data import generate_sample_price_data
except ImportError:

    def generate_sample_price_data(
        symbol="TEST", days=30, start_price=100.0, volatility=0.01
    ):
        """Fallback sample data generator for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=days, freq="D")
        prices = [start_price]

        for _ in range(1, days):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))

        return pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                "close": prices,
                "volume": [np.random.randint(1000, 10000) for _ in prices],
            }
        )


# Global test data manager
_test_manager = None

# ============================================================================
# SESSION-LEVEL FIXTURES
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up global test environment configuration."""
    # Suppress specific warnings during testing
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*ray.*")
    warnings.filterwarnings("ignore", message=".*torch.*")

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Configure environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    yield

    # Cleanup after all tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]


@pytest.fixture(scope="session", autouse=True)
def setup_test_data():
    """Auto-setup test data for entire test session."""
    global _test_manager
    _test_manager = TestDataManager()
    yield _test_manager
    # Cleanup after all tests
    if _test_manager:
        _test_manager.cleanup()


@pytest.fixture(scope="session")
def temp_project_dir():
    """Create a temporary project directory for session-level tests."""
    temp_dir = tempfile.mkdtemp(prefix="trading_rl_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# DATA FIXTURES
# ============================================================================


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
def sample_price_data():
    """Generate sample price data for testing."""
    return generate_sample_price_data(
        symbol="TEST", days=100, start_price=100.0, volatility=0.02
    )


@pytest.fixture
def multi_symbol_data():
    """Generate multi-symbol test data."""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    data_frames = []

    for symbol in symbols:
        df = generate_sample_price_data(symbol=symbol, days=50, start_price=100.0)
        data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a temporary CSV file with ``TestDataManager``."""
    import pandas as pd

    from tests.test_data_utils import TestDataManager

    manager = TestDataManager(str(tmp_path))
    df = manager.generate_synthetic_dataset()
    df = df.drop(columns=["timestamp"])  # Keep numeric columns only
    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)
    manager.temp_files.append(file_path)

    yield str(file_path)
    manager.cleanup()


@pytest.fixture(scope="session")
def sample_csv_path_session(tmp_path_factory):
    """Session-scoped CSV path generated with ``TestDataManager``."""
    import pandas as pd

    from tests.test_data_utils import TestDataManager

    data_dir = tmp_path_factory.mktemp("data")
    manager = TestDataManager(str(data_dir))
    df = manager.generate_synthetic_dataset()
    df = df.drop(columns=["timestamp"])  # Numeric columns only
    file_path = data_dir / "sample.csv"
    df.to_csv(file_path, index=False)
    manager.temp_files.append(file_path)

    yield str(file_path)
    manager.cleanup()


@pytest.fixture
def large_dataset():
    """Generate a large dataset for performance testing."""
    return generate_sample_price_data(symbol="LARGE", days=1000, start_price=100.0)


@pytest.fixture
def noisy_data():
    """Generate noisy data with missing values and outliers for robustness testing."""
    df = generate_sample_price_data(symbol="NOISY", days=100)

    # Add missing values
    missing_indices = np.random.choice(
        df.index, size=int(0.05 * len(df)), replace=False
    )
    df.loc[missing_indices, "close"] = np.nan

    # Add outliers
    outlier_indices = np.random.choice(
        df.index, size=int(0.02 * len(df)), replace=False
    )
    df.loc[outlier_indices, "close"] *= np.random.choice(
        [0.1, 10.0], size=len(outlier_indices)
    )

    return df


# ============================================================================
# ENVIRONMENT FIXTURES
# ============================================================================


@pytest.fixture
def trading_env_config():
    """Provide basic trading environment configuration."""
    return {
        "window_size": 20,
        "initial_balance": 10000.0,
        "transaction_cost": 0.001,
        "max_position": 1.0,
        "enable_shorting": True,
        "normalize_observations": True,
    }


@pytest.fixture
def mock_trading_env():
    """Create a mock trading environment for testing."""
    env = Mock()
    env.reset.return_value = (np.random.random(20), {})
    env.step.return_value = (np.random.random(20), 1.0, False, False, {})
    env.observation_space = gym.spaces.Box(low=-1, high=1, shape=(20,))
    env.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    return env


# ============================================================================
# AGENT FIXTURES
# ============================================================================


@pytest.fixture
def td3_config():
    """Provide TD3 agent configuration for testing."""
    return {
        "learning_rate": 3e-4,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "tau": 0.005,
        "policy_delay": 2,
        "target_noise": 0.2,
        "noise_clip": 0.5,
        "hidden_layers": [64, 64],
    }


@pytest.fixture
def sac_config():
    """Provide SAC agent configuration for testing."""
    return {
        "learning_rate": 3e-4,
        "buffer_size": 10000,
        "batch_size": 64,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "auto_alpha": True,
        "target_entropy": "auto",
        "hidden_layers": [64, 64],
    }


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.select_action.return_value = np.array([0.5])
    agent.train.return_value = {"loss": 0.1, "q_value": 1.0}
    agent.save.return_value = None
    agent.load.return_value = None
    return agent


# ============================================================================
# MODEL FIXTURES
# ============================================================================


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.predict.return_value = np.array([0.1, 0.2, 0.3])
    model.train.return_value = {"loss": 0.05}
    model.eval.return_value = None
    return model


@pytest.fixture
def simple_nn_model():
    """Create a simple neural network model for testing."""
    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=32, output_size=1):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        def forward(self, x):
            return self.network(x)

    return SimpleModel()


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def dynamic_test_config():
    """Provide dynamic test configuration."""
    return get_dynamic_test_config()


@pytest.fixture
def sample_yaml_config(tmp_path):
    """Create a sample YAML configuration file."""
    config = {
        "model": {"type": "td3", "learning_rate": 0.001, "hidden_layers": [64, 64]},
        "environment": {"window_size": 20, "initial_balance": 10000},
        "training": {"total_timesteps": 10000, "eval_freq": 1000},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary directory with multiple config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Environment config
    env_config = {
        "dataset_paths": ["test_data.csv"],
        "window_size": 20,
        "initial_balance": 10000,
    }
    with open(config_dir / "env.yaml", "w") as f:
        yaml.dump(env_config, f)

    # Model config
    model_config = {
        "architecture": "td3",
        "learning_rate": 0.001,
        "hidden_layers": [64, 64],
    }
    with open(config_dir / "model.yaml", "w") as f:
        yaml.dump(model_config, f)

    # Training config
    train_config = {"total_timesteps": 10000, "eval_freq": 1000, "save_freq": 2000}
    with open(config_dir / "train.yaml", "w") as f:
        yaml.dump(train_config, f)

    return config_dir


# ============================================================================
# RAY FIXTURES
# ============================================================================


@pytest.fixture
def mock_ray():
    """Mock Ray for testing without actual Ray cluster."""
    with (
        patch("ray.init") as mock_init,
        patch("ray.shutdown") as mock_shutdown,
        patch("ray.is_initialized", return_value=False),
    ):
        yield {"init": mock_init, "shutdown": mock_shutdown}


@pytest.fixture(scope="session")
def ray_cluster():
    """Initialize Ray cluster for session-level tests."""
    try:
        import ray

        if not ray.is_initialized():
            ray.init(local_mode=True, ignore_reinit_error=True, log_to_driver=False)
        yield ray
    except ImportError:
        pytest.skip("Ray not installed")
    finally:
        try:
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def captured_logs():
    """Capture log output for testing."""
    from io import StringIO
    import logging

    log_capture_string = StringIO()
    ch = logging.StreamHandler(log_capture_string)
    ch.setLevel(logging.DEBUG)

    # Get the root logger
    logger = logging.getLogger()
    logger.addHandler(ch)

    yield log_capture_string

    logger.removeHandler(ch)


@pytest.fixture
def mock_datetime():
    """Mock datetime for time-dependent tests."""
    with patch("datetime.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        mock_dt.utcnow.return_value = datetime(2024, 1, 1, 12, 0, 0)
        yield mock_dt


@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after each test."""
    temp_files_before = set()
    try:
        import tempfile

        temp_files_before = set(os.listdir(tempfile.gettempdir()))
    except Exception:
        pass

    yield

    # Cleanup any new temp files
    try:
        temp_files_after = set(os.listdir(tempfile.gettempdir()))
        new_files = temp_files_after - temp_files_before

        for file in new_files:
            if file.startswith(("test_", "pytest_", "tmp_")):
                try:
                    file_path = Path(tempfile.gettempdir()) / file
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path, ignore_errors=True)
                except Exception:
                    pass
    except Exception:
        pass


# ============================================================================
# PERFORMANCE FIXTURES
# ============================================================================


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    try:
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        yield {
            "initial": initial_memory,
            "current": lambda: process.memory_info().rss / 1024 / 1024,
        }
    except ImportError:
        yield {"initial": 0, "current": lambda: 0}


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {"min_rounds": 5, "max_time": 1.0, "warmup": True, "warmup_iterations": 2}


# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================


@pytest.fixture
def integration_environment(sample_csv_path, trading_env_config):
    """Set up full integration test environment."""
    # Mock imports that might not be available
    mocks = {}

    try:
        from src.envs.finrl_trading_env import TradingEnv

        trading_env_config["dataset_paths"] = [sample_csv_path]
        env = TradingEnv(**trading_env_config)
        mocks["env"] = env
    except ImportError:
        mocks["env"] = mock_trading_env()

    yield mocks


@pytest.fixture
def end_to_end_setup(tmp_path, sample_price_data):
    """Set up end-to-end test environment with all components."""
    # Create test data file
    data_path = tmp_path / "e2e_data.csv"
    sample_price_data.to_csv(data_path, index=False)

    # Create config files
    configs = {}

    # Environment config
    env_config_path = tmp_path / "env_config.yaml"
    env_config = {
        "dataset_paths": [str(data_path)],
        "window_size": 10,
        "initial_balance": 10000,
    }
    with open(env_config_path, "w") as f:
        yaml.dump(env_config, f)
    configs["env"] = str(env_config_path)

    # Model config
    model_config_path = tmp_path / "model_config.yaml"
    model_config = {"architecture": "td3", "learning_rate": 0.001}
    with open(model_config_path, "w") as f:
        yaml.dump(model_config, f)
    configs["model"] = str(model_config_path)

    # Results directory
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    configs["results"] = str(results_dir)

    yield configs


# ============================================================================
# ERROR HANDLING FIXTURES
# ============================================================================


@pytest.fixture
def error_conditions():
    """Provide various error conditions for testing."""
    return {
        "file_not_found": lambda: FileNotFoundError("Test file not found"),
        "value_error": lambda: ValueError("Test value error"),
        "key_error": lambda: KeyError("Test key error"),
        "type_error": lambda: TypeError("Test type error"),
        "runtime_error": lambda: RuntimeError("Test runtime error"),
    }


# ============================================================================
# PARAMETRIZED FIXTURES
# ============================================================================


@pytest.fixture(params=[10, 50, 100])
def variable_window_size(request):
    """Parametrized fixture for different window sizes."""
    return request.param


@pytest.fixture(params=[1000.0, 10000.0, 100000.0])
def variable_balance(request):
    """Parametrized fixture for different initial balances."""
    return request.param


@pytest.fixture(params=["td3", "sac"])
def agent_type(request):
    """Parametrized fixture for different agent types."""
    return request.param


# ============================================================================
# ASYNC FIXTURES
# ============================================================================


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# COMPATIBILITY FIXTURES (Legacy Support)
# ============================================================================

# Keep legacy fixture names for backward compatibility
sample_csv_path_legacy = sample_csv_path_session
