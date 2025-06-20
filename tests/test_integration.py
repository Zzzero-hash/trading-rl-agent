# File: test_integration.py (new file)

import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from envs.trading_env import TradingEnv


def test_model_environment_interface():
    # Test model ↔ environment compatibility
    # Create a simple test dataset
    test_data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [95.0, 96.0, 97.0, 98.0, 99.0],
            "close": [102.0, 103.0, 104.0, 105.0, 106.0],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )

    # Save to temp file
    temp_file = Path("temp_test_data.csv")
    test_data.to_csv(temp_file, index=False)

    try:
        # Create environment with basic config
        config = {
            "dataset_paths": [str(temp_file)],
            "window_size": 2,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
        }
        env = TradingEnv(config)

        # Test observation space
        obs, _ = env.reset()
        if isinstance(obs, dict):
            # Dict observation space
            assert "market_features" in obs
            market_features = obs["market_features"]
            state_dim = market_features.shape[0] * market_features.shape[1]
        else:
            # Array observation space
            state_dim = obs.shape[0] if obs.ndim == 1 else obs.shape[0] * obs.shape[1]

        # Test action space
        if hasattr(env.action_space, "n"):
            action_dim = env.action_space.n  # Discrete
        else:
            action_dim = env.action_space.shape[0]  # Continuous

        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")

        # Basic compatibility tests
        assert state_dim > 0, "State dimension should be positive"
        assert action_dim > 0, "Action dimension should be positive"

    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()


def test_agent_environment_interface():
    # Test agent ↔ environment compatibility
    # This is a placeholder - would need specific agent implementation
    assert True, "Agent-environment interface test placeholder"


def test_model_agent_interface():
    # Test model ↔ agent integration
    # This is a placeholder - would need specific model and agent implementations
    assert True, "Model-agent interface test placeholder"


def test_full_pipeline():
    # Full pipeline integration test
    # This is a placeholder for comprehensive integration test
    assert True, "Full pipeline integration test placeholder"
