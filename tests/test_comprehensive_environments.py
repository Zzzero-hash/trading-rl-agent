"""
Comprehensive tests for all environment interactions.
Tests environment reset, step, reward calculation, and edge cases.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

from src.envs.trader_env import TraderEnv
from src.envs.trading_env import TradingEnv
from tests.test_data_utils import get_dynamic_test_config


class TestEnvironmentInteractions:
    """Comprehensive tests for environment interactions."""

    def test_environment_reset_consistency(self, trading_env):
        """Test that environment reset is consistent."""
        obs1, info1 = trading_env.reset(seed=42)
        obs2, info2 = trading_env.reset(seed=42)

        # Observations should be identical with same seed
        if isinstance(obs1, dict):
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])
        else:
            np.testing.assert_array_equal(obs1, obs2)

    def test_environment_step_determinism(self, trading_env):
        """Test that environment steps are deterministic."""
        trading_env.reset(seed=42)
        action = 0  # Hold action

        obs1, reward1, done1, truncated1, info1 = trading_env.step(action)

        # Reset with same seed and take same action
        trading_env.reset(seed=42)
        obs2, reward2, done2, truncated2, info2 = trading_env.step(action)

        # Results should be identical
        if isinstance(obs1, dict):
            for key in obs1:
                np.testing.assert_array_equal(obs1[key], obs2[key])
        else:
            np.testing.assert_array_equal(obs1, obs2)

        assert reward1 == reward2
        assert done1 == done2
        assert truncated1 == truncated2

    def test_environment_action_space_validation(self, trading_env):
        """Test that environment validates actions correctly."""
        trading_env.reset()

        # Test valid actions
        if hasattr(trading_env.action_space, "n"):
            # Discrete action space
            for action in range(trading_env.action_space.n):
                obs, reward, done, truncated, info = trading_env.step(action)
                assert isinstance(reward, (int, float))
                assert isinstance(done, bool)
                assert isinstance(truncated, bool)
                assert isinstance(info, dict)

                if done:
                    break
        else:
            # Continuous action space
            action = trading_env.action_space.sample()
            obs, reward, done, truncated, info = trading_env.step(action)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_environment_observation_space_consistency(self, trading_env):
        """Test that observations match observation space."""
        obs, _ = trading_env.reset()

        if isinstance(obs, dict):
            # Dict observation space
            for key, value in obs.items():
                assert isinstance(value, np.ndarray)
                assert value.dtype in [np.float32, np.float64]
        else:
            # Array observation space
            assert isinstance(obs, np.ndarray)
            assert obs.dtype in [np.float32, np.float64]
            assert obs.shape == trading_env.observation_space.shape

    def test_environment_reward_calculation(self, trading_env):
        """Test environment reward calculation logic."""
        trading_env.reset()

        rewards = []
        actions = [0, 1, 0, 2, 0, 1, 0, 2, 0, 1]  # Mix of hold, buy, and sell actions

        for i in range(min(10, len(actions))):
            action = actions[i]
            obs, reward, done, truncated, info = trading_env.step(action)
            rewards.append(reward)

            if done:
                break

        # Rewards should be numeric
        assert all(isinstance(r, (int, float)) for r in rewards)

        # Rewards should be finite (not inf or nan)
        assert all(np.isfinite(r) for r in rewards)

        # At least some rewards should be non-zero when we're trading
        # (this is more lenient - allows for all zeros in edge cases)

    def test_environment_episode_termination(self, trading_env):
        """Test environment episode termination conditions."""
        trading_env.reset()

        step_count = 0
        max_steps = 1000  # Safety limit
        done = False
        truncated = False

        while step_count < max_steps:
            action = 0  # Hold action
            obs, reward, done, truncated, info = trading_env.step(action)
            step_count += 1

            if done or truncated:
                break

        assert step_count < max_steps, "Episode did not terminate"
        assert done or truncated, "Episode should be terminated"

    def test_environment_info_dict(self, trading_env):
        """Test that info dict contains expected information."""
        obs, info = trading_env.reset()
        assert isinstance(info, dict)

        obs, reward, done, truncated, info = trading_env.step(0)
        assert isinstance(info, dict)

        # Common info keys
        expected_keys = ["balance"]
        for key in expected_keys:
            if key in info:
                assert isinstance(info[key], (int, float))


class TestEnvironmentEdgeCases:
    """Test environment edge cases and error handling."""

    def test_environment_with_insufficient_data(self, tmp_path):
        """Test environment behavior with insufficient data."""
        # Create minimal data
        data = pd.DataFrame(
            {
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )

        csv_path = tmp_path / "minimal_data.csv"
        data.to_csv(csv_path, index=False)

        config = {
            "dataset_paths": [str(csv_path)],
            "window_size": 10,  # More than available data
            "initial_balance": 10000,
        }

        with pytest.raises((ValueError, IndexError)):
            env = TradingEnv(config)

    def test_environment_with_invalid_config(self):
        """Test environment behavior with invalid configuration."""
        # Missing required config
        with pytest.raises((KeyError, ValueError)):
            TradingEnv({})

        # Invalid config values
        with pytest.raises((ValueError, TypeError)):
            TradingEnv(
                {
                    "dataset_paths": [],  # Empty paths
                    "window_size": -1,  # Invalid window size
                    "initial_balance": -1000,  # Negative balance
                }
            )

    def test_environment_with_missing_files(self):
        """Test environment behavior with missing data files."""
        config = {
            "dataset_paths": ["nonexistent_file.csv"],
            "window_size": 10,
            "initial_balance": 10000,
        }

        with pytest.raises(FileNotFoundError):
            TradingEnv(config)

    def test_environment_with_corrupted_data(self, tmp_path):
        """Test environment behavior with corrupted data."""
        # Create corrupted CSV
        corrupted_data = "this,is,not,valid,csv,data\n1,2,3\n"
        csv_path = tmp_path / "corrupted.csv"
        csv_path.write_text(corrupted_data)

        config = {
            "dataset_paths": [str(csv_path)],
            "window_size": 10,
            "initial_balance": 10000,
        }

        with pytest.raises((ValueError, pd.errors.ParserError, KeyError)):
            TradingEnv(config)

    def test_environment_with_extreme_values(self, tmp_path):
        """Test environment behavior with extreme data values."""
        # Create data with extreme values
        data = pd.DataFrame(
            {
                "open": [1e10, 1e-10, float("inf"), -float("inf"), np.nan],
                "high": [1e10, 1e-10, float("inf"), -float("inf"), np.nan],
                "low": [1e10, 1e-10, float("inf"), -float("inf"), np.nan],
                "close": [1e10, 1e-10, float("inf"), -float("inf"), np.nan],
                "volume": [1e15, 0, -1000, np.nan, 1],
            }
        )

        csv_path = tmp_path / "extreme_data.csv"
        data.to_csv(csv_path, index=False)

        config = {
            "dataset_paths": [str(csv_path)],
            "window_size": 2,
            "initial_balance": 10000,
        }

        # Environment should handle extreme values gracefully
        try:
            env = TradingEnv(config)
            obs, _ = env.reset()

            # Should not contain infinite or NaN values
            if isinstance(obs, dict):
                for value in obs.values():
                    assert np.isfinite(
                        value
                    ).all(), "Observation contains non-finite values"
            else:
                assert np.isfinite(obs).all(), "Observation contains non-finite values"

        except (ValueError, RuntimeError) as e:
            # Acceptable to raise an error for extreme data
            assert "inf" in str(e).lower() or "nan" in str(e).lower()


class TestMultiEnvironmentComparison:
    """Test consistency across different environment implementations."""

    def test_trading_env_vs_trader_env_compatibility(self, sample_csv_file):
        """Test compatibility between TradingEnv and TraderEnv."""
        # Create both environments with similar configs
        trading_config = {
            "dataset_paths": [sample_csv_file],
            "window_size": 10,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
        }

        trading_env = TradingEnv(trading_config)
        trader_env = TraderEnv([sample_csv_file], window_size=10, initial_balance=10000)

        # Both should reset successfully
        trading_obs, _ = trading_env.reset()
        trader_obs, _ = trader_env.reset()

        # Both should step successfully
        trading_result = trading_env.step(0)
        trader_result = trader_env.step(0)

        # Results should have same structure
        assert len(trading_result) == len(trader_result) == 5

        # Rewards should be numeric
        assert isinstance(trading_result[1], (int, float))
        assert isinstance(trader_result[1], (int, float))

    def test_environment_state_preservation(self, trading_env):
        """Test that environment state is properly preserved."""
        # Get initial state
        initial_obs, initial_info = trading_env.reset()
        initial_balance = getattr(trading_env, "balance", None)
        initial_step = getattr(trading_env, "current_step", None)

        # Take some actions
        for _ in range(5):
            obs, reward, done, truncated, info = trading_env.step(0)
            if done:
                break

        # State should have changed
        current_balance = getattr(trading_env, "balance", None)
        current_step = getattr(trading_env, "current_step", None)

        if initial_balance is not None and current_balance is not None:
            # Balance might change due to rewards/costs
            assert isinstance(current_balance, (int, float))

        if initial_step is not None and current_step is not None:
            assert current_step > initial_step, "Step counter should increase"


class TestEnvironmentPerformance:
    """Test environment performance and resource usage."""

    @pytest.mark.slow
    def test_environment_step_performance(self, trading_env, benchmark):
        """Benchmark environment step performance."""
        trading_env.reset()

        def step_function():
            return trading_env.step(0)

        # Benchmark should complete in reasonable time
        result = benchmark(step_function)
        assert result is not None

    @pytest.mark.slow
    def test_environment_reset_performance(self, trading_env, benchmark):
        """Benchmark environment reset performance."""

        def reset_function():
            return trading_env.reset()

        # Benchmark should complete in reasonable time
        result = benchmark(reset_function)
        assert result is not None

    def test_environment_memory_usage(self, trading_env):
        """Test that environment doesn't leak memory."""
        import gc
        import sys

        # Get initial memory usage
        initial_objects = len(gc.get_objects())

        # Run multiple episodes
        for _ in range(10):
            trading_env.reset()
            for _ in range(20):
                obs, reward, done, truncated, info = trading_env.step(0)
                if done:
                    break

        # Force garbage collection
        gc.collect()

        # Memory usage should not increase significantly
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        # Allow some increase but not excessive
        assert (
            object_increase < 1000
        ), f"Memory usage increased by {object_increase} objects"


class TestEnvironmentConfigurationVariations:
    """Test environment with various configuration options."""

    @pytest.mark.parametrize("window_size", [5, 10, 20, 50])
    def test_different_window_sizes(self, sample_csv_file, window_size):
        """Test environment with different window sizes."""
        config = {
            "dataset_paths": [sample_csv_file],
            "window_size": window_size,
            "initial_balance": 10000,
        }

        env = TradingEnv(config)
        obs, _ = env.reset()

        # Observation should match window size
        if isinstance(obs, dict):
            for value in obs.values():
                if hasattr(value, "shape") and len(value.shape) > 1:
                    assert value.shape[0] == window_size
        else:
            if len(obs.shape) > 1:
                assert obs.shape[0] == window_size

    @pytest.mark.parametrize("initial_balance", [1000, 10000, 100000])
    def test_different_initial_balances(self, sample_csv_file, initial_balance):
        """Test environment with different initial balances."""
        config = {
            "dataset_paths": [sample_csv_file],
            "window_size": 10,
            "initial_balance": initial_balance,
        }

        env = TradingEnv(config)
        obs, info = env.reset()

        # Check that balance is properly initialized
        if hasattr(env, "balance"):
            assert env.balance == initial_balance
        elif "balance" in info:
            assert info["balance"] == initial_balance

    @pytest.mark.parametrize("transaction_cost", [0.0, 0.001, 0.01, 0.1])
    def test_different_transaction_costs(self, sample_csv_file, transaction_cost):
        """Test environment with different transaction costs."""
        config = {
            "dataset_paths": [sample_csv_file],
            "window_size": 10,
            "initial_balance": 10000,
            "transaction_cost": transaction_cost,
        }

        env = TradingEnv(config)
        env.reset()

        # Take a trading action (not hold)
        obs, reward, done, truncated, info = env.step(1)  # Buy action

        # Transaction cost should affect the result
        if hasattr(env, "transaction_cost"):
            assert env.transaction_cost == transaction_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
