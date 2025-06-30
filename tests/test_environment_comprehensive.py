"""
Comprehensive test suite for trading environment interactions.
Tests all environment functionality including initialization, reset, step, and edge cases.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import gymnasium as gym
import numpy as np
import pandas as pd
import pytest
import torch

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def comprehensive_market_data(tmp_path):
    """Generate comprehensive market data for thorough testing."""
    # Create varied market conditions
    data = []

    # Bull market phase
    bull_prices = [100]
    for i in range(100):
        change = np.random.normal(0.001, 0.02)  # Positive drift
        bull_prices.append(bull_prices[-1] * (1 + change))

    # Bear market phase
    bear_prices = [bull_prices[-1]]
    for i in range(100):
        change = np.random.normal(-0.001, 0.02)  # Negative drift
        bear_prices.append(bear_prices[-1] * (1 + change))

    # Sideways market phase
    sideways_prices = [bear_prices[-1]]
    for i in range(100):
        change = np.random.normal(0, 0.01)  # No drift, low volatility
        sideways_prices.append(sideways_prices[-1] * (1 + change))

    all_prices = bull_prices + bear_prices[1:] + sideways_prices[1:]

    df = pd.DataFrame(
        {
            "open": all_prices,
            "high": [p * (1 + abs(np.random.normal(0, 0.005))) for p in all_prices],
            "low": [p * (1 - abs(np.random.normal(0, 0.005))) for p in all_prices],
            "close": all_prices,
            "volume": np.random.randint(1000, 50000, len(all_prices)),
        }
    )

    csv_path = tmp_path / "comprehensive_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestTradingEnvironmentComprehensive:
    """Comprehensive test suite for trading environments."""

    @pytest.fixture
    def environment_configs(self):
        """Provide various environment configurations for testing."""
        return {
            "basic": {
                "window_size": 20,
                "initial_balance": 10000,
                "transaction_cost": 0.001,
            },
            "advanced": {
                "window_size": 50,
                "initial_balance": 50000,
                "transaction_cost": 0.0005,
                "enable_shorting": True,
                "max_position": 2.0,
                "normalize_observations": True,
            },
            "high_frequency": {
                "window_size": 10,
                "initial_balance": 100000,
                "transaction_cost": 0.0001,
                "max_trades_per_episode": 1000,
            },
        }

    def test_environment_initialization_comprehensive(
        self, comprehensive_market_data, environment_configs
    ):
        """Test environment initialization with various configurations."""
        try:
            from src.envs.trading_env import TradingEnv

            for config_name, config in environment_configs.items():
                config["dataset_paths"] = [comprehensive_market_data]

                env = TradingEnv(**config)

                # Test basic properties
                assert hasattr(env, "observation_space")
                assert hasattr(env, "action_space")
                assert env.window_size == config["window_size"]
                assert env.initial_balance == config["initial_balance"]

                # Test observation space
                obs_space = env.observation_space
                assert isinstance(obs_space, (gym.spaces.Box, gym.spaces.Dict))

                # Test action space
                action_space = env.action_space
                assert isinstance(action_space, (gym.spaces.Box, gym.spaces.Discrete))

                print(f"✅ Environment initialization test passed for {config_name}")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    def test_environment_reset_comprehensive(
        self, comprehensive_market_data, environment_configs
    ):
        """Test environment reset functionality comprehensively."""
        try:
            from src.envs.trading_env import TradingEnv

            config = environment_configs["basic"]
            config["dataset_paths"] = [comprehensive_market_data]
            env = TradingEnv(**config)

            # Test multiple resets
            for i in range(5):
                obs, info = env.reset()

                # Validate observation
                assert obs is not None
                assert isinstance(obs, (np.ndarray, dict))

                # Validate info
                assert isinstance(info, dict)

                # Test with different seeds
                obs_seed, info_seed = env.reset(seed=i)
                assert obs_seed is not None

                print(f"✅ Reset test {i+1} passed")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    @pytest.mark.parametrize("action_type", ["buy", "sell", "hold"])
    def test_environment_step_comprehensive(
        self, comprehensive_market_data, environment_configs, action_type
    ):
        """Test environment step functionality with different action types."""
        try:
            from src.envs.trading_env import TradingEnv

            config = environment_configs["basic"]
            config["dataset_paths"] = [comprehensive_market_data]
            env = TradingEnv(**config)

            obs, info = env.reset()

            # Define actions based on type
            action_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}

            action = action_map[action_type]
            if hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0:
                action = np.array([action])

            # Take step
            next_obs, reward, terminated, truncated, step_info = env.step(action)

            # Validate step output
            assert next_obs is not None
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(step_info, dict)

            # Validate observation consistency
            if isinstance(obs, np.ndarray) and isinstance(next_obs, np.ndarray):
                assert obs.shape == next_obs.shape

            print(f"✅ Step test passed for {action_type} action")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    def test_environment_episode_completion(
        self, comprehensive_market_data, environment_configs
    ):
        """Test complete episode execution."""
        try:
            from src.envs.trading_env import TradingEnv

            config = environment_configs["basic"]
            config["dataset_paths"] = [comprehensive_market_data]
            env = TradingEnv(**config)

            obs, info = env.reset()

            total_reward = 0
            steps = 0
            max_steps = 100

            while steps < max_steps:
                # Random action
                if hasattr(env.action_space, "sample"):
                    action = env.action_space.sample()
                else:
                    action = np.random.uniform(-1, 1)

                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1

                if terminated or truncated:
                    break

            assert steps > 0
            assert isinstance(total_reward, (int, float))

            print(
                f"✅ Episode completed in {steps} steps with total reward: {total_reward:.4f}"
            )

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    def test_environment_observation_consistency(
        self, comprehensive_market_data, environment_configs
    ):
        """Test observation consistency across steps."""
        try:
            from src.envs.trading_env import TradingEnv

            config = environment_configs["basic"]
            config["dataset_paths"] = [comprehensive_market_data]
            env = TradingEnv(**config)

            obs, _ = env.reset()

            observations = [obs]

            # Collect multiple observations
            for _ in range(10):
                action = 0.0  # Hold action
                if (
                    hasattr(env.action_space, "shape")
                    and len(env.action_space.shape) > 0
                ):
                    action = np.array([action])

                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
                observations.append(obs)

            # Check consistency
            if len(observations) > 1:
                first_obs = observations[0]
                for obs in observations[1:]:
                    if isinstance(first_obs, np.ndarray):
                        assert obs.shape == first_obs.shape
                    elif isinstance(first_obs, dict):
                        assert obs.keys() == first_obs.keys()

            print(
                f"✅ Observation consistency test passed with {len(observations)} observations"
            )

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    def test_environment_edge_cases(self, tmp_path):
        """Test environment behavior with edge cases."""
        try:
            from src.envs.trading_env import TradingEnv

            # Test with minimal data
            minimal_data = pd.DataFrame(
                {
                    "open": [100.0, 101.0],
                    "high": [105.0, 106.0],
                    "low": [95.0, 96.0],
                    "close": [101.0, 102.0],
                    "volume": [1000, 1100],
                }
            )

            csv_path = tmp_path / "minimal_data.csv"
            minimal_data.to_csv(csv_path, index=False)

            # Test with window size larger than data
            with pytest.raises((ValueError, IndexError)):
                env = TradingEnv(
                    dataset_paths=[str(csv_path)],
                    window_size=10,  # Larger than data
                    initial_balance=10000,
                )

            print("✅ Edge case test passed - properly handles insufficient data")

            # Test with extreme action values
            config = {
                "dataset_paths": [str(csv_path)],
                "window_size": 1,
                "initial_balance": 10000,
            }

            env = TradingEnv(**config)
            obs, _ = env.reset()

            # Test extreme actions
            extreme_actions = [float("inf"), float("-inf"), np.nan]

            for extreme_action in extreme_actions:
                try:
                    action = extreme_action
                    if (
                        hasattr(env.action_space, "shape")
                        and len(env.action_space.shape) > 0
                    ):
                        action = np.array([extreme_action])

                    # This should either clip the action or raise an appropriate error
                    obs, reward, terminated, truncated, info = env.step(action)

                    # If it doesn't raise an error, check that reward is valid
                    assert not np.isnan(reward)
                    assert not np.isinf(reward)

                except (ValueError, TypeError):
                    # It's okay if extreme actions raise appropriate errors
                    pass

            print("✅ Edge case test passed - properly handles extreme actions")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    def test_environment_performance(self, comprehensive_market_data, memory_monitor):
        """Test environment performance and memory usage."""
        try:
            from src.envs.trading_env import TradingEnv

            config = {
                "dataset_paths": [comprehensive_market_data],
                "window_size": 20,
                "initial_balance": 10000,
            }

            env = TradingEnv(**config)

            # Performance test - measure time for multiple episodes
            import time

            start_time = time.time()

            for episode in range(5):
                obs, _ = env.reset()

                for step in range(50):
                    action = np.random.uniform(-1, 1)
                    if (
                        hasattr(env.action_space, "shape")
                        and len(env.action_space.shape) > 0
                    ):
                        action = np.array([action])

                    obs, reward, terminated, truncated, info = env.step(action)

                    if terminated or truncated:
                        break

            end_time = time.time()
            total_time = end_time - start_time

            # Should complete 5 episodes in reasonable time
            assert (
                total_time < 30
            ), f"Performance test failed: took {total_time:.2f} seconds"

            # Memory usage should be reasonable
            current_memory = memory_monitor["current"]()
            memory_increase = current_memory - memory_monitor["initial"]

            assert (
                memory_increase < 500
            ), f"Memory usage too high: {memory_increase:.2f} MB"

            print(
                f"✅ Performance test passed: {total_time:.2f}s, {memory_increase:.2f}MB"
            )

        except ImportError:
            pytest.skip("TradingEnv not available for testing")


class TestEnvironmentIntegration:
    """Test environment integration with other components."""

    def test_environment_agent_interface(self, sample_csv_path, mock_agent):
        """Test environment compatibility with agents."""
        try:
            from src.envs.trading_env import TradingEnv

            env = TradingEnv(
                dataset_paths=[sample_csv_path], window_size=10, initial_balance=10000
            )

            obs, _ = env.reset()

            # Test agent-environment interface
            action = mock_agent.select_action(obs)

            # Action should be compatible with environment
            if hasattr(env.action_space, "contains"):
                # Convert to proper format if needed
                if (
                    hasattr(env.action_space, "shape")
                    and len(env.action_space.shape) > 0
                ):
                    if np.isscalar(action):
                        action = np.array([action])

                # Some action spaces might not have contains method implemented
                try:
                    is_valid = env.action_space.contains(action)
                    if not is_valid:
                        # Clip action to valid range
                        if hasattr(env.action_space, "low") and hasattr(
                            env.action_space, "high"
                        ):
                            action = np.clip(
                                action, env.action_space.low, env.action_space.high
                            )
                except:
                    pass  # Some spaces might not support contains check

            # Take step with agent action
            next_obs, reward, terminated, truncated, info = env.step(action)

            assert next_obs is not None
            assert isinstance(reward, (int, float))

            print("✅ Environment-agent interface test passed")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    def test_environment_model_integration(self, sample_csv_path, mock_model):
        """Test environment integration with predictive models."""
        try:
            from src.envs.trading_env import TradingEnv

            env = TradingEnv(
                dataset_paths=[sample_csv_path], window_size=10, initial_balance=10000
            )

            obs, _ = env.reset()

            # Use model to make prediction based on observation
            if isinstance(obs, np.ndarray):
                prediction = mock_model.predict(obs.reshape(1, -1))
            else:
                # Handle dict observations
                prediction = mock_model.predict(np.array([1.0]))

            # Use prediction to make trading decision
            action = np.tanh(prediction[0])  # Convert prediction to action

            if hasattr(env.action_space, "shape") and len(env.action_space.shape) > 0:
                action = np.array([action])

            # Execute action
            next_obs, reward, terminated, truncated, info = env.step(action)

            assert next_obs is not None
            assert isinstance(reward, (int, float))

            print("✅ Environment-model integration test passed")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")


class TestEnvironmentStressTests:
    """Stress tests for environment robustness."""

    @pytest.mark.slow
    def test_long_episode_stability(self, comprehensive_market_data):
        """Test environment stability over long episodes."""
        try:
            from src.envs.trading_env import TradingEnv

            env = TradingEnv(
                dataset_paths=[comprehensive_market_data],
                window_size=20,
                initial_balance=10000,
            )

            obs, _ = env.reset()

            # Run for many steps
            for step in range(1000):
                action = np.random.uniform(-1, 1)
                if (
                    hasattr(env.action_space, "shape")
                    and len(env.action_space.shape) > 0
                ):
                    action = np.array([action])

                obs, reward, terminated, truncated, info = env.step(action)

                # Check for numerical stability
                if isinstance(obs, np.ndarray):
                    assert not np.any(
                        np.isnan(obs)
                    ), f"NaN in observation at step {step}"
                    assert not np.any(
                        np.isinf(obs)
                    ), f"Inf in observation at step {step}"

                assert not np.isnan(reward), f"NaN reward at step {step}"
                assert not np.isinf(reward), f"Inf reward at step {step}"

                if terminated or truncated:
                    break

            print(f"✅ Long episode stability test passed - ran {step+1} steps")

        except ImportError:
            pytest.skip("TradingEnv not available for testing")

    @pytest.mark.memory
    def test_memory_leak_prevention(self, comprehensive_market_data, memory_monitor):
        """Test that environment doesn't have memory leaks."""
        try:
            from src.envs.trading_env import TradingEnv

            initial_memory = memory_monitor["initial"]

            # Create and destroy many environments
            for i in range(10):
                env = TradingEnv(
                    dataset_paths=[comprehensive_market_data],
                    window_size=20,
                    initial_balance=10000,
                )

                # Run short episode
                obs, _ = env.reset()
                for _ in range(10):
                    action = np.random.uniform(-1, 1)
                    if (
                        hasattr(env.action_space, "shape")
                        and len(env.action_space.shape) > 0
                    ):
                        action = np.array([action])

                    obs, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break

                # Explicitly delete environment
                del env

            # Check memory usage
            final_memory = memory_monitor["current"]()
            memory_increase = final_memory - initial_memory

            # Memory increase should be minimal
            assert (
                memory_increase < 100
            ), f"Potential memory leak: {memory_increase:.2f} MB increase"

            print(
                f"✅ Memory leak test passed - memory increase: {memory_increase:.2f} MB"
            )

        except ImportError:
            pytest.skip("TradingEnv not available for testing")
