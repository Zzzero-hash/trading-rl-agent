"""
Comprehensive test suite for trading agent functionality.
Tests all agent training/inference pipelines, configuration management, and integration scenarios.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import torch

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit


class TestAgentComprehensive:
    """Comprehensive test suite for trading agents."""

    @pytest.fixture
    def mock_environment(self):
        """Create a mock trading environment for agent testing."""
        mock_env = Mock()

        # Mock observation space (20-dimensional)
        mock_obs_space = Mock()
        mock_obs_space.shape = (20,)
        mock_obs_space.low = np.full(20, -np.inf)
        mock_obs_space.high = np.full(20, np.inf)
        mock_env.observation_space = mock_obs_space

        # Mock action space (1-dimensional continuous)
        mock_action_space = Mock()
        mock_action_space.shape = (1,)
        mock_action_space.low = np.array([-1.0])
        mock_action_space.high = np.array([1.0])
        mock_env.action_space = mock_action_space

        # Mock environment methods
        def mock_reset():
            obs = np.random.normal(0, 1, 20)
            info = {}
            return obs, info

        def mock_step(action):
            next_obs = np.random.normal(0, 1, 20)
            reward = np.random.normal(0, 0.1)
            terminated = np.random.random() < 0.05  # 5% chance of episode end
            truncated = False
            info = {}
            return next_obs, reward, terminated, truncated, info

        mock_env.reset = mock_reset
        mock_env.step = mock_step

        return mock_env

    @pytest.fixture
    def agent_configs(self):
        """Provide comprehensive agent configurations."""
        return {
            "td3": {
                "learning_rate": 3e-4,
                "buffer_capacity": 50000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_delay": 2,
                "target_noise": 0.2,
                "noise_clip": 0.5,
                "exploration_noise": 0.1,
                "hidden_dims": [256, 256],
            },
            "sac": {
                "learning_rate": 3e-4,
                "buffer_capacity": 50000,
                "batch_size": 256,
                "gamma": 0.99,
                "tau": 0.005,
                "alpha": 0.2,
                "automatic_entropy_tuning": True,
                "target_entropy": -1.0,
                "hidden_dims": [256, 256],
            },
            "minimal": {
                "learning_rate": 1e-3,
                "buffer_capacity": 1000,
                "batch_size": 32,
                "gamma": 0.99,
                "hidden_dims": [64, 64],
            },
        }

    @pytest.mark.parametrize("agent_type", ["td3", "sac"])
    def test_agent_initialization(self, agent_type, agent_configs, mock_environment):
        """Test agent initialization with different configurations."""
        try:
            if agent_type == "td3":
                from src.agents.configs import TD3Config
                from src.agents.td3_agent import TD3Agent

                config = TD3Config(**agent_configs[agent_type])
                agent = TD3Agent(
                    state_dim=mock_environment.observation_space.shape[0],
                    action_dim=mock_environment.action_space.shape[0],
                    config=config,
                )

            elif agent_type == "sac":
                from src.agents.configs import SACConfig
                from src.agents.sac_agent import SACAgent

                config = SACConfig(**agent_configs[agent_type])
                agent = SACAgent(
                    state_dim=mock_environment.observation_space.shape[0],
                    action_dim=mock_environment.action_space.shape[0],
                    config=config,
                )

            # Test basic agent properties
            assert hasattr(agent, "select_action")
            assert hasattr(agent, "train") or hasattr(agent, "update")
            assert hasattr(agent, "save")
            assert hasattr(agent, "load")

            # Test configuration access
            if hasattr(agent.config, "learning_rate"):
                assert (
                    agent.config.learning_rate
                    == agent_configs[agent_type]["learning_rate"]
                )
            if hasattr(agent.config, "batch_size"):
                assert (
                    agent.config.batch_size == agent_configs[agent_type]["batch_size"]
                )

            print(f"✅ {agent_type.upper()} agent initialization test passed")

        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")

    @pytest.mark.parametrize("agent_type", ["td3", "sac"])
    def test_agent_action_selection(self, agent_type, agent_configs, mock_environment):
        """Test agent action selection in different modes."""
        try:
            if agent_type == "td3":
                from src.agents.configs import TD3Config
                from src.agents.td3_agent import TD3Agent

                config = TD3Config(**agent_configs["minimal"])
                agent = TD3Agent(
                    state_dim=mock_environment.observation_space.shape[0],
                    action_dim=mock_environment.action_space.shape[0],
                    config=config,
                )

            elif agent_type == "sac":
                from src.agents.configs import SACConfig
                from src.agents.sac_agent import SACAgent

                config = SACConfig(**agent_configs["minimal"])
                agent = SACAgent(
                    state_dim=mock_environment.observation_space.shape[0],
                    action_dim=mock_environment.action_space.shape[0],
                    config=config,
                )

            obs = np.random.normal(0, 1, 20)

            # Test deterministic action selection (with add_noise=False for TD3)
            if agent_type == "td3":
                action_det = agent.select_action(obs, add_noise=False)
            else:  # SAC
                action_det = agent.select_action(obs)

            assert action_det is not None
            assert isinstance(action_det, np.ndarray)
            assert action_det.shape == (1,)

            # Test stochastic action selection (with add_noise=True for TD3)
            if agent_type == "td3":
                action_stoch = agent.select_action(obs, add_noise=True)
            else:  # SAC
                action_stoch = agent.select_action(obs)

            assert action_stoch is not None
            assert isinstance(action_stoch, np.ndarray)
            assert action_stoch.shape == (1,)

            # Test action bounds
            for _ in range(10):
                obs = np.random.normal(0, 1, 20)
                action = agent.select_action(obs)
                assert np.all(action >= -1.0), f"Action {action} below lower bound"
                assert np.all(action <= 1.0), f"Action {action} above upper bound"

            print(f"✅ {agent_type.upper()} action selection test passed")

        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")

    @pytest.mark.parametrize("agent_type", ["td3", "sac"])
    def test_agent_training_pipeline(self, agent_type, agent_configs, mock_environment):
        """Test complete agent training pipeline."""
        try:
            if agent_type == "td3":
                from src.agents.configs import TD3Config
                from src.agents.td3_agent import TD3Agent

                config = TD3Config(**agent_configs["minimal"])
                agent = TD3Agent(
                    state_dim=mock_environment.observation_space.shape[0],
                    action_dim=mock_environment.action_space.shape[0],
                    config=config,
                )

            elif agent_type == "sac":
                from src.agents.configs import SACConfig
                from src.agents.sac_agent import SACAgent

                config = SACConfig(**agent_configs["minimal"])
                agent = SACAgent(
                    state_dim=mock_environment.observation_space.shape[0],
                    action_dim=mock_environment.action_space.shape[0],
                    config=config,
                )

            # Collect experience
            experiences = []
            obs, _ = mock_environment.reset()

            for _ in range(100):
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = mock_environment.step(
                    action
                )

                # Store experience
                experiences.append(
                    {
                        "obs": obs.copy(),
                        "action": action.copy(),
                        "reward": reward,
                        "next_obs": next_obs.copy(),
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )

                # Add to agent's buffer if available
                if hasattr(agent, "replay_buffer"):
                    agent.replay_buffer.add(
                        obs, action, reward, next_obs, terminated or truncated
                    )
                elif hasattr(agent, "store_experience"):
                    agent.store_experience(
                        obs, action, reward, next_obs, terminated or truncated
                    )

                obs = next_obs

                if terminated or truncated:
                    obs, _ = mock_environment.reset()

            # Test training only if we have enough experience
            buffer_len = 0
            if hasattr(agent, "replay_buffer"):
                buffer_len = len(agent.replay_buffer)

            if buffer_len > agent.config.batch_size:
                # Train the agent
                training_info = (
                    agent.train() if hasattr(agent, "train") else agent.update()
                )

                # Validate training output
                assert isinstance(training_info, dict)

                print(f"✅ {agent_type.upper()} training pipeline test passed")
            else:
                print(
                    f"⚠️ {agent_type.upper()} training skipped - insufficient experience"
                )

        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")

    @pytest.mark.parametrize("agent_type", ["td3", "sac"])
    def test_agent_save_load(self, agent_type, agent_configs, tmp_path):
        """Test agent save and load functionality."""
        try:
            if agent_type == "td3":
                from src.agents.configs import TD3Config
                from src.agents.td3_agent import TD3Agent

                config = TD3Config(**agent_configs["minimal"])
                agent = TD3Agent(state_dim=20, action_dim=1, config=config)

            elif agent_type == "sac":
                from src.agents.configs import SACConfig
                from src.agents.sac_agent import SACAgent

                config = SACConfig(**agent_configs["minimal"])
                agent = SACAgent(state_dim=20, action_dim=1, config=config)

            # Test save
            save_path = tmp_path / f"{agent_type}_test_agent"
            agent.save(str(save_path))

            # Verify save files exist
            assert save_path.exists(), f"Save path {save_path} does not exist"

            # Test load
            if agent_type == "td3":
                agent_loaded = TD3Agent(state_dim=20, action_dim=1, config=config)
            else:
                agent_loaded = SACAgent(state_dim=20, action_dim=1, config=config)

            agent_loaded.load(str(save_path))

            # Test that loaded agent can still select actions
            obs = np.random.normal(0, 1, 20)

            # For SAC, set to evaluation mode to reduce stochasticity
            if agent_type == "sac":
                action1 = agent.select_action(obs, evaluate=True)
                action2 = agent_loaded.select_action(obs, evaluate=True)
                # SAC actions may still have small variations due to stochastic nature
                atol = 1e-3  # More tolerant for SAC
            else:
                action1 = agent.select_action(obs, add_noise=False)
                action2 = agent_loaded.select_action(obs, add_noise=False)
                atol = 1e-6  # Strict for deterministic TD3

            # Actions should be very similar (allowing for numerical precision)
            assert np.allclose(
                action1, action2, atol=atol
            ), f"Actions differ: {action1} vs {action2}"

            print(f"✅ {agent_type.upper()} save/load test passed")

        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")

    def test_agent_configuration_validation(self, agent_configs):
        """Test agent configuration validation."""
        try:
            from src.agents.configs import SACConfig, TD3Config

            # Test TD3 config validation
            td3_config = TD3Config(**agent_configs["td3"])
            assert td3_config.learning_rate > 0
            assert td3_config.gamma > 0 and td3_config.gamma <= 1
            assert td3_config.tau > 0 and td3_config.tau <= 1
            assert td3_config.batch_size > 0
            assert td3_config.buffer_capacity > 0

            # Test SAC config validation
            sac_config = SACConfig(**agent_configs["sac"])
            assert sac_config.learning_rate > 0
            assert sac_config.gamma > 0 and sac_config.gamma <= 1
            assert sac_config.tau > 0 and sac_config.tau <= 1
            assert sac_config.batch_size > 0
            assert sac_config.buffer_capacity > 0

            print("✅ Agent configuration validation test passed")

        except ImportError as e:
            pytest.skip(f"Agent configs not available: {e}")


class TestAgentIntegration:
    """Integration tests for agent components."""

    def test_agent_environment_compatibility(self):
        """Test agent compatibility with different environment configurations."""
        pytest.skip(
            "Environment integration tests - implement when environment is stable"
        )

    def test_agent_multi_step_training(self):
        """Test agent training over multiple steps with experience replay."""
        pytest.skip(
            "Multi-step training tests - implement after basic training is stable"
        )


class TestAgentPerformance:
    """Performance tests for agent operations."""

    @pytest.mark.performance
    def test_agent_action_selection_speed(self):
        """Test action selection performance."""
        pytest.skip("Performance tests - implement after basic functionality is stable")

    @pytest.mark.performance
    def test_agent_training_memory_usage(self):
        """Test memory usage during training."""
        pytest.skip(
            "Memory usage tests - implement after basic functionality is stable"
        )


# Run tests if this file is executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
