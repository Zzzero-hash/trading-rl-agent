"""
Comprehensive tests for all agent training and inference pipelines.
Tests agent initialization, training, inference, save/load, and error handling.
"""

from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.agents.configs import SACConfig, TD3Config
from src.agents.sac_agent import SACAgent
from src.agents.td3_agent import TD3Agent
from src.envs.trading_env import TradingEnv


class TestAgentInitialization:
    """Test agent initialization and configuration."""

    @pytest.mark.parametrize(
        "agent_class,config_class", [(TD3Agent, TD3Config), (SACAgent, SACConfig)]
    )
    def test_agent_initialization_default_config(self, agent_class, config_class):
        """Test agent initialization with default configuration."""
        config = config_class()
        agent = agent_class(config=config, state_dim=10, action_dim=3)

        assert agent.state_dim == 10
        assert agent.action_dim == 3
        assert agent.config == config

        # Check that networks are initialized
        assert hasattr(agent, "actor")
        assert hasattr(agent, "critic_1")
        if hasattr(agent, "critic_2"):
            assert agent.critic_2 is not None

    @pytest.mark.parametrize(
        "agent_class,config_class", [(TD3Agent, TD3Config), (SACAgent, SACConfig)]
    )
    def test_agent_initialization_custom_config(self, agent_class, config_class):
        """Test agent initialization with custom configuration."""
        config = config_class(
            learning_rate=5e-4, batch_size=64, hidden_dims=[128, 128], gamma=0.95
        )
        agent = agent_class(config=config, state_dim=20, action_dim=5)

        assert agent.lr == 5e-4
        assert agent.batch_size == 64
        assert agent.gamma == 0.95
        assert agent.hidden_dims == [128, 128]

    def test_agent_initialization_invalid_dimensions(self):
        """Test agent initialization with invalid dimensions."""
        config = TD3Config()

        # Invalid state dimension
        with pytest.raises((ValueError, TypeError)):
            TD3Agent(config=config, state_dim=0, action_dim=3)

        # Invalid action dimension
        with pytest.raises((ValueError, TypeError)):
            TD3Agent(config=config, state_dim=10, action_dim=0)

        # Negative dimensions
        with pytest.raises((ValueError, TypeError)):
            TD3Agent(config=config, state_dim=-5, action_dim=3)


class TestAgentNetworkArchitectures:
    """Test agent network architectures and forward passes."""

    def test_td3_network_forward_pass(self, td3_agent):
        """Test TD3 network forward passes."""
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        # Test actor network
        state = torch.randn(1, state_dim)
        action = td3_agent.actor(state)
        assert action.shape == (1, action_dim)
        assert torch.all(action >= -1) and torch.all(action <= 1)

        # Test critic networks
        action_input = torch.randn(1, action_dim)
        q1 = td3_agent.critic_1(state, action_input)
        q2 = td3_agent.critic_2(state, action_input)
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)

    def test_sac_network_forward_pass(self, sac_agent):
        """Test SAC network forward passes."""
        state_dim = sac_agent.state_dim
        action_dim = sac_agent.action_dim

        # Test actor network
        state = torch.randn(1, state_dim)
        action, log_prob = sac_agent.actor(state)
        assert action.shape == (1, action_dim)
        assert log_prob.shape == (1, 1)

        # Test critic networks
        action_input = torch.randn(1, action_dim)
        q1 = sac_agent.critic_1(state, action_input)
        q2 = sac_agent.critic_2(state, action_input)
        assert q1.shape == (1, 1)
        assert q2.shape == (1, 1)

    @pytest.mark.parametrize("batch_size", [1, 16, 32, 64])
    def test_agent_batch_processing(self, td3_agent, batch_size):
        """Test agent processing with different batch sizes."""
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        # Test batch processing
        states = torch.randn(batch_size, state_dim)
        actions = td3_agent.actor(states)
        assert actions.shape == (batch_size, action_dim)

        # Test critic batch processing
        action_inputs = torch.randn(batch_size, action_dim)
        q1_values = td3_agent.critic_1(states, action_inputs)
        q2_values = td3_agent.critic_2(states, action_inputs)
        assert q1_values.shape == (batch_size, 1)
        assert q2_values.shape == (batch_size, 1)


class TestAgentActionSelection:
    """Test agent action selection mechanisms."""

    def test_td3_deterministic_action_selection(self, td3_agent):
        """Test TD3 deterministic action selection."""
        state = np.random.randn(td3_agent.state_dim).astype(np.float32)

        # Multiple calls should return same action
        action1 = td3_agent.select_action(state, add_noise=False)
        action2 = td3_agent.select_action(state, add_noise=False)

        np.testing.assert_array_almost_equal(action1, action2, decimal=6)
        assert len(action1) == td3_agent.action_dim
        assert np.all(action1 >= -1) and np.all(action1 <= 1)

    def test_td3_stochastic_action_selection(self, td3_agent):
        """Test TD3 stochastic action selection."""
        state = np.random.randn(td3_agent.state_dim).astype(np.float32)

        # With noise, actions should be different
        actions = []
        for _ in range(10):
            action = td3_agent.select_action(state, add_noise=True)
            actions.append(action)

        # At least some actions should be different
        unique_actions = len({tuple(a) for a in actions})
        assert unique_actions > 1, "Actions should vary with noise"

    def test_sac_stochastic_action_selection(self, sac_agent):
        """Test SAC stochastic action selection."""
        state = np.random.randn(sac_agent.state_dim).astype(np.float32)

        # SAC should produce different actions each time
        actions = []
        for _ in range(10):
            action = sac_agent.select_action(state)
            actions.append(action)

        # Most actions should be different due to stochasticity
        unique_actions = len({tuple(a) for a in actions})
        assert unique_actions > 3, "SAC actions should be highly variable"

    def test_action_bounds_enforcement(self, td3_agent):
        """Test that actions are properly bounded."""
        # Test with extreme states
        extreme_state = np.full(td3_agent.state_dim, 1000.0, dtype=np.float32)
        action = td3_agent.select_action(extreme_state)

        assert np.all(action >= -1) and np.all(action <= 1), "Actions should be bounded"

        extreme_state = np.full(td3_agent.state_dim, -1000.0, dtype=np.float32)
        action = td3_agent.select_action(extreme_state)

        assert np.all(action >= -1) and np.all(action <= 1), "Actions should be bounded"


class TestAgentTraining:
    """Test agent training mechanisms."""

    def test_experience_storage(self, td3_agent):
        """Test experience storage in replay buffer."""
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        # Store experiences
        for i in range(50):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = i % 10 == 0  # Occasional episode endings

            td3_agent.store_experience(state, action, reward, next_state, done)

        assert len(td3_agent.replay_buffer) == 50

    def test_training_with_sufficient_data(self, td3_agent):
        """Test training when buffer has sufficient data."""
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        # Fill buffer with enough data
        for _ in range(td3_agent.batch_size * 2):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = False

            td3_agent.store_experience(state, action, reward, next_state, done)

        # Training should work
        metrics = td3_agent.train()
        assert isinstance(metrics, dict)
        assert "critic_1_loss" in metrics
        assert "critic_2_loss" in metrics

        # Metrics should be reasonable
        assert isinstance(metrics["critic_1_loss"], (int, float))
        assert isinstance(metrics["critic_2_loss"], (int, float))

    def test_training_with_insufficient_data(self, td3_agent):
        """Test training when buffer has insufficient data."""
        # Don't add enough data
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        for _ in range(td3_agent.batch_size // 2):  # Not enough
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = False

            td3_agent.store_experience(state, action, reward, next_state, done)

        # Training should handle insufficient data gracefully
        metrics = td3_agent.train()
        # Should return None or empty dict when not enough data
        assert metrics is None or metrics == {}

    def test_training_convergence(self, td3_agent):
        """Test that training loss decreases over time."""
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        # Fill buffer
        for _ in range(1000):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = np.random.random() < 0.1

            td3_agent.store_experience(state, action, reward, next_state, done)

        # Train for multiple steps and track loss
        losses = []
        for _ in range(20):
            metrics = td3_agent.train()
            if metrics:
                losses.append(metrics.get("critic_1_loss", float("inf")))

        # Loss should generally decrease (allowing some variance)
        if len(losses) > 10:
            early_avg = np.mean(losses[:5])
            late_avg = np.mean(losses[-5:])
            assert late_avg <= early_avg * 2, "Loss should not increase significantly"


class TestAgentSaveLoad:
    """Test agent save and load functionality."""

    def test_td3_save_load_basic(self, td3_agent, tmp_path):
        """Test basic TD3 save/load functionality."""
        save_path = tmp_path / "td3_test.pth"

        # Save agent
        td3_agent.save(str(save_path))
        assert save_path.exists()

        # Create new agent and load
        new_agent = TD3Agent(
            config=td3_agent.config,
            state_dim=td3_agent.state_dim,
            action_dim=td3_agent.action_dim,
        )
        new_agent.load(str(save_path))

        # Test that loaded agent works
        state = np.random.randn(td3_agent.state_dim).astype(np.float32)
        action1 = td3_agent.select_action(state, add_noise=False)
        action2 = new_agent.select_action(state, add_noise=False)

        np.testing.assert_array_almost_equal(action1, action2, decimal=5)

    def test_sac_save_load_basic(self, sac_agent, tmp_path):
        """Test basic SAC save/load functionality."""
        save_path = tmp_path / "sac_test.pth"

        # Save agent
        sac_agent.save(str(save_path))
        assert save_path.exists()

        # Create new agent and load
        new_agent = SACAgent(
            config=sac_agent.config,
            state_dim=sac_agent.state_dim,
            action_dim=sac_agent.action_dim,
        )
        new_agent.load(str(save_path))

        # Test that networks have similar weights
        state = torch.randn(1, sac_agent.state_dim)
        with torch.no_grad():
            output1 = sac_agent.actor.mu_net(state)
            output2 = new_agent.actor.mu_net(state)

        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-5)

    def test_save_load_with_training_state(self, td3_agent, tmp_path):
        """Test save/load preserves training state."""
        # Train agent a bit
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        for _ in range(100):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.randn(action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = False

            td3_agent.store_experience(state, action, reward, next_state, done)

        # Train for a few steps
        for _ in range(5):
            td3_agent.train()

        original_total_it = td3_agent.total_it

        # Save and reload
        save_path = tmp_path / "trained_agent.pth"
        td3_agent.save(str(save_path))

        new_agent = TD3Agent(
            config=td3_agent.config,
            state_dim=td3_agent.state_dim,
            action_dim=td3_agent.action_dim,
        )
        new_agent.load(str(save_path))

        # Training state should be preserved
        assert new_agent.total_it == original_total_it

    def test_load_nonexistent_file(self, td3_agent):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            td3_agent.load("nonexistent_file.pth")

    def test_load_corrupted_file(self, td3_agent, tmp_path):
        """Test loading from corrupted file."""
        corrupted_path = tmp_path / "corrupted.pth"
        corrupted_path.write_text("this is not a valid pytorch file")

        with pytest.raises((RuntimeError, ValueError, TypeError)):
            td3_agent.load(str(corrupted_path))


class TestAgentErrorHandling:
    """Test agent error handling and edge cases."""

    def test_invalid_state_dimensions(self, td3_agent):
        """Test agent behavior with invalid state dimensions."""
        # Wrong state dimension
        wrong_state = np.random.randn(td3_agent.state_dim + 5).astype(np.float32)

        with pytest.raises((ValueError, RuntimeError)):
            td3_agent.select_action(wrong_state)

    def test_invalid_action_storage(self, td3_agent):
        """Test error handling for invalid experience storage."""
        state_dim = td3_agent.state_dim
        action_dim = td3_agent.action_dim

        # Wrong dimensions
        with pytest.raises((ValueError, TypeError)):
            td3_agent.store_experience(
                state=np.random.randn(state_dim + 1),  # Wrong size
                action=np.random.randn(action_dim),
                reward=0.0,
                next_state=np.random.randn(state_dim),
                done=False,
            )

    def test_extreme_values_handling(self, td3_agent):
        """Test agent behavior with extreme values."""
        state_dim = td3_agent.state_dim

        # Test with extreme state values
        extreme_states = [
            np.full(state_dim, 1e10, dtype=np.float32),
            np.full(state_dim, -1e10, dtype=np.float32),
            np.full(state_dim, np.inf, dtype=np.float32),
            np.full(state_dim, -np.inf, dtype=np.float32),
        ]

        for state in extreme_states:
            if np.isfinite(state).all():
                action = td3_agent.select_action(state)
                assert np.isfinite(action).all(), "Action should be finite"
                assert np.all(action >= -1) and np.all(
                    action <= 1
                ), "Action should be bounded"
            else:
                # Infinite values should be handled gracefully
                with pytest.raises((ValueError, RuntimeError)):
                    td3_agent.select_action(state)


class TestAgentIntegrationWithEnvironment:
    """Test agent integration with trading environments."""

    def test_agent_environment_compatibility(self, integration_env_agent_pair):
        """Test that agent and environment are compatible."""
        env, agent = integration_env_agent_pair

        obs, _ = env.reset()

        # Handle different observation formats
        if isinstance(obs, dict):
            obs = obs.get("market_features", list(obs.values())[0])
        obs = np.asarray(obs)
        if obs.ndim > 1:
            obs = obs.flatten()

        # Agent should accept environment observations
        action = agent.select_action(obs)

        # Environment should accept agent actions
        next_obs, reward, done, truncated, info = env.step(action)

        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)

    def test_full_episode_run(self, integration_env_agent_pair):
        """Test running a full episode with agent and environment."""
        env, agent = integration_env_agent_pair

        obs, _ = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            # Handle observation format
            if isinstance(obs, dict):
                obs = obs.get("market_features", list(obs.values())[0])
            obs = np.asarray(obs)
            if obs.ndim > 1:
                obs = obs.flatten()

            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)

            total_reward += reward
            obs = next_obs
            steps += 1

            if done or truncated:
                break

        assert steps > 0, "Episode should run for at least one step"
        assert isinstance(total_reward, (int, float)), "Total reward should be numeric"

    def test_training_during_episode(self, integration_env_agent_pair):
        """Test training agent during episode execution."""
        env, agent = integration_env_agent_pair

        # Pre-fill buffer with some data
        obs, _ = env.reset()
        for _ in range(agent.batch_size):
            if isinstance(obs, dict):
                obs = obs.get("market_features", list(obs.values())[0])
            obs = np.asarray(obs)
            if obs.ndim > 1:
                obs = obs.flatten()

            action = agent.select_action(obs)
            next_obs, reward, done, truncated, info = env.step(action)

            agent.store_experience(obs, action, reward, next_obs, done)
            obs = next_obs

            if done or truncated:
                obs, _ = env.reset()

        # Now try training
        metrics = agent.train()
        assert metrics is not None, "Training should succeed with sufficient data"
        assert isinstance(metrics, dict), "Training should return metrics dictionary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
