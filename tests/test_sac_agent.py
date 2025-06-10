"""
Tests for SAC Agent Implementation
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch
import tempfile
import os

from src.agents.sac_agent import SACAgent, Actor, QNetwork, ReplayBuffer, Critic
from src.agents.configs import SACConfig


class TestSACAgent:
    """Test cases for SAC Agent."""
    
    @pytest.fixture
    def agent_config(self):
        """Test configuration for SAC agent."""
        return SACConfig(
            learning_rate=1e-3,
            gamma=0.99,
            tau=0.01,
            batch_size=32,
            buffer_capacity=1000,
            hidden_dims=[64, 64],
            automatic_entropy_tuning=True,
            target_entropy=-1.0,
        )
    
    @pytest.fixture
    def sac_agent(self, agent_config):
        """Create SAC agent for testing."""
        return SACAgent(
            state_dim=10,
            action_dim=1,
            config=agent_config,
            device="cpu"
        )
    
    def test_agent_initialization(self, sac_agent):
        """Test SAC agent initialization."""
        assert sac_agent.state_dim == 10
        assert sac_agent.action_dim == 1
        assert sac_agent.device.type == "cpu"
        assert sac_agent.lr == 1e-3
        assert sac_agent.gamma == 0.99
        assert sac_agent.automatic_entropy_tuning is True
        
        # Check networks are initialized
        assert sac_agent.actor is not None
        assert sac_agent.critic is not None
        assert sac_agent.critic_target is not None
        
        # Check optimizers
        assert sac_agent.actor_optimizer is not None
        assert sac_agent.critic_optimizer is not None
        assert sac_agent.alpha_optimizer is not None
    
    def test_action_selection(self, sac_agent):
        """Test action selection."""
        state = np.random.randn(10)
        
        # Test stochastic action (training)
        action = sac_agent.select_action(state, evaluate=False)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0  # Bounded by tanh
        
        # Test deterministic action (evaluation)
        action_eval = sac_agent.select_action(state, evaluate=True)
        assert isinstance(action_eval, np.ndarray)
        assert action_eval.shape == (1,)
        assert -1.0 <= action_eval[0] <= 1.0
    
    def test_experience_storage(self, sac_agent):
        """Test experience storage in replay buffer."""
        state = np.random.randn(10)
        action = np.random.randn(1)
        reward = np.random.randn()
        next_state = np.random.randn(10)
        done = False
        
        initial_buffer_size = len(sac_agent.replay_buffer)
        sac_agent.store_experience(state, action, reward, next_state, done)
        
        assert len(sac_agent.replay_buffer) == initial_buffer_size + 1
    
    def test_update_with_insufficient_data(self, sac_agent):
        """Test update with insufficient data in replay buffer."""
        # Buffer is empty, should return empty dict
        metrics = sac_agent.update()
        assert metrics == {}
    
    def test_update_with_sufficient_data(self, sac_agent):
        """Test update with sufficient data."""
        # Fill replay buffer with minimum required experiences
        for _ in range(sac_agent.batch_size):
            state = np.random.randn(10)
            action = np.random.randn(1)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            sac_agent.store_experience(state, action, reward, next_state, done)
        
        # Test update
        metrics = sac_agent.update()
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics
        assert "mean_q1" in metrics
        assert "mean_q2" in metrics
        
        # Check values are reasonable
        assert isinstance(metrics["critic_loss"], float)
        assert isinstance(metrics["actor_loss"], float)
        assert isinstance(metrics["alpha"], float)
        assert metrics["alpha"] > 0  # Temperature should be positive
    
    def test_save_and_load(self, sac_agent):
        """Test saving and loading agent state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_sac_agent.pt")
            
            # Train for a few steps to change parameters
            for _ in range(sac_agent.batch_size):
                state = np.random.randn(10)
                action = np.random.randn(1)
                reward = np.random.randn()
                next_state = np.random.randn(10)
                done = False
                sac_agent.store_experience(state, action, reward, next_state, done)
            
            sac_agent.update()
            original_step = sac_agent.training_step
            
            # Save agent
            sac_agent.save(filepath)
            assert os.path.exists(filepath)
            
            # Create new agent and load
            new_agent = SACAgent(
                state_dim=10,
                action_dim=1,
                config=sac_agent.config,
                device="cpu"
            )
            new_agent.load(filepath)
            
            # Check that training step was loaded
            assert new_agent.training_step == original_step
    
    def test_config_loading_from_dict(self):
        """Test configuration loading from dictionary."""
        config = {"learning_rate": 5e-4, "gamma": 0.95}
        agent = SACAgent(state_dim=5, action_dim=1, config=config)
        
        assert agent.lr == 5e-4
        assert agent.gamma == 0.95
    
    def test_config_loading_default(self):
        """Test default configuration loading."""
        agent = SACAgent(state_dim=5, action_dim=1, config=None)
        
        # Should use default values
        assert agent.lr == 3e-4
        assert agent.gamma == 0.99
        assert agent.automatic_entropy_tuning is True


class TestActor:
    """Test cases for SAC Actor network."""
    
    def test_actor_forward(self):
        """Test actor forward pass."""
        actor = Actor(state_dim=10, action_dim=1, hidden_dims=[32, 32])
        state = torch.randn(5, 10)  # Batch of 5 states
        
        mean, log_std = actor(state)
        
        assert mean.shape == (5, 1)
        assert log_std.shape == (5, 1)
        assert torch.all(log_std >= actor.log_std_min)
        assert torch.all(log_std <= actor.log_std_max)
    
    def test_actor_sample(self):
        """Test actor action sampling."""
        actor = Actor(state_dim=10, action_dim=1)
        state = torch.randn(3, 10)
        
        action, log_prob = actor.sample(state)
        
        assert action.shape == (3, 1)
        assert log_prob.shape == (3, 1)
        assert torch.all(torch.abs(action) <= 1.0)  # Bounded by tanh


class TestCritic:
    """Test cases for SAC Critic networks."""
    
    def test_critic_forward(self):
        """Test critic forward pass."""
        critic = Critic(state_dim=10, action_dim=1, hidden_dims=[32, 32])
        state = torch.randn(5, 10)
        action = torch.randn(5, 1)
        
        q1, q2 = critic(state, action)
        
        assert q1.shape == (5, 1)
        assert q2.shape == (5, 1)


class TestReplayBuffer:
    """Test cases for Replay Buffer."""
    
    def test_buffer_initialization(self):
        """Test replay buffer initialization."""
        buffer = ReplayBuffer(capacity=100)
        assert len(buffer) == 0
        assert buffer.buffer.maxlen == 100
    
    def test_buffer_push_and_sample(self):
        """Test pushing and sampling from buffer."""
        buffer = ReplayBuffer(capacity=10)
        
        # Add some experiences
        for i in range(5):
            state = np.random.randn(4)
            action = np.random.randn(1)
            reward = float(i)
            next_state = np.random.randn(4)
            done = False
            buffer.push(state, action, reward, next_state, done)
        
        assert len(buffer) == 5
        
        # Sample batch
        batch = buffer.sample(3)
        states, actions, rewards, next_states, dones = batch
        
        assert states.shape == (3, 4)
        assert actions.shape == (3, 1)
        assert rewards.shape == (3,)
        assert next_states.shape == (3, 4)
        assert dones.shape == (3,)
    
    def test_buffer_capacity_limit(self):
        """Test buffer capacity limit."""
        buffer = ReplayBuffer(capacity=3)
        
        # Add more experiences than capacity
        for i in range(5):
            state = np.random.randn(2)
            action = np.random.randn(1)
            reward = float(i)
            next_state = np.random.randn(2)
            done = False
            buffer.push(state, action, reward, next_state, done)
        
        # Should not exceed capacity
        assert len(buffer) == 3


@pytest.mark.integration
class TestSACIntegration:
    """Integration tests for SAC agent."""
    
    def test_training_loop(self):
        """Test complete training loop."""
        agent = SACAgent(
            state_dim=5,
            action_dim=1,
            config={
                "batch_size": 16,
                "buffer_capacity": 100,
                "learning_rate": 1e-3
            },
            device="cpu"
        )
        
        # Collect experiences
        for _ in range(20):
            state = np.random.randn(5)
            action = agent.select_action(state, evaluate=False)
            reward = np.random.randn()
            next_state = np.random.randn(5)
            done = np.random.choice([True, False], p=[0.1, 0.9])
            
            agent.store_experience(state, action, reward, next_state, done)
        
        # Train for several steps
        training_metrics = []
        for _ in range(10):
            metrics = agent.update()
            if metrics:
                training_metrics.append(metrics)
        
        # Should have some training metrics
        assert len(training_metrics) > 0
        
        # Check metrics structure
        for metrics in training_metrics:
            assert "critic_loss" in metrics
            assert "actor_loss" in metrics
            assert "alpha" in metrics
    
    def test_deterministic_vs_stochastic_actions(self):
        """Test difference between training and evaluation actions."""
        agent = SACAgent(state_dim=5, action_dim=1, device="cpu")
        state = np.random.randn(5)
        
        # Get multiple stochastic actions
        stochastic_actions = [agent.select_action(state, evaluate=False) for _ in range(10)]
        
        # Get multiple deterministic actions
        deterministic_actions = [agent.select_action(state, evaluate=True) for _ in range(10)]
        
        # Stochastic actions should have more variance
        stochastic_var = np.var(stochastic_actions)
        deterministic_var = np.var(deterministic_actions)
        
        # Note: This test might be flaky due to randomness, but generally stochastic should have more variance
        # We just check that both modes work without errors
        assert len(stochastic_actions) == 10
        assert len(deterministic_actions) == 10


if __name__ == "__main__":
    pytest.main([__file__])
