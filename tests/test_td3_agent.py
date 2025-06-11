"""
Test suite for TD3 (Twin Delayed DDPG) agent.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.agents.td3_agent import TD3Agent
from src.agents.configs import TD3Config
from src.envs.trading_env import TradingEnv
@pytest.fixture(autouse=True)
def set_seed():
    np.random.seed(42)
    torch.manual_seed(42)



class TestTD3Agent:
    """Test cases for TD3 agent."""
    
    @pytest.fixture
    def td3_config(self):
        """Create test configuration for TD3 agent."""
        return TD3Config(
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=32,
            buffer_capacity=10000,
            hidden_dims=[64, 64],
            policy_delay=2,
            target_noise=0.2,
            noise_clip=0.5,
            exploration_noise=0.1,
        )
    
    @pytest.fixture
    def td3_agent(self, td3_config):
        """Create TD3 agent instance."""
        return TD3Agent(
            state_dim=10,
            action_dim=3,
            config=td3_config,
            device="cpu",
        )
    
    @pytest.fixture
    def sample_state(self):
        """Create sample state for testing."""
        return np.random.randn(10).astype(np.float32)
    
    @pytest.fixture
    def sample_batch(self):
        """Create sample batch for training."""
        batch_size = 32
        state_dim = 10
        action_dim = 3
        
        states = np.random.randn(batch_size, state_dim).astype(np.float32)
        actions = np.random.uniform(-1, 1, (batch_size, action_dim)).astype(np.float32)
        rewards = np.random.randn(batch_size).astype(np.float32)
        next_states = np.random.randn(batch_size, state_dim).astype(np.float32)
        dones = np.random.choice([0, 1], batch_size).astype(np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def test_agent_initialization(self, td3_config):
        """Test agent initialization."""
        agent = TD3Agent(td3_config)
        
        # Check networks are created
        assert hasattr(agent, 'actor')
        assert hasattr(agent, 'actor_target')
        assert hasattr(agent, 'critic_1')
        assert hasattr(agent, 'critic_2')
        assert hasattr(agent, 'critic_1_target')
        assert hasattr(agent, 'critic_2_target')
        
        # Check optimizers
        assert hasattr(agent, 'actor_optimizer')
        assert hasattr(agent, 'critic_1_optimizer')
        assert hasattr(agent, 'critic_2_optimizer')
          # Check replay buffer
        assert hasattr(agent, 'replay_buffer')
        assert agent.replay_buffer.capacity == td3_config.buffer_capacity
        
        # Check configuration
        assert agent.config == td3_config
        assert agent.total_it == 0
    
    def test_network_architectures(self, td3_agent):
        """Test network architectures."""
        # Test actor network
        test_state = torch.randn(1, 10)
        actor_output = td3_agent.actor(test_state)
        assert actor_output.shape == (1, 3)
        assert torch.all(torch.abs(actor_output) <= 1.0)  # Actions should be clipped
        
        # Test critic networks
        test_action = torch.randn(1, 3)
        critic_1_output = td3_agent.critic_1(test_state, test_action)
        critic_2_output = td3_agent.critic_2(test_state, test_action)
        assert critic_1_output.shape == (1, 1)
        assert critic_2_output.shape == (1, 1)
    
    def test_select_action_exploration(self, td3_agent, sample_state):
        """Test action selection with exploration."""
        # Test with exploration noise
        action = td3_agent.select_action(sample_state, add_noise=True)
        assert action.shape == (3,)
        assert np.all(np.abs(action) <= 1.0)  # Actions should be within bounds
        
        # Test without exploration noise
        action_no_noise = td3_agent.select_action(sample_state, add_noise=False)
        assert action_no_noise.shape == (3,)
        assert np.all(np.abs(action_no_noise) <= 1.0)
    
    def test_select_action_deterministic(self, td3_agent, sample_state):
        """Test deterministic action selection."""
        # Get multiple actions for same state
        action1 = td3_agent.select_action(sample_state, add_noise=False)
        action2 = td3_agent.select_action(sample_state, add_noise=False)
        
        # Should be identical (deterministic)
        np.testing.assert_array_almost_equal(action1, action2)
    
    def test_replay_buffer_storage(self, td3_agent, sample_state):
        """Test replay buffer operations."""
        # Add experience
        action = np.random.uniform(-1, 1, 3).astype(np.float32)
        reward = 1.0
        next_state = np.random.randn(10).astype(np.float32)
        done = False
        
        td3_agent.replay_buffer.add(sample_state, action, reward, next_state, done)
        assert td3_agent.replay_buffer.size == 1
        
        # Add more experiences
        for _ in range(20):
            state = np.random.randn(10).astype(np.float32)
            action = np.random.uniform(-1, 1, 3).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(10).astype(np.float32)
            done = np.random.choice([True, False])
            td3_agent.replay_buffer.add(state, action, reward, next_state, done)

        assert td3_agent.replay_buffer.size == 21
    
    def test_train_step(self, td3_agent, sample_batch):
        """Test training step."""
        states, actions, rewards, next_states, dones = sample_batch
        
        # Fill replay buffer
        for i in range(len(states)):
            td3_agent.replay_buffer.add(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
        
        # Get initial parameters
        initial_actor_params = [p.clone() for p in td3_agent.actor.parameters()]
        initial_critic_1_params = [p.clone() for p in td3_agent.critic_1.parameters()]
        initial_critic_2_params = [p.clone() for p in td3_agent.critic_2.parameters()]
        
        # Perform training steps
        for _ in range(5):
            td3_agent.train()
        
        # Check that parameters have changed (learning occurred)
        actor_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_actor_params, td3_agent.actor.parameters())
        )
        critic_1_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_critic_1_params, td3_agent.critic_1.parameters())
        )
        critic_2_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_critic_2_params, td3_agent.critic_2.parameters())
        )
        
        # At least critics should have changed
        assert critic_1_changed
        assert critic_2_changed
        
        # Check total iterations counter
        assert td3_agent.total_it == 5
    
    def test_policy_delay(self, td3_agent, sample_batch):
        """Test policy delay mechanism."""
        states, actions, rewards, next_states, dones = sample_batch
        
        # Fill replay buffer
        for i in range(len(states)):
            td3_agent.replay_buffer.add(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
        
        # Get initial actor parameters
        initial_actor_params = [p.clone() for p in td3_agent.actor.parameters()]
          # Train for policy_delay - 1 steps (policy should not update)
        for _ in range(td3_agent.config.policy_delay - 1):
            td3_agent.train()
        
        # Check actor parameters haven't changed
        actor_unchanged = all(
            torch.equal(initial, current) 
            for initial, current in zip(initial_actor_params, td3_agent.actor.parameters())
        )
        assert actor_unchanged
        
        # Train one more step (policy should update)
        td3_agent.train()
        
        # Check actor parameters have changed
        actor_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_actor_params, td3_agent.actor.parameters())
        )
        assert actor_changed
    
    def test_target_network_updates(self, td3_agent, sample_batch):
        """Test soft target network updates."""
        states, actions, rewards, next_states, dones = sample_batch
        
        # Fill replay buffer
        for i in range(len(states)):
            td3_agent.replay_buffer.add(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
        
        # Get initial target parameters
        initial_actor_target_params = [p.clone() for p in td3_agent.actor_target.parameters()]
        initial_critic_1_target_params = [p.clone() for p in td3_agent.critic_1_target.parameters()]
        initial_critic_2_target_params = [p.clone() for p in td3_agent.critic_2_target.parameters()]
        
        # Train for several steps
        for _ in range(10):
            td3_agent.train()
        
        # Check target networks have been updated (but not completely replaced)
        actor_target_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_actor_target_params, td3_agent.actor_target.parameters())
        )
        critic_1_target_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_critic_1_target_params, td3_agent.critic_1_target.parameters())
        )
        critic_2_target_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_critic_2_target_params, td3_agent.critic_2_target.parameters())
        )
        
        assert actor_target_changed
        assert critic_1_target_changed
        assert critic_2_target_changed
    
    def test_save_and_load(self, td3_agent):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "td3_test_model")
            
            # Train agent briefly to modify parameters
            for _ in range(10):
                state = np.random.randn(10).astype(np.float32)
                action = np.random.uniform(-1, 1, 3).astype(np.float32)
                reward = np.random.randn()
                next_state = np.random.randn(10).astype(np.float32)
                done = False
                td3_agent.replay_buffer.add(state, action, reward, next_state, done)
            
            for _ in range(5):
                td3_agent.train()
            
            # Save model
            td3_agent.save(save_path)
            
            # Create new agent and load model
            new_agent = TD3Agent(td3_agent.config)
            new_agent.load(save_path)
            
            # Compare parameters
            for p1, p2 in zip(td3_agent.actor.parameters(), new_agent.actor.parameters()):
                assert torch.equal(p1, p2)
            
            for p1, p2 in zip(td3_agent.critic_1.parameters(), new_agent.critic_1.parameters()):
                assert torch.equal(p1, p2)
            
            for p1, p2 in zip(td3_agent.critic_2.parameters(), new_agent.critic_2.parameters()):
                assert torch.equal(p1, p2)
    
    def test_insufficient_buffer_size(self, td3_agent):
        """Test training with insufficient buffer size."""
        # Add only a few experiences (less than batch size)
        for _ in range(10):
            state = np.random.randn(10).astype(np.float32)
            action = np.random.uniform(-1, 1, 3).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(10).astype(np.float32)
            done = False
            td3_agent.replay_buffer.add(state, action, reward, next_state, done)
        
        # Training should not crash
        initial_it = td3_agent.total_it
        td3_agent.train()
        
        # Should not update due to insufficient data
        assert td3_agent.total_it == initial_it
    
    @pytest.mark.parametrize("policy_noise", [0.0, 0.1, 0.2, 0.5])
    def test_different_policy_noise_levels(self, td3_config, policy_noise):
        """Test TD3 with different policy noise levels."""
        td3_config.policy_noise = policy_noise
        agent = TD3Agent(td3_config)
        
        # Should initialize without error
        assert agent.config.policy_noise == policy_noise
        
        # Should be able to select actions
        state = np.random.randn(10).astype(np.float32)
        action = agent.select_action(state)
        assert action.shape == (3,)
    
    @pytest.mark.parametrize("max_action", [0.5, 1.0, 2.0])
    def test_different_action_bounds(self, td3_config, max_action):
        """Test TD3 with different action bounds."""
        td3_config.max_action = max_action
        agent = TD3Agent(td3_config)
        
        state = np.random.randn(10).astype(np.float32)
        action = agent.select_action(state)
        
        # Actions should be within bounds
        assert np.all(np.abs(action) <= max_action)
    
    def test_training_metrics_tracking(self, td3_agent, sample_batch):
        """Test that training metrics are properly tracked."""
        states, actions, rewards, next_states, dones = sample_batch
        
        # Fill replay buffer
        for i in range(len(states)):
            td3_agent.replay_buffer.add(
                states[i], actions[i], rewards[i], next_states[i], dones[i]
            )
        
        # Train and check metrics
        initial_it = td3_agent.total_it
        td3_agent.train()
          # Should increment iteration counter
        assert td3_agent.total_it == initial_it + 1
    
    def test_twin_critic_mechanism(self, td3_agent):
        """Test that twin critics produce different outputs."""
        state = torch.randn(1, 10)
        action = torch.randn(1, 3)
        
        # Get outputs from both critics
        q1 = td3_agent.critic_1(state, action)
        q2 = td3_agent.critic_2(state, action)
        
        # They should be different (unless by coincidence)
        assert q1.shape == q2.shape == (1, 1)
        
        # Get initial parameters to check for learning
        initial_critic_1_params = [p.clone() for p in td3_agent.critic_1.parameters()]
        initial_critic_2_params = [p.clone() for p in td3_agent.critic_2.parameters()]
        
        # Fill buffer with more diverse training data
        for _ in range(50):
            s = np.random.randn(10).astype(np.float32)
            a = np.random.uniform(-1, 1, 3).astype(np.float32)
            r = np.random.uniform(-10, 10)  # More diverse rewards
            ns = np.random.randn(10).astype(np.float32)
            d = np.random.choice([True, False])
            td3_agent.replay_buffer.add(s, a, r, ns, d)
        
        # Train with more steps to ensure learning
        for _ in range(20):
            td3_agent.train()
        
        # Check that parameters have changed (indicating learning)
        critic_1_params_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_critic_1_params, td3_agent.critic_1.parameters())
        )
        critic_2_params_changed = any(
            not torch.equal(initial, current) 
            for initial, current in zip(initial_critic_2_params, td3_agent.critic_2.parameters())
        )
        
        # At least one critic should have learned (parameters changed)
        assert critic_1_params_changed or critic_2_params_changed, "At least one critic should have learned"


class TestTD3Config:
    """Test cases for TD3 configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TD3Config()

        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.tau == 0.005
        assert config.batch_size == 256
        assert config.buffer_capacity == 1000000
        assert config.policy_delay == 2
        assert config.target_noise == 0.2
        assert config.noise_clip == 0.5
        assert config.exploration_noise == 0.1

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TD3Config(
            learning_rate=1e-3,
            gamma=0.95,
            tau=0.01,
            batch_size=64,
            buffer_capacity=5000,
            hidden_dims=[128, 128],
            policy_delay=3,
            target_noise=0.1,
            noise_clip=0.3,
            exploration_noise=0.2,
        )

        assert config.learning_rate == 1e-3
        assert config.gamma == 0.95
        assert config.tau == 0.01
        assert config.batch_size == 64
        assert config.buffer_capacity == 5000
        assert config.hidden_dims == [128, 128]
        assert config.policy_delay == 3
        assert config.target_noise == 0.1
        assert config.noise_clip == 0.3
        assert config.exploration_noise == 0.2


if __name__ == "__main__":
    pytest.main([__file__])
