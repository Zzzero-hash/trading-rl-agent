"""
Integration tests for TD3 Agent with Trading Environment.
Tests the complete pipeline from config to training.
"""

import pytest
import numpy as np
import torch

pytestmark = pytest.mark.integration
from src.agents.td3_agent import TD3Agent
from src.agents.configs import TD3Config
from src.envs.trading_env import TradingEnv


@pytest.fixture
def td3_config():
    """Create TD3 config optimized for integration testing."""
    from src.agents.configs import TD3Config
    return TD3Config(
        learning_rate=1e-3,
        gamma=0.99,
        tau=0.01,  # Faster updates for testing
        batch_size=16,  # Smaller batch for faster testing
        buffer_capacity=1000,
        hidden_dims=[32, 32],  # Smaller networks for speed
        policy_delay=2,
        target_noise=0.1,
        noise_clip=0.3,
        exploration_noise=0.1
    )


class TestTD3Integration:
    """Integration tests for TD3 agent with trading environment."""
    
    @pytest.fixture
    def trading_env(self, sample_csv_path):
        """Create a trading environment for testing."""
        env_cfg = {
            "dataset_paths": [sample_csv_path],
            "window_size": 10,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
            "include_features": False,
            "continuous_actions": True  # Ensure continuous action space for TD3
        }
        return TradingEnv(env_cfg)

    def _flatten_obs(self, obs):
        # Robustly flatten observation for TD3: handle dict, tuple, array
        if isinstance(obs, tuple) and len(obs) > 0:
            obs = obs[0]
        if isinstance(obs, dict):
            obs = obs.get("market_features", obs)
        obs = np.asarray(obs)
        if obs.ndim > 1:
            obs = obs.flatten()
        return obs

    def test_td3_with_trading_env_initialization(self, trading_env, td3_config):
        obs = trading_env.reset()
        obs_flat = self._flatten_obs(obs)
        state_dim = obs_flat.shape[0]
        action_dim = trading_env.action_space.shape[0]
        agent = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        assert agent.state_dim == state_dim
        assert agent.action_dim == action_dim
        dummy_state = torch.randn(1, state_dim)
        dummy_action = torch.randn(1, action_dim)
        with torch.no_grad():
            action_output = agent.actor(dummy_state)
            assert action_output.shape == (1, action_dim)
        with torch.no_grad():
            q1_output = agent.critic_1(dummy_state, dummy_action)
            q2_output = agent.critic_2(dummy_state, dummy_action)
            assert q1_output.shape == (1, 1)
            assert q2_output.shape == (1, 1)

    def test_td3_environment_interaction(self, trading_env, td3_config):
        obs = trading_env.reset()
        obs_flat = self._flatten_obs(obs)
        state_dim = obs_flat.shape[0]
        action_dim = trading_env.action_space.shape[0]
        agent = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        state = obs_flat
        assert len(state) == state_dim
        action = agent.select_action(state, add_noise=False)
        assert len(action) == action_dim
        assert all(-1.0 <= a <= 1.0 for a in action)
        next_obs, reward, done, *_ = trading_env.step(action)
        next_state = self._flatten_obs(next_obs)
        assert len(next_state) == state_dim
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(done, (bool, np.bool_))

    def test_td3_training_episode(self, trading_env, td3_config):
        obs = trading_env.reset()
        obs_flat = self._flatten_obs(obs)
        state_dim = obs_flat.shape[0]
        action_dim = trading_env.action_space.shape[0]
        agent = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        state = obs_flat
        episode_length = 0
        max_steps = 100
        for _ in range(max_steps):
            action = agent.select_action(state, add_noise=False)
            next_obs, reward, done, *_ = trading_env.step(action)
            next_state = self._flatten_obs(next_obs)
            agent.store_experience(state, action, reward, next_state, done)            
            state = next_state
            episode_length += 1
            if done:
                break
        assert episode_length > 0
        # Try a training step
        agent.train()

    def test_td3_config_dataclass_integration(self, trading_env):
        """Test TD3 works with dataclass config in realistic scenario."""
        # Test various config scenarios
        configs_to_test = [
            TD3Config(),  # Default config
            TD3Config(learning_rate=5e-4, batch_size=32),  # Custom params
            TD3Config(hidden_dims=[128, 128], policy_delay=3)  # Different architecture
        ]
        
        # Get actual flattened state dimension like other working tests
        obs = trading_env.reset()
        obs_flat = self._flatten_obs(obs)
        state_dim = obs_flat.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        for config in configs_to_test:
            agent = TD3Agent(
                config=config,
                state_dim=state_dim,
                action_dim=action_dim
            )
              # Verify config was applied correctly
            assert agent.lr == config.learning_rate
            assert agent.batch_size == config.batch_size
            assert agent.hidden_dims == config.hidden_dims
            assert agent.policy_delay == config.policy_delay
            
            # Test basic functionality
            state = trading_env.reset()
            # Use the same flattening logic as other working tests
            state = self._flatten_obs(state)
            action = agent.select_action(state)
            assert len(action) == action_dim

    def test_td3_save_load_with_training_state(self, trading_env, td3_config, tmp_path):
        """Test TD3 agent save/load preserves training state."""
        # Get actual flattened state dimension like other working tests
        obs = trading_env.reset()
        obs_flat = self._flatten_obs(obs)
        state_dim = obs_flat.shape[0]
        action_dim = trading_env.action_space.shape[0]
        
        # Create and train agent
        agent1 = TD3Agent(
            config=td3_config,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # Add some experiences and train
        for _ in range(20):
            state = np.random.randn(state_dim).astype(np.float32)
            action = np.random.uniform(-1, 1, action_dim).astype(np.float32)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim).astype(np.float32)
            done = False
            agent1.store_experience(state, action, reward, next_state, done)
        
        # Train a few steps
        for _ in range(3):
            agent1.train()
        
        original_total_it = agent1.total_it
        
        # Save agent
        save_path = tmp_path / "td3_agent.pth"
        agent1.save(str(save_path))
        
        # Create new agent and load
        agent2 = TD3Agent(
            config=td3_config,            state_dim=state_dim,
            action_dim=action_dim
        )
        agent2.load(str(save_path))
        
        # Verify training state was preserved
        assert agent2.total_it == original_total_it
        
        # Verify agents produce similar outputs
        test_state = np.random.randn(state_dim)
        action1 = agent1.select_action(test_state, add_noise=False)
        action2 = agent2.select_action(test_state, add_noise=False)
        
        # Actions should be very similar (allowing for small numerical differences)
        np.testing.assert_allclose(action1, action2, rtol=1e-5)


if __name__ == "__main__":
    # Quick integration test
    import sys
    sys.path.append('/workspaces/trading-rl-agent')
    
    from src.envs.trading_env import TradingEnv
    from src.agents.configs import TD3Config
    
    print("🧪 Running TD3 Integration Test...")
    
    # Initialize environment and agent
    env_cfg = {
        "dataset_paths": ["data/sample_training_data_simple_20250607_192034.csv"],
        "window_size": 10,
        "initial_balance": 10000,
        "transaction_cost": 0.001,
        "include_features": False
    }
    env = TradingEnv(env_cfg)
    config = TD3Config(batch_size=16, buffer_capacity=100)
    
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # Flatten
    action_dim = 1  # TD3 needs continuous actions, but trading env has discrete
    
    agent = TD3Agent(config=config, state_dim=state_dim, action_dim=action_dim)
    
    print(f"✅ Environment: {state_dim} states, {action_dim} actions")
    print(f"✅ Agent initialized with {sum(p.numel() for p in agent.actor.parameters())} actor parameters")
      # Test interaction (simplified)
    state, info = env.reset()
    if isinstance(state, dict):
        state = list(state.values())[0]  # Get first value if dict
    if hasattr(state, 'shape') and len(state.shape) > 1:
        state = state.flatten()  # Flatten for TD3
        
    action_continuous = agent.select_action(state)
    # Convert continuous action to discrete for environment
    action_discrete = 1 if action_continuous[0] > 0.33 else (2 if action_continuous[0] < -0.33 else 0)
    
    result = env.step(action_discrete)
    next_state, reward, terminated, truncated, info = result
    done = terminated or truncated
    
    print(f"✅ Environment interaction successful")
    print(f"   Continuous Action: {action_continuous}")
    print(f"   Discrete Action: {action_discrete}")
    print(f"   Reward: {reward:.4f}")
    print(f"   Done: {done}")
    
    print("🎉 TD3 Integration Test Passed!")
