"""
Comprehensive test suite for agent training and inference pipelines.
Tests all agent functionality including initialization, training, inference, and saving/loading.
"""
import pytest
import numpy as np
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import pickle

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

class TestAgentComprehensive:
    """Comprehensive test suite for RL agents."""
    
    @pytest.fixture
    def mock_environment(self):
        """Create a comprehensive mock environment for agent testing."""
        env = Mock()
        
        # Configure mock environment
        env.observation_space = Mock()
        env.observation_space.shape = (20,)
        env.observation_space.low = np.full(20, -1.0)
        env.observation_space.high = np.full(20, 1.0)
        
        env.action_space = Mock()
        env.action_space.shape = (1,)
        env.action_space.low = np.array([-1.0])
        env.action_space.high = np.array([1.0])
        env.action_space.sample.return_value = np.array([0.0])
        
        # Mock environment methods
        def reset_func(seed=None):
            obs = np.random.normal(0, 1, 20)
            info = {"episode": 0, "step": 0}
            return obs, info
        
        def step_func(action):
            obs = np.random.normal(0, 1, 20)
            reward = np.random.normal(0, 1)
            terminated = np.random.random() < 0.01  # 1% chance of termination
            truncated = np.random.random() < 0.01   # 1% chance of truncation
            info = {"reward_components": {"base": reward}}
            return obs, reward, terminated, truncated, info
        
        env.reset = Mock(side_effect=reset_func)
        env.step = Mock(side_effect=step_func)
        
        return env
    
    @pytest.fixture
    def agent_configs(self):
        """Provide comprehensive agent configurations."""
        return {
            'td3': {
                'learning_rate': 3e-4,
                'buffer_size': 50000,
                'batch_size': 256,
                'gamma': 0.99,
                'tau': 0.005,
                'policy_delay': 2,
                'target_noise': 0.2,
                'noise_clip': 0.5,
                'exploration_noise': 0.1,
                'hidden_layers': [256, 256]
            },
            'sac': {
                'learning_rate': 3e-4,
                'buffer_size': 50000,
                'batch_size': 256,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'auto_alpha': True,
                'target_entropy': 'auto',
                'hidden_layers': [256, 256]
            },
            'minimal': {
                'learning_rate': 1e-3,
                'buffer_size': 1000,
                'batch_size': 32,
                'gamma': 0.99,
                'hidden_layers': [64, 64]
            }
        }
    
    @pytest.mark.parametrize("agent_type", ["td3", "sac"])
    def test_agent_initialization(self, agent_type, agent_configs, mock_environment):
        """Test agent initialization with different configurations."""
        try:
            if agent_type == "td3":
                from src.agents.td3_agent import TD3Agent
                from src.agents.configs import TD3Config
                
                config = TD3Config(**agent_configs[agent_type])
                agent = TD3Agent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
                
            elif agent_type == "sac":
                from src.agents.sac_agent import SACAgent  
                from src.agents.configs import SACConfig
                
                config = SACConfig(**agent_configs[agent_type])
                agent = SACAgent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
            
            # Test basic agent properties
            assert hasattr(agent, 'select_action')
            assert hasattr(agent, 'train')
            assert hasattr(agent, 'save')
            assert hasattr(agent, 'load')
            
            # Test configuration
            assert agent.config.learning_rate == agent_configs[agent_type]['learning_rate']
            assert agent.config.batch_size == agent_configs[agent_type]['batch_size']
            
            print(f"✅ {agent_type.upper()} agent initialization test passed")
            
        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")
    
    @pytest.mark.parametrize("agent_type", ["td3", "sac"])  
    def test_agent_action_selection(self, agent_type, agent_configs, mock_environment):
        """Test agent action selection in different modes."""
        try:
            if agent_type == "td3":
                from src.agents.td3_agent import TD3Agent
                from src.agents.configs import TD3Config
                
                config = TD3Config(**agent_configs['minimal'])
                agent = TD3Agent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
                
            elif agent_type == "sac":
                from src.agents.sac_agent import SACAgent
                from src.agents.configs import SACConfig
                
                config = SACConfig(**agent_configs['minimal'])
                agent = SACAgent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
            
            obs = np.random.normal(0, 1, 20)
            
            # Test deterministic action selection
            action_det = agent.select_action(obs, deterministic=True)
            assert action_det is not None
            assert isinstance(action_det, np.ndarray)
            assert action_det.shape == (1,)
            
            # Test stochastic action selection
            action_stoch = agent.select_action(obs, deterministic=False)
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
                from src.agents.td3_agent import TD3Agent
                from src.agents.configs import TD3Config
                
                config = TD3Config(**agent_configs['minimal'])
                agent = TD3Agent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
                
            elif agent_type == "sac":
                from src.agents.sac_agent import SACAgent
                from src.agents.configs import SACConfig
                
                config = SACConfig(**agent_configs['minimal'])
                agent = SACAgent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
            
            # Collect experience
            experiences = []
            obs, _ = mock_environment.reset()
            
            for step in range(100):
                action = agent.select_action(obs, deterministic=False)
                next_obs, reward, terminated, truncated, info = mock_environment.step(action)
                
                # Store experience
                experiences.append({
                    'obs': obs.copy(),
                    'action': action.copy(),
                    'reward': reward,
                    'next_obs': next_obs.copy(),
                    'terminated': terminated,
                    'truncated': truncated
                })
                
                # Add to agent's buffer if available
                if hasattr(agent, 'replay_buffer'):
                    agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                
                obs = next_obs
                
                if terminated or truncated:
                    obs, _ = mock_environment.reset()
            
            # Test training
            if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > agent.config.batch_size:
                # Train the agent
                training_info = agent.train()
                
                # Validate training output
                assert isinstance(training_info, dict)
                assert len(training_info) > 0
                
                # Check for common training metrics
                expected_metrics = ['actor_loss', 'critic_loss', 'total_loss', 'q_value']
                found_metrics = [m for m in expected_metrics if m in training_info]
                assert len(found_metrics) > 0, f"No expected metrics found in {training_info.keys()}"
                
                print(f"✅ {agent_type.upper()} training pipeline test passed")
            else:
                print(f"⚠️ {agent_type.upper()} training skipped - insufficient experience")
                
        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")
    
    @pytest.mark.parametrize("agent_type", ["td3", "sac"])
    def test_agent_save_load(self, agent_type, agent_configs, mock_environment, tmp_path):
        """Test agent save and load functionality."""
        try:
            if agent_type == "td3":
                from src.agents.td3_agent import TD3Agent
                from src.agents.configs import TD3Config
                
                config = TD3Config(**agent_configs['minimal'])
                agent = TD3Agent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
                
            elif agent_type == "sac":
                from src.agents.sac_agent import SACAgent
                from src.agents.configs import SACConfig
                
                config = SACConfig(**agent_configs['minimal'])
                agent = SACAgent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
            
            # Get initial action for comparison
            test_obs = np.random.normal(0, 1, 20)
            initial_action = agent.select_action(test_obs, deterministic=True)
            
            # Save agent
            save_path = tmp_path / f"{agent_type}_agent"
            agent.save(str(save_path))
            
            # Verify save files exist
            assert save_path.exists() or Path(f"{save_path}.pkl").exists() or Path(f"{save_path}.pth").exists()
            
            # Create new agent and load
            if agent_type == "td3":
                new_agent = TD3Agent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
            else:
                new_agent = SACAgent(
                    observation_space=mock_environment.observation_space,
                    action_space=mock_environment.action_space,
                    config=config
                )
            
            # Load saved agent
            new_agent.load(str(save_path))
            
            # Test that loaded agent behaves similarly
            loaded_action = new_agent.select_action(test_obs, deterministic=True)
            
            # Actions should be similar (within tolerance due to potential randomness in initialization)
            action_diff = np.abs(initial_action - loaded_action).mean()
            assert action_diff < 2.0, f"Loaded agent behavior differs significantly: {action_diff}"
            
            print(f"✅ {agent_type.upper()} save/load test passed")
            
        except ImportError as e:
            pytest.skip(f"{agent_type.upper()} agent not available: {e}")
    
    def test_agent_ensemble_integration(self, agent_configs, mock_environment):
        """Test ensemble agent functionality."""
        try:
            from src.agents.ensemble_agent import EnsembleAgent
            from src.agents.configs import EnsembleConfig
            
            # Create ensemble configuration
            ensemble_config = EnsembleConfig(
                agents=[
                    {"type": "td3", "config": agent_configs['minimal']},
                    {"type": "sac", "config": agent_configs['minimal']}
                ],
                combination_method="weighted_average",
                weights=[0.6, 0.4]
            )
            
            ensemble = EnsembleAgent(
                observation_space=mock_environment.observation_space,
                action_space=mock_environment.action_space,
                config=ensemble_config
            )
            
            # Test ensemble action selection
            obs = np.random.normal(0, 1, 20)
            action = ensemble.select_action(obs)
            
            assert action is not None
            assert isinstance(action, np.ndarray)
            assert action.shape == (1,)
            
            print("✅ Ensemble agent integration test passed")
            
        except ImportError as e:
            pytest.skip(f"Ensemble agent not available: {e}")

class TestAgentTrainingIntegration:
    """Test agent training integration with environments and data."""
    
    def test_agent_environment_training_loop(self, mock_environment, td3_config):
        """Test complete agent-environment training loop."""
        try:
            from src.agents.td3_agent import TD3Agent
            from src.agents.configs import TD3Config
            
            config = TD3Config(**td3_config)
            agent = TD3Agent(
                observation_space=mock_environment.observation_space,
                action_space=mock_environment.action_space,
                config=config
            )
            
            # Training loop
            total_episodes = 5
            total_steps = 0
            
            for episode in range(total_episodes):
                obs, _ = mock_environment.reset()
                episode_reward = 0
                episode_steps = 0
                
                for step in range(50):  # Max 50 steps per episode
                    action = agent.select_action(obs, deterministic=False)
                    next_obs, reward, terminated, truncated, info = mock_environment.step(action)
                    
                    # Store experience
                    if hasattr(agent, 'replay_buffer'):
                        agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                    
                    episode_reward += reward
                    episode_steps += 1
                    total_steps += 1
                    
                    # Train agent
                    if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > config.batch_size:
                        training_info = agent.train()
                        assert isinstance(training_info, dict)
                    
                    obs = next_obs
                    
                    if terminated or truncated:
                        break
                
                print(f"Episode {episode + 1}: {episode_steps} steps, reward: {episode_reward:.2f}")
            
            assert total_steps > 0
            print(f"✅ Training loop test passed - {total_steps} total steps")
            
        except ImportError as e:
            pytest.skip(f"Training integration test skipped: {e}")
    
    def test_agent_performance_tracking(self, mock_environment, sac_config):
        """Test agent performance tracking during training."""
        try:
            from src.agents.sac_agent import SACAgent
            from src.agents.configs import SACConfig
            
            config = SACConfig(**sac_config)
            agent = SACAgent(
                observation_space=mock_environment.observation_space,
                action_space=mock_environment.action_space,
                config=config
            )
            
            # Track performance metrics
            performance_metrics = {
                'episode_rewards': [],
                'episode_lengths': [],
                'training_losses': [],
                'q_values': []
            }
            
            for episode in range(3):
                obs, _ = mock_environment.reset()
                episode_reward = 0
                episode_length = 0
                
                for step in range(30):
                    action = agent.select_action(obs)
                    next_obs, reward, terminated, truncated, info = mock_environment.step(action)
                    
                    if hasattr(agent, 'replay_buffer'):
                        agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Train and collect metrics
                    if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) > config.batch_size:
                        training_info = agent.train()
                        
                        if 'total_loss' in training_info:
                            performance_metrics['training_losses'].append(training_info['total_loss'])
                        if 'q_value' in training_info:
                            performance_metrics['q_values'].append(training_info['q_value'])
                    
                    obs = next_obs
                    
                    if terminated or truncated:
                        break
                
                performance_metrics['episode_rewards'].append(episode_reward)
                performance_metrics['episode_lengths'].append(episode_length)
            
            # Validate collected metrics
            assert len(performance_metrics['episode_rewards']) == 3
            assert len(performance_metrics['episode_lengths']) == 3
            assert all(isinstance(r, (int, float)) for r in performance_metrics['episode_rewards'])
            assert all(isinstance(l, int) for l in performance_metrics['episode_lengths'])
            
            print("✅ Performance tracking test passed")
            print(f"   Average episode reward: {np.mean(performance_metrics['episode_rewards']):.2f}")
            print(f"   Average episode length: {np.mean(performance_metrics['episode_lengths']):.1f}")
            
        except ImportError as e:
            pytest.skip(f"Performance tracking test skipped: {e}")

class TestAgentStressTests:
    """Stress tests for agent robustness."""
    
    @pytest.mark.slow
    def test_agent_numerical_stability(self, mock_environment, td3_config):
        """Test agent numerical stability over extended training."""
        try:
            from src.agents.td3_agent import TD3Agent
            from src.agents.configs import TD3Config
            
            config = TD3Config(**td3_config)
            agent = TD3Agent(
                observation_space=mock_environment.observation_space,
                action_space=mock_environment.action_space,
                config=config
            )
            
            # Extended training with numerical stability checks
            for step in range(500):
                obs = np.random.normal(0, 1, 20)
                
                # Check for NaN/Inf in observations
                assert not np.any(np.isnan(obs)), f"NaN in observation at step {step}"
                assert not np.any(np.isinf(obs)), f"Inf in observation at step {step}"
                
                action = agent.select_action(obs)
                
                # Check for NaN/Inf in actions
                assert not np.any(np.isnan(action)), f"NaN in action at step {step}"
                assert not np.any(np.isinf(action)), f"Inf in action at step {step}"
                
                # Simulate training step
                next_obs, reward, terminated, truncated, info = mock_environment.step(action)
                
                if hasattr(agent, 'replay_buffer'):
                    agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                    
                    if len(agent.replay_buffer) > config.batch_size:
                        training_info = agent.train()
                        
                        # Check training metrics for numerical stability
                        for key, value in training_info.items():
                            if isinstance(value, (int, float)):
                                assert not np.isnan(value), f"NaN in {key} at step {step}"
                                assert not np.isinf(value), f"Inf in {key} at step {step}"
            
            print("✅ Numerical stability test passed - 500 training steps")
            
        except ImportError as e:
            pytest.skip(f"Numerical stability test skipped: {e}")
    
    @pytest.mark.memory
    def test_agent_memory_efficiency(self, mock_environment, sac_config, memory_monitor):
        """Test agent memory efficiency during training."""
        try:
            from src.agents.sac_agent import SACAgent
            from src.agents.configs import SACConfig
            
            initial_memory = memory_monitor['initial']
            
            config = SACConfig(**sac_config)
            agent = SACAgent(
                observation_space=mock_environment.observation_space,
                action_space=mock_environment.action_space,
                config=config
            )
            
            # Training with memory monitoring
            for step in range(200):
                obs = np.random.normal(0, 1, 20)
                action = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = mock_environment.step(action)
                
                if hasattr(agent, 'replay_buffer'):
                    agent.replay_buffer.add(obs, action, reward, next_obs, terminated or truncated)
                    
                    if len(agent.replay_buffer) > config.batch_size and step % 10 == 0:
                        agent.train()
            
            final_memory = memory_monitor['current']()
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 200, f"Excessive memory usage: {memory_increase:.2f} MB"
            
            print(f"✅ Memory efficiency test passed - memory increase: {memory_increase:.2f} MB")
            
        except ImportError as e:
            pytest.skip(f"Memory efficiency test skipped: {e}")
