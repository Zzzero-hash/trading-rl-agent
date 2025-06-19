"""
Integration tests for Model ↔ Environment, Agent ↔ Environment, Model ↔ Agent, and Full Pipeline.
"""

import os
import sys

import numpy as np
import pytest
import torch

from src.agents.configs import TD3Config
from src.agents.td3_agent import TD3Agent
from src.envs.trading_env import TradingEnv

sys.path.insert(0, os.path.dirname(__file__))
from test_setup_utils import setup_test_env, teardown_test_env


@pytest.mark.integration
def test_model_environment_interface():
    """Test model ↔ environment compatibility."""
    env, agent = setup_test_env()
    try:
        state_dim = env.observation_space.shape[0]
        # Robust action_dim logic
        if hasattr(env.action_space, "n"):
            action_dim = env.action_space.n
        elif (
            hasattr(env.action_space, "shape")
            and env.action_space.shape is not None
            and len(env.action_space.shape) > 0
        ):
            action_dim = env.action_space.shape[0]
        else:
            raise ValueError(
                "env.action_space does not have 'n' or a valid 'shape' attribute"
            )
        assert state_dim > 0, f"State dim should be > 0, got {state_dim}"
        assert action_dim > 0, f"Action dim should be > 0, got {action_dim}"
    except Exception as e:
        print(f"Error in model-environment interface: {e}")
        print(f"Env state: {getattr(env, 'state', None)}")
        raise
    finally:
        teardown_test_env(env, agent)


@pytest.mark.integration
def test_agent_environment_interface():
    """Test agent ↔ environment compatibility."""
    env, agent = setup_test_env()
    try:
        obs, info = env.reset()
        # Robustly flatten observation for TD3: handle dict, tuple, array
        if isinstance(obs, tuple) and len(obs) > 0:
            obs = obs[0]
        if isinstance(obs, dict):
            obs = obs.get("market_features", obs)
        obs = np.asarray(obs)
        if obs.ndim > 1:
            obs = obs.flatten()
        action = agent.select_action(obs)
        next_obs, reward, done, _, _ = env.step(action)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
    except Exception as e:
        print(f"Error in agent-environment interface: {e}")
        print(f"Env state: {getattr(env, 'state', None)}")
        raise
    finally:
        teardown_test_env(env, agent)


@pytest.mark.integration
def test_model_agent_interface():
    """Test model ↔ agent integration."""
    _, agent = setup_test_env()
    try:
        state = np.random.randn(agent.state_dim).astype(np.float32)
        action = agent.select_action(state)
        assert len(action) == agent.action_dim
        agent.store_experience(state, action, 1.0, state, False)
        if len(agent.replay_buffer) >= agent.batch_size:
            metrics = agent.train()
            assert isinstance(metrics, dict)
            assert "critic_1_loss" in metrics
    except Exception as e:
        print(f"Error in model-agent interface: {e}")
        raise
    finally:
        teardown_test_env(None, agent)


@pytest.mark.integration
def test_full_pipeline_integration():
    """Test full pipeline integration: Model ↔ Agent ↔ Environment."""
    env, agent = setup_test_env()
    try:
        obs, info = env.reset()
        for _ in range(5):
            # Always flatten the full observation window for TD3
            if isinstance(obs, dict):
                obs_state = obs.get("market_features", obs)
            else:
                obs_state = obs
            obs_state = np.asarray(obs_state)
            if obs_state.ndim > 1:
                obs_state = obs_state.flatten()
            action = agent.select_action(obs_state)
            next_obs, reward, done, _, _ = env.step(action)
            obs = next_obs
            if done:
                break
        if len(agent.replay_buffer) >= agent.batch_size:
            metrics = agent.train()
            assert isinstance(metrics, dict)
            assert "critic_1_loss" in metrics
    except Exception as e:
        print(f"Error in full pipeline integration: {e}")
        print(f"Env state: {getattr(env, 'state', None)}")
        raise
    finally:
        teardown_test_env(env, agent)
