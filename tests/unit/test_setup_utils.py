"""
Test setup utilities for integration tests.
"""

import numpy as np

from trading_rl_agent.agents.configs import TD3Config
from trading_rl_agent.agents.td3_agent import TD3Agent
from trading_rl_agent.envs.finrl_trading_env import TradingEnv
from tests.unit.test_data_utils import TestDataManager, get_dynamic_test_config


def setup_test_env(env_cfg=None, td3_config=None):
    if env_cfg is None:
        # Use dynamic test configuration
        env_cfg = get_dynamic_test_config()

    if td3_config is None:
        td3_config = TD3Config(batch_size=8, buffer_capacity=32)

    env = TradingEnv(env_cfg)
    # Determine state_dim (flattened for continuous action/TD3)
    obs, _ = env.reset()
    if isinstance(obs, dict):
        obs = obs.get("market_features", obs)
    obs = np.asarray(obs)
    if obs.ndim > 1:
        obs = obs.flatten()
    state_dim = obs.shape[0]
    # For continuous action, force action_dim=1 (Box shape)
    if (
        hasattr(env.action_space, "shape")
        and env.action_space.shape is not None
        and len(env.action_space.shape) > 0
    ):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 1
    agent = TD3Agent(config=td3_config, state_dim=state_dim, action_dim=action_dim)
    return env, agent


def teardown_test_env(env, agent):
    """Clean up test environment and any associated test data."""
    # Cleanup test data if manager was stored in config
    if hasattr(env, "config") and "_test_manager" in env.config:
        test_manager = env.config["_test_manager"]
        if isinstance(test_manager, TestDataManager):
            test_manager.cleanup()

    del env
    del agent
