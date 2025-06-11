"""
Test setup utilities for integration tests.
"""
import numpy as np
from src.envs.trading_env import TradingEnv
from src.agents.td3_agent import TD3Agent
from src.agents.configs import TD3Config

def setup_test_env(env_cfg=None, td3_config=None):
    if env_cfg is None:
        env_cfg = {
            "dataset_paths": ["data/sample_training_data_simple_20250607_192034.csv"],
            "window_size": 10,
            "initial_balance": 10000,
            "transaction_cost": 0.001,
            "include_features": False,
            "continuous_actions": True  # Ensure continuous for TD3
        }
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
    if hasattr(env.action_space, 'shape') and env.action_space.shape is not None and len(env.action_space.shape) > 0:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 1
    agent = TD3Agent(config=td3_config, state_dim=state_dim, action_dim=action_dim)
    return env, agent

def teardown_test_env(env, agent):
    del env
    del agent
