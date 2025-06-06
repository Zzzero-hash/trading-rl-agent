import os
import sys
import numpy as np
import pandas as pd
import gymnasium as gym

# Patch gym before importing the environment module
sys.modules["gym"] = gym

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from src.envs.trading_env import env_creator

class FlattenWrapper(gym.ObservationWrapper):
    """Flatten observations so RLlib can handle 2D arrays."""
    def __init__(self, env):
        super().__init__(env)
        shape = int(np.prod(env.observation_space.shape))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(shape,), dtype=np.float32)
    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32).flatten()

def register_flat_env():
    register_env("FlatTraderEnv", lambda cfg: FlattenWrapper(env_creator(cfg)))

def test_ray_trainer_checkpoint(tmp_path):
    sys.modules["gym"] = gym  # trader_env expects gym
    df = pd.DataFrame({
        "open": [1.0] * 60,
        "high": [1.0] * 60,
        "low": [1.0] * 60,
        "close": [1.0] * 60,
        "volume": [1.0] * 60,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    register_flat_env()
    ray.init(log_to_driver=False)
    config = (
        PPOConfig()
        .environment("FlatTraderEnv", env_config={"dataset_paths": [str(csv_path)], "window_size": 10})
        .env_runners(num_env_runners=0)
        .framework("torch")
    )

    algo = config.build()
    algo.train()
    chk_dir = tmp_path / "chkpt"
    algo.save("file://" + str(chk_dir))
    ray.shutdown()

    assert chk_dir.is_dir()
