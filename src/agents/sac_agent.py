from __future__ import annotations

import numpy as np
from gymnasium import spaces
import gymnasium as gym
from stable_baselines3 import SAC


class DummyEnv(gym.Env):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class SACAgent:
    """Wrapper around :class:`stable_baselines3.SAC`."""

    def __init__(self, state_dim: int, action_dim: int, config: dict | None = None, device: str = "cpu"):
        config = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        env = DummyEnv(state_dim, action_dim)
        self.model = SAC(
            "MlpPolicy",
            env,
            verbose=0,
            device=device,
            learning_rate=config.get("learning_rate", 3e-4),
            buffer_size=config.get("buffer_capacity", 1000000),
            batch_size=config.get("batch_size", 256),
            gamma=config.get("gamma", 0.99),
            tau=config.get("tau", 0.005),
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        return self.model.predict(state, deterministic=evaluate)[0]

    def train(self, total_timesteps: int = 1) -> dict:
        self.model.learn(total_timesteps=total_timesteps)
        return {}

    update = train

    def save(self, filepath: str) -> None:
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        self.model = SAC.load(filepath)
