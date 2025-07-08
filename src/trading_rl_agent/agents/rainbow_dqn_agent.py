from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sb3_contrib import QRDQN


class DummyDiscreteEnv(gym.Env):
    """Minimal discrete environment for initializing SB3 agents."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)

    def reset(self, *, seed: int | None = None, options=None):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = True
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class RainbowDQNAgent:
    """Lightweight wrapper around :class:`sb3_contrib.QRDQN`."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: dict | None = None,
        device: str = "cpu",
    ):
        config = config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        env = DummyDiscreteEnv(state_dim, action_dim)
        self.model = QRDQN(
            "MlpPolicy",
            env,
            verbose=0,
            device=device,
            learning_rate=config.get("learning_rate", 1e-4),
            buffer_size=config.get("buffer_capacity", 100000),
            batch_size=config.get("batch_size", 32),
            gamma=config.get("gamma", 0.99),
            train_freq=config.get("train_freq", 4),
            target_update_interval=config.get("target_update_freq", 1000),
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        return int(self.model.predict(state, deterministic=evaluate)[0])

    def train(self, total_timesteps: int = 1) -> dict:
        self.model.learn(total_timesteps=total_timesteps)
        return {}

    update = train

    def save(self, filepath: str) -> None:
        self.model.save(filepath)

    def load(self, filepath: str) -> None:
        self.model = QRDQN.load(filepath)
