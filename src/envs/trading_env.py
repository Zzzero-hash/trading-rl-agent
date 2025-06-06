"""Custom trading environment using Gym interface compatible with RLlib."""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from pathlib import Path

from src.data.features import generate_features


class TradingEnv(gym.Env):
    """A simple trading environment with configurable parameters."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, env_cfg: dict):
        cfg = env_cfg or {}
        self.data_paths = cfg.get("dataset_paths", [])
        if isinstance(self.data_paths, str):
            self.data_paths = [self.data_paths]

        self.window_size = int(cfg.get("window_size", 50))
        self.initial_balance = float(cfg.get("initial_balance", 1_0000))
        self.transaction_cost = float(cfg.get("transaction_cost", 0.001))
        self.include_features = bool(cfg.get("include_features", False))

        self.data = self._load_data()
        if len(self.data) <= self.window_size:
            raise ValueError("Not enough data for the specified window_size")

        self.action_space = gym.spaces.Discrete(3)  # hold/buy/sell
        obs_shape = (self.window_size, self.data.shape[1])
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

        self.reset()

    # ------------------------------------------------------------------
    def _load_data(self) -> pd.DataFrame:
        frames = []
        for path in self.data_paths:
            df = pd.read_csv(path)
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)
        if self.include_features:
            data = generate_features(data)
        return data.astype(np.float32)

    def _get_observation(self) -> np.ndarray:
        obs = self.data.iloc[self.current_step - self.window_size : self.current_step].values
        return obs.astype(np.float32)

    # Gym API -----------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = float(self.initial_balance)
        self.position = 0
        return self._get_observation(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), "Invalid action"
        prev_price = self.data.loc[self.current_step - 1, "close"]

        new_position = {0: self.position, 1: 1, 2: -1}[action]
        cost = 0.0
        if new_position != self.position:
            cost = self.transaction_cost * abs(new_position - self.position)
        self.position = new_position

        self.current_step += 1
        done = self.current_step >= len(self.data)
        current_price = self.data.loc[self.current_step - 1, "close"]
        price_diff = float(current_price - prev_price)
        reward = float(self.position * price_diff - cost)
        self.balance += reward

        obs = self._get_observation() if not done else np.zeros_like(self.observation_space.sample())
        info = {"balance": self.balance}
        return obs, reward, done, False, info

    def render(self):
        print(
            f"Step: {self.current_step}, Price: {self.data.loc[self.current_step - 1, 'close']}, "
            f"Position: {self.position}, Balance: {self.balance}"
        )


# Registration helpers ---------------------------------------------------------

def env_creator(env_cfg):
    return TradingEnv(env_cfg)


def register_env(name: str = "TradingEnv"):
    from ray.tune.registry import register_env as ray_register_env

    ray_register_env(name, lambda cfg: TradingEnv(cfg))
