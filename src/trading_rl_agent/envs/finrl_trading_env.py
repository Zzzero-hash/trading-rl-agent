from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

if TYPE_CHECKING:
    from collections.abc import Iterable


class TradingEnv(gym.Env):
    """A pure Gymnasium environment for stock trading."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__()
        cfg = {**(env_cfg or {}), **kwargs}

        data_paths: Iterable[str | Path] = cfg.get("dataset_paths", [])
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError("dataset_paths is required")

        try:
            frames = [pd.read_csv(p) for p in data_paths]
            self.df = pd.concat(frames, ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            raise ValueError(f"Error loading data: {e}") from e

        self.feature_columns = [c for c in self.df.columns if c not in ["date", "tic"]]
        self.stock_dim = self.df["tic"].nunique() if "tic" in self.df.columns else 1
        self.hmax = int(cfg.get("hmax", 100))
        self.initial_amount = float(cfg.get("initial_capital", 1_000_000))
        self.buy_cost_pct = cfg.get("buy_cost_pct", 0.001)
        self.sell_cost_pct = cfg.get("sell_cost_pct", 0.001)
        self.reward_scaling = float(cfg.get("reward_scaling", 1.0))
        self.reward_type = cfg.get("reward_type", "profit")

        if self.reward_type not in ["profit", "sharpe", "risk_adjusted"]:
            raise ValueError(f"Invalid reward_type: {self.reward_type}")

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.stock_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1 + self.stock_dim + len(self.feature_columns) * self.stock_dim,),
            dtype=np.float32,
        )

        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._get_initial_state()
        self.terminal = False
        self.rewards_memory: list[float] = []
        self.asset_memory: list[float] = [self.initial_amount]

    def _get_initial_state(self) -> np.ndarray:
        cash = np.array([self.initial_amount], dtype=np.float32)
        shares = np.zeros(self.stock_dim, dtype=np.float32)
        market_features = self.df.loc[self.day, self.feature_columns].values.flatten().astype(np.float32)
        return np.hstack([cash, shares, market_features])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._get_initial_state()
        self.terminal = False
        self.rewards_memory = []
        self.asset_memory = [self.initial_amount]
        return self.state, {}

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Executes one time step within the environment.

        Args:
            actions (np.ndarray): an action provided by the agent

        Returns:
            tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
                A tuple containing the new state, reward, terminated, truncated, and info.
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            return self.state, 0.0, True, False, {}

        actions = actions * self.hmax
        begin_total_asset = self.state[0] + (self.state[1 : 1 + self.stock_dim] * self.data.close).sum()

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            self._sell_stock(index, actions[index])
        for index in buy_index:
            self._buy_stock(index, actions[index])

        self.day += 1
        self.data = self.df.loc[self.day, :]
        self.state = self._update_state()

        end_total_asset = self.state[0] + (self.state[1 : 1 + self.stock_dim] * self.data.close).sum()
        self.asset_memory.append(end_total_asset)

        if self.reward_type == "profit":
            reward = (end_total_asset - begin_total_asset) * self.reward_scaling
        elif self.reward_type == "sharpe":
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
            reward = sharpe * self.reward_scaling
        elif self.reward_type == "risk_adjusted":
            returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
            drawdown = 1 - end_total_asset / np.maximum.accumulate(self.asset_memory)[-1]
            reward = sharpe - drawdown * 0.5 * self.reward_scaling
        else:
            reward = 0.0

        self.rewards_memory.append(reward)

        return self.state, reward, self.terminal, False, {}

    def _sell_stock(self, index: int, action: float) -> None:
        if self.state[index + 1] > 0:
            sell_num_shares = min(abs(action), self.state[index + 1])
            close_price = self.data.close if self.stock_dim == 1 else self.data.close.iloc[index]
            sell_amount = close_price * sell_num_shares * (1 - self.sell_cost_pct)
            self.state[0] += sell_amount
            self.state[index + 1] -= sell_num_shares

    def _buy_stock(self, index: int, action: float) -> None:
        close_price = self.data.close if self.stock_dim == 1 else self.data.close.iloc[index]
        if close_price > 0:
            available_amount = self.state[0] / (close_price * (1 + self.buy_cost_pct))
            buy_num_shares = min(available_amount, action)
            buy_amount = close_price * buy_num_shares * (1 + self.buy_cost_pct)
            self.state[0] -= buy_amount
            self.state[index + 1] += buy_num_shares

    def _update_state(self) -> np.ndarray:
        cash = np.array([self.state[0]], dtype=np.float32)
        shares = self.state[1 : 1 + self.stock_dim]
        market_features = self.df.loc[self.day, self.feature_columns].values.flatten().astype(np.float32)
        return np.hstack([cash, shares, market_features])

    def render(self, mode: str = "human") -> np.ndarray:
        return self.state


def register_env(name: str = "TradingEnv") -> None:
    from ray.tune.registry import register_env as ray_register_env

    ray_register_env(name, lambda cfg: TradingEnv(cfg))
