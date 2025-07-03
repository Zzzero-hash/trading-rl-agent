from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
try:
    import empyrical
except ImportError as e:
    raise ImportError(
        "The 'empyrical' library is required but not installed. Please install it using 'pip install empyrical'."
    ) from e

import pandas as pd

try:  # FinRL 0.4+
    from finrl.env.env_stocktrading import StockTradingEnv as _FinRLTradingEnv
except Exception:  # FinRL 0.3.x
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv as _FinRLTradingEnv


class TradingEnv(_FinRLTradingEnv):
    """Thin wrapper around FinRL's :class:`StockTradingEnv`."""

    def __init__(self, env_cfg: Dict[str, Any] | None = None, **kwargs: Any) -> None:
        cfg = {**(env_cfg or {}), **kwargs}

        data_paths: Iterable[str | Path] = cfg.get("dataset_paths", [])
        if isinstance(data_paths, (str, Path)):
            data_paths = [data_paths]
        if not data_paths:
            raise ValueError("dataset_paths is required")

        try:
            frames = [pd.read_csv(p) for p in data_paths]
            df = pd.concat(frames, ignore_index=True)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find data file: {e}")
        except pd.errors.EmptyDataError as e:
            raise ValueError(f"Empty or invalid CSV data: {e}")
        except Exception as e:
            raise ValueError(f"Error reading CSV data: {e}")

        if "tic" not in df.columns:
            df["tic"] = cfg.get("symbol", "TIC")
        if "date" not in df.columns:
            df["date"] = range(len(df))

        stock_dim = df["tic"].nunique()
        hmax = int(cfg.get("hmax", 1))
        initial_amount = float(cfg.get("initial_balance", 10_000))
        self.reward_type = cfg.get("reward_type", "profit")
        self.risk_penalty = float(cfg.get("risk_penalty", 0.1))
        num_stock_shares = cfg.get("num_stock_shares", [0] * stock_dim)
        buy_cost_pct = cfg.get("buy_cost_pct", [0.001] * stock_dim)
        sell_cost_pct = cfg.get("sell_cost_pct", [0.001] * stock_dim)
        reward_scaling = float(cfg.get("reward_scaling", 1.0))
        tech_indicators = cfg.get("tech_indicator_list", [])
        state_space = stock_dim * (len(tech_indicators) + 2) + 1
        action_space = stock_dim

        super().__init__(
            df=df,
            stock_dim=stock_dim,
            hmax=hmax,
            initial_amount=initial_amount,
            num_stock_shares=num_stock_shares,
            buy_cost_pct=buy_cost_pct,
            sell_cost_pct=sell_cost_pct,
            reward_scaling=reward_scaling,
            state_space=state_space,
            action_space=action_space,
            tech_indicator_list=tech_indicators,
        )

        self._return_history: list[float] = []

    def reset(self, **kwargs) -> tuple:
        self._return_history = []
        obs = super().reset(**kwargs)
        info = {}
        return obs, info

    def step(self, action: Any) -> tuple:
        obs, reward, done, truncated, info = super().step(action)

        if self.reward_type == "sharpe":
            self._return_history.append(float(reward))
            if len(self._return_history) > 1 and np.std(self._return_history) != 0:
                sharpe = empyrical.sharpe_ratio(np.array(self._return_history))
                reward = float(sharpe) if np.isfinite(sharpe) else 0.0
            else:
                reward = 0.0
        elif self.reward_type == "risk_adjusted":
            self._return_history.append(float(reward))
            arr = np.array(self._return_history)
            mean_r = arr.mean()
            vol = arr.std()
            reward = float(mean_r - self.risk_penalty * vol)
            reward *= self.reward_scaling

        return obs, float(reward), done, truncated, info
def env_creator(env_cfg: Dict[str, Any] | None = None) -> TradingEnv:
    return TradingEnv(env_cfg)


def register_env(name: str = "TradingEnv") -> None:
    from ray.tune.registry import register_env as ray_register_env

    ray_register_env(name, lambda cfg: TradingEnv(cfg))
