from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

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

        frames = [pd.read_csv(p) for p in data_paths]
        df = pd.concat(frames, ignore_index=True)

        if "tic" not in df.columns:
            df["tic"] = cfg.get("symbol", "TIC")
        if "date" not in df.columns:
            df["date"] = range(len(df))

        stock_dim = df["tic"].nunique()
        hmax = int(cfg.get("hmax", 1))
        initial_amount = float(cfg.get("initial_balance", 10_000))
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


def env_creator(env_cfg: Dict[str, Any] | None = None) -> TradingEnv:
    return TradingEnv(env_cfg)


def register_env(name: str = "TradingEnv") -> None:
    from ray.tune.registry import register_env as ray_register_env

    ray_register_env(name, lambda cfg: TradingEnv(cfg))
