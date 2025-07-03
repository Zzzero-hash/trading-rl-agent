import numpy as np
import pandas as pd

from src.envs.finrl_trading_env import TradingEnv


def make_env(tmp_path, closes, tc=0.1):
    df = pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": np.ones(len(closes)),
        }
    )
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "initial_balance": 1000}
    return TradingEnv(cfg)


def test_reward_computation_buy_hold_sell(tmp_path):
    closes = np.arange(1.0, 7.0)
    env = make_env(tmp_path, closes)

    env.reset()

    _, reward_buy, *_ = env.step(np.array([0]))
    assert isinstance(reward_buy, (int, float, np.floating))

    _, reward_hold, *_ = env.step(np.array([0]))
    assert isinstance(reward_hold, (int, float, np.floating))

    _, reward_sell, *_ = env.step(np.array([0]))
    assert isinstance(reward_sell, (int, float, np.floating))
