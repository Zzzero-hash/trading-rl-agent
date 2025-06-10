import pandas as pd
import numpy as np

from src.envs.trading_env import TradingEnv


def make_env(tmp_path, closes, tc=0.1):
    df = pd.DataFrame({
        "open": closes,
        "high": closes,
        "low": closes,
        "close": closes,
        "volume": np.ones(len(closes)),
    })
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "window_size": 2, "transaction_cost": tc, "initial_balance": 1000}
    return TradingEnv(cfg)


def test_reward_computation_buy_hold_sell(tmp_path):
    closes = np.arange(1.0, 7.0)
    env = make_env(tmp_path, closes)

    env.reset()

    # Buy
    _, reward_buy, _, _, _ = env.step(1)
    expected_buy = (closes[2] - closes[1]) - 0.1
    assert np.isclose(reward_buy, expected_buy)

    # Hold
    _, reward_hold, _, _, _ = env.step(0)
    expected_hold = (closes[3] - closes[2])
    assert np.isclose(reward_hold, expected_hold)

    # Sell
    _, reward_sell, _, _, _ = env.step(2)
    expected_sell = (-1 * (closes[4] - closes[3])) - 0.2
    assert np.isclose(reward_sell, expected_sell)

