import pandas as pd
import numpy as np

from src.envs.trading_env import TradingEnv


def create_env(tmp_path):
    df = pd.DataFrame({
        "open": np.arange(6, dtype=float),
        "high": np.arange(6, dtype=float),
        "low": np.arange(6, dtype=float),
        "close": np.arange(6, dtype=float),
        "volume": np.ones(6),
    })
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "window_size": 2, "initial_balance": 500, "transaction_cost": 0.0}
    return TradingEnv(cfg)


def test_reset_restores_state(tmp_path):
    env = create_env(tmp_path)
    env.reset()
    env.step(1)
    assert env.position != 0
    assert env.balance != env.initial_balance

    obs, _ = env.reset()
    assert env.position == 0
    assert env.balance == env.initial_balance
    assert env.current_step == env.window_size
    assert obs.shape == (env.window_size, env.data.shape[1])


def test_reset_with_seed_reproducible(tmp_path):
    env = create_env(tmp_path)
    env.reset(seed=123)
    val1 = env.np_random.random()
    env.reset(seed=123)
    val2 = env.np_random.random()
    assert np.isclose(val1, val2)

