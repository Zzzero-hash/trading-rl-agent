import numpy as np
import pandas as pd

from trading_rl_agent.envs.finrl_trading_env import TradingEnv


def create_env(tmp_path):
    df = pd.DataFrame(
        {
            "open": np.arange(6, dtype=float),
            "high": np.arange(6, dtype=float),
            "low": np.arange(6, dtype=float),
            "close": np.arange(6, dtype=float),
            "volume": np.ones(6),
        },
    )
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "initial_balance": 500}
    return TradingEnv(cfg)


def test_reset_restores_state(tmp_path):
    env = create_env(tmp_path)
    env.reset()
    env.step(np.zeros(env.action_space.shape))
    assert env.day > 0

    obs, _ = env.reset()
    assert env.day == 0
    assert isinstance(obs, list | np.ndarray)
