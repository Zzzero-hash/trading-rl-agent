import numpy as np
import pandas as pd
import pytest

from src.envs.trading_env import TradingEnv


def make_env(tmp_path, **overrides):
    df = pd.DataFrame(
        {
            "open": np.arange(6, dtype=float),
            "high": np.arange(6, dtype=float) + 1,
            "low": np.arange(6, dtype=float) - 1,
            "close": np.arange(6, dtype=float),
            "volume": np.ones(6),
        }
    )
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "window_size": 2}
    cfg.update(overrides)
    return TradingEnv(cfg)


def test_invalid_action_shape(tmp_path):
    env = make_env(tmp_path)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.array([1, 2]))


def test_step_when_done_returns_terminal_obs(tmp_path):
    env = make_env(tmp_path)
    env.reset()
    env.current_step = len(env.data) + 1
    obs, reward, done, _, info = env.step(0)
    assert done
    assert np.allclose(obs, 0)
    assert info["balance"] == env.balance
