import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.envs.finrl_trading_env import TradingEnv


def make_env(tmp_path, reward_type="profit"):
    df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [1, 2, 3, 4],
            "low": [1, 2, 3, 4],
            "close": [1, 2, 3, 4],
            "volume": [1, 1, 1, 1],
        }
    )
    csv = tmp_path / "prices.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "reward_type": reward_type}
    return TradingEnv(cfg)


def test_reset_and_step_returns(tmp_path):
    env = make_env(tmp_path)
    obs, info = env.reset()
    assert isinstance(obs, (list, np.ndarray))
    assert info == {}
    action = np.zeros(env.action_space.shape)
    next_obs, reward, done, truncated, _ = env.step(action)
    assert isinstance(reward, float)
    assert next_obs is not None
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)


def test_sharpe_reward(tmp_path):
    env = make_env(tmp_path, reward_type="sharpe")
    env.reset()
    _, r1, *_ = env.step(np.zeros(env.action_space.shape))
    assert r1 == 0.0
    _, r2, *_ = env.step(np.zeros(env.action_space.shape))
    assert isinstance(r2, float)


def test_risk_adjusted_reward(tmp_path):
    env = make_env(tmp_path, reward_type="risk_adjusted")
    env.reset()
    env.step(np.zeros(env.action_space.shape))
    _, r2, *_ = env.step(np.zeros(env.action_space.shape))
    assert isinstance(r2, float)


def test_invalid_reward_type(tmp_path):
    df = pd.DataFrame(
        {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}
    )
    csv = tmp_path / "d.csv"
    df.to_csv(csv, index=False)
    cfg = {"dataset_paths": [str(csv)], "reward_type": "unknown"}
    with pytest.raises(ValueError):
        TradingEnv(cfg)


def test_missing_dataset_paths():
    with pytest.raises(ValueError):
        TradingEnv({})


def test_file_not_found(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(FileNotFoundError):
        TradingEnv({"dataset_paths": [str(missing)]})
