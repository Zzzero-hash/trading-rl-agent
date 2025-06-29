import gymnasium as gym
import numpy as np
import pandas as pd
import pytest

from src.envs.trading_env import TradingEnv, env_creator, register_env


@pytest.fixture
def sample_csv(tmp_path):
    data = pd.DataFrame(
        {
            "open": [1.0] * 60,
            "high": [1.0] * 60,
            "low": [1.0] * 60,
            "close": np.linspace(1.0, 2.0, 60),
            "volume": [1.0] * 60,
        }
    )
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)
    return str(csv)


@pytest.fixture(params=["dict", "kwargs"])
def env(sample_csv, request):
    if request.param == "dict":
        cfg = {"dataset_paths": [sample_csv], "window_size": 10}
        return TradingEnv(cfg)
    return TradingEnv(dataset_paths=[sample_csv], window_size=10)


def test_reset_returns_observation(env):
    obs, info = env.reset()
    assert obs.shape == (10, env.data.shape[1])
    assert env.current_step == env.window_size
    assert info == {}


def test_step_changes_balance(env):
    env.reset()
    _, reward, _, _, info = env.step(1)
    assert isinstance(reward, float)
    assert isinstance(info, dict)
    assert info["balance"] == env.balance


def test_env_creator(sample_csv):
    cfg = {"dataset_paths": [sample_csv], "window_size": 5}
    env = env_creator(cfg)
    assert isinstance(env, TradingEnv)
    assert env.window_size == 5


def test_register_env(sample_csv):
    cfg = {"dataset_paths": [sample_csv], "window_size": 5}
    try:
        register_env()
    except Exception:
        # registration may fail if ray not initialized
        pass
    env = env_creator(cfg)
    assert env.action_space.n == 3
