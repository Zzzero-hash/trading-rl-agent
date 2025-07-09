import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.envs.finrl_trading_env import (
    TradingEnv,
    env_creator,
    register_env,
)


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(params=["dict", "kwargs"])
def env(sample_csv, request):
    cfg = {"dataset_paths": [sample_csv]}
    if request.param == "dict":
        return TradingEnv(cfg)
    return TradingEnv(**cfg)


def test_reset_and_step(env):
    obs, info = env.reset()
    assert isinstance(info, dict)
    result = env.step(np.zeros(env.action_space.shape))
    assert len(result) == 5


def test_env_creator(sample_csv):
    env = env_creator({"dataset_paths": [sample_csv]})
    assert isinstance(env, TradingEnv)


def test_register_env(sample_csv):
    register_env()
    env = env_creator({"dataset_paths": [sample_csv]})
    assert isinstance(env, TradingEnv)
