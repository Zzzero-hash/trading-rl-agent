import numpy as np
import pytest

from trading_rl_agent.envs.finrl_trading_env import TradingEnv

pytestmark = pytest.mark.unit


def _make_env(path):
    return TradingEnv({"dataset_paths": path, "reward_type": "sharpe"})


def test_reset_clears_history(sample_csv_file):
    env = _make_env(sample_csv_file)
    env.reset()
    env.step(np.zeros(env.action_space.shape))
    assert env._return_history  # history recorded
    env.reset()
    assert env._return_history == []


def test_dataset_path_string(sample_csv_file):
    env = TradingEnv({"dataset_paths": sample_csv_file})
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(info, dict)
