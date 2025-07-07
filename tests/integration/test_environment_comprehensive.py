"""Simplified environment tests using the mock trading environment."""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


def test_reset_returns_observation(mock_trading_env):
    obs, info = mock_trading_env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)
    assert obs.shape == mock_trading_env.observation_space.shape


def test_step_returns_expected_structure(mock_trading_env):
    action = np.zeros(mock_trading_env.action_space.shape)
    obs, reward, terminated, truncated, info = mock_trading_env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


@pytest.mark.slow
def test_multiple_steps(mock_trading_env):
    mock_trading_env.reset()
    for _ in range(100):
        action = np.zeros(mock_trading_env.action_space.shape)
        obs, reward, terminated, truncated, _ = mock_trading_env.step(action)
        assert isinstance(obs, np.ndarray)
        if terminated or truncated:
            break
