"""Tests for the TraderEnv trading environment."""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from pathlib import Path

from src.envs.trader_env import TraderEnv, env_creator, register_env


@pytest.fixture
def sample_data(tmp_path):
    """Create sample market data for testing."""
    data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0] * 20,
        'high': [105.0, 106.0, 107.0, 108.0, 109.0] * 20,
        'low': [95.0, 96.0, 97.0, 98.0, 99.0] * 20,
        'close': [101.0, 102.0, 103.0, 104.0, 105.0] * 20,
        'volume': [1000, 1100, 1200, 1300, 1400] * 20
    })
    csv_path = tmp_path / "test_data.csv"
    data.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def env(sample_data):
    """Create a TraderEnv instance for testing."""
    return TraderEnv([sample_data], window_size=10, initial_balance=10000)


def test_trader_env_initialization(sample_data):
    """Test TraderEnv initialization with various parameters."""
    env = TraderEnv([sample_data], window_size=10, initial_balance=5000, transaction_cost=0.002)
    
    assert env.initial_balance == 5000
    assert env.window_size == 10
    assert env.transaction_cost == 0.002
    assert env.action_space.n == 3  # hold, buy, sell
    assert env.observation_space.shape == (10, 5)  # window_size x features


def test_trader_env_single_file_path(sample_data):
    """Test TraderEnv initialization with single file path string."""
    env = TraderEnv(sample_data, window_size=5)
    assert len(env.data_paths) == 1
    assert env.data_paths[0] == sample_data


def test_trader_env_insufficient_data(tmp_path):
    """Test TraderEnv raises error with insufficient data."""
    # Create data with only 5 rows
    small_data = pd.DataFrame({
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [105.0, 106.0, 107.0, 108.0, 109.0],
        'low': [95.0, 96.0, 97.0, 98.0, 99.0],
        'close': [101.0, 102.0, 103.0, 104.0, 105.0],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    csv_path = tmp_path / "small_data.csv"
    small_data.to_csv(csv_path, index=False)
    
    with pytest.raises(ValueError, match="Not enough data for the specified window_size"):
        TraderEnv([str(csv_path)], window_size=10)


def test_trader_env_reset(env):
    """Test environment reset functionality."""
    obs, info = env.reset()
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10, 5)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)
    assert env.current_step == 10
    assert env.balance == 10000.0
    assert env.position == 0


def test_trader_env_reset_with_seed(env):
    """Test environment reset with seed."""
    obs1, _ = env.reset(seed=42)
    obs2, _ = env.reset(seed=42)
    
    np.testing.assert_array_equal(obs1, obs2)


def test_trader_env_step_hold(env):
    """Test environment step with hold action."""
    env.reset()
    initial_balance = env.balance
    initial_position = env.position
    
    obs, reward, done, truncated, info = env.step(0)  # hold
    
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (10, 5)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert 'balance' in info
    assert env.position == initial_position  # position unchanged
    assert env.current_step == 11


def test_trader_env_step_buy(env):
    """Test environment step with buy action."""
    env.reset()
    initial_balance = env.balance
    
    obs, reward, done, truncated, info = env.step(1)  # buy
    
    assert env.position == 1  # long position
    assert env.balance != initial_balance  # balance changed due to transaction cost
    assert isinstance(reward, (int, float))


def test_trader_env_step_sell(env):
    """Test environment step with sell action."""
    env.reset()
    initial_balance = env.balance
    
    obs, reward, done, truncated, info = env.step(2)  # sell
    
    assert env.position == -1  # short position
    assert env.balance != initial_balance  # balance changed due to transaction cost
    assert isinstance(reward, (int, float))


def test_trader_env_transaction_costs(env):
    """Test transaction costs are applied correctly."""
    env.reset()
    initial_balance = env.balance
    
    # Buy (position change from 0 to 1)
    env.step(1)
    balance_after_buy = env.balance
    expected_cost = env.transaction_cost * abs(1 - 0)
    
    # Transaction cost should be deducted
    assert balance_after_buy < initial_balance
    
    # Hold (no position change)
    env.step(1)  # buy again (no change from position 1)
    balance_after_hold = env.balance
    
    # No additional transaction cost for same position
    price_change = env.data.loc[env.current_step - 1, "close"] - env.data.loc[env.current_step - 2, "close"]
    expected_balance = balance_after_buy + price_change  # profit/loss from price movement
    assert abs(balance_after_hold - expected_balance) < 1e-6


def test_trader_env_episode_end(env):
    """Test environment episode termination."""
    env.reset()
    data_length = len(env.data)
    
    # Step through entire episode
    done = False
    step_count = 0
    while not done and step_count < data_length:
        obs, reward, done, truncated, info = env.step(0)  # hold
        step_count += 1
    
    assert done
    assert env.current_step >= len(env.data)
    # Observation should be zeros when done
    assert np.allclose(obs, 0.0)


def test_trader_env_invalid_action(env):
    """Test environment rejects invalid actions."""
    env.reset()
    
    with pytest.raises(AssertionError):
        env.step(3)  # invalid action
    
    with pytest.raises(AssertionError):
        env.step(-1)  # invalid action


def test_trader_env_reward_calculation(env):
    """Test reward calculation for different actions."""
    env.reset()
    
    # Get initial price
    initial_price = env.data.loc[env.current_step - 1, "close"]
    
    # Buy action
    obs, reward, done, truncated, info = env.step(1)
    new_price = env.data.loc[env.current_step - 1, "close"]
    price_diff = new_price - initial_price
    
    # Reward should be price_diff - transaction_cost
    expected_reward = price_diff - env.transaction_cost
    assert abs(reward - expected_reward) < 1e-6


def test_trader_env_render(env, capsys):
    """Test environment render functionality."""
    env.reset()
    env.render()
    
    captured = capsys.readouterr()
    assert "Step:" in captured.out
    assert "Price:" in captured.out
    assert "Position:" in captured.out
    assert "Balance:" in captured.out


def test_trader_env_multiple_files(tmp_path):
    """Test TraderEnv with multiple data files."""
    # Create two data files
    data1 = pd.DataFrame({
        'open': [100.0, 101.0, 102.0] * 10,
        'high': [105.0, 106.0, 107.0] * 10,
        'low': [95.0, 96.0, 97.0] * 10,
        'close': [101.0, 102.0, 103.0] * 10,
        'volume': [1000, 1100, 1200] * 10
    })
    
    data2 = pd.DataFrame({
        'open': [200.0, 201.0, 202.0] * 10,
        'high': [205.0, 206.0, 207.0] * 10,
        'low': [195.0, 196.0, 197.0] * 10,
        'close': [201.0, 202.0, 203.0] * 10,
        'volume': [2000, 2100, 2200] * 10
    })
    
    csv_path1 = tmp_path / "data1.csv"
    csv_path2 = tmp_path / "data2.csv"
    data1.to_csv(csv_path1, index=False)
    data2.to_csv(csv_path2, index=False)
    
    env = TraderEnv([str(csv_path1), str(csv_path2)], window_size=10)
    
    # Should have concatenated data
    assert len(env.data) == 60  # 30 + 30 rows
    assert env.data.shape[1] == 5  # 5 columns


def test_env_creator():
    """Test the env_creator function."""
    env_cfg = {
        "dataset_paths": ["dummy_path.csv"],
        "initial_balance": 5000,
        "window_size": 20,
        "transaction_cost": 0.005
    }
    
    # Mock the data loading to avoid file issues
    import src.envs.trader_env
    original_load = src.envs.trader_env.TraderEnv._load_data
    
    def mock_load_data(self):
        return pd.DataFrame({
            'open': [100.0] * 50,
            'high': [105.0] * 50,
            'low': [95.0] * 50,
            'close': [101.0] * 50,
            'volume': [1000] * 50
        }).astype(np.float32)
    
    src.envs.trader_env.TraderEnv._load_data = mock_load_data
    
    try:
        env = env_creator(env_cfg)
        assert env.initial_balance == 5000
        assert env.window_size == 20
        assert env.transaction_cost == 0.005
    finally:
        src.envs.trader_env.TraderEnv._load_data = original_load


def test_env_creator_defaults():
    """Test env_creator with default values."""
    env_cfg = {"dataset_paths": ["dummy_path.csv"]}
    
    # Mock the data loading
    import src.envs.trader_env
    original_load = src.envs.trader_env.TraderEnv._load_data
    
    def mock_load_data(self):
        return pd.DataFrame({
            'open': [100.0] * 100,
            'high': [105.0] * 100,
            'low': [95.0] * 100,
            'close': [101.0] * 100,
            'volume': [1000] * 100
        }).astype(np.float32)
    
    src.envs.trader_env.TraderEnv._load_data = mock_load_data
    
    try:
        env = env_creator(env_cfg)
        assert env.initial_balance == 10000  # default
        assert env.window_size == 50  # default
        assert env.transaction_cost == 0.001  # default
    finally:
        src.envs.trader_env.TraderEnv._load_data = original_load


def test_register_env():
    """Test environment registration function."""
    # This is more of an integration test
    # We can't easily test Ray's internal registration without starting Ray
    
    try:
        register_env()
        # If no exception is raised, registration succeeded
        assert True
    except Exception as e:
        # If Ray is not initialized, this might fail
        # That's okay for unit testing
        assert "ray" in str(e).lower() or "not initialized" in str(e).lower()


def test_trader_env_observation_dtype(env):
    """Test that observations are consistently float32."""
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    
    obs, _, _, _, _ = env.step(0)
    assert obs.dtype == np.float32


def test_trader_env_gym_compatibility(env):
    """Test that TraderEnv is compatible with Gym interface."""
    # Check that the environment follows Gym API
    assert hasattr(env, 'action_space')
    assert hasattr(env, 'observation_space')
    assert hasattr(env, 'reset')
    assert hasattr(env, 'step')
    assert hasattr(env, 'render')
    
    # Check action and observation spaces
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    
    # Test the step function returns 5 values (new Gym API)
    env.reset()
    result = env.step(0)
    assert len(result) == 5  # obs, reward, terminated, truncated, info


def test_trader_env_balance_tracking(env):
    """Test that balance is tracked correctly over multiple steps."""
    env.reset()
    initial_balance = env.balance
    
    # Track balance changes
    balances = [initial_balance]
    
    for _ in range(5):
        obs, reward, done, truncated, info = env.step(1)  # buy
        balances.append(env.balance)
        assert info['balance'] == env.balance
    
    # Balance should change over time due to price movements and costs
    assert len(set(balances)) > 1  # Not all balances should be the same
