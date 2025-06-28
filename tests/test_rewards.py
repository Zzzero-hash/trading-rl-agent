import numpy as np

from src.utils.rewards import (
    simple_profit_reward,
    risk_adjusted_reward,
    drawdown_penalty_reward,
    sharpe_based_reward,
    portfolio_diversification_reward,
    transaction_cost_penalty,
    momentum_reward,
    custom_trading_reward,
)


def test_simple_profit_reward():
    assert np.isclose(simple_profit_reward(100.0, 110.0), 0.1)
    assert np.isclose(simple_profit_reward(0.0, 50.0), 0.0)


def test_risk_adjusted_reward_scalar_and_array():
    scalar_result = risk_adjusted_reward(0.05, volatility=0.02, risk_penalty=0.1)
    assert np.isclose(scalar_result, 0.05 - 0.1 * 0.02)

    returns = np.array([0.1, 0.2, 0.3])
    expected = np.mean(returns) - 0.1 * np.std(returns)
    array_result = risk_adjusted_reward(returns, risk_penalty=0.1)
    assert np.isclose(array_result, expected)


def test_drawdown_penalty_reward():
    assert np.isclose(drawdown_penalty_reward(0.05, 0.05), 0.05)
    penalized = drawdown_penalty_reward(0.05, 0.15)
    assert np.isclose(penalized, 0.05 - (0.15 - 0.1) * 2.0)


def test_sharpe_based_reward():
    returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    excess = returns - 0.0
    expected = np.mean(excess) / np.std(excess) * np.sqrt(252)
    result = sharpe_based_reward(returns)
    assert np.isclose(result, expected)


def test_portfolio_diversification_reward():
    weights = np.array([0.5, 0.5])
    reward = portfolio_diversification_reward(weights, diversification_bonus=0.1)
    assert np.isclose(reward, 0.05)


def test_transaction_cost_penalty():
    assert np.isclose(transaction_cost_penalty(1, 0, cost_rate=0.001), -0.001)
    arr_penalty = transaction_cost_penalty([1, 0], [0, 0], cost_rate=0.001)
    assert np.isclose(arr_penalty, -0.001)


def test_momentum_reward():
    changes = np.ones(5)
    assert np.isclose(momentum_reward(changes, 1, momentum_window=5), 0.1)
    assert np.isclose(momentum_reward(-changes, -1, momentum_window=5), 0.1)
    assert np.isclose(momentum_reward(changes, -1, momentum_window=5), -0.05)


def test_custom_trading_reward():
    weights = np.array([0.5, 0.5])
    reward = custom_trading_reward(
        portfolio_return=0.1,
        benchmark_return=0.05,
        volatility=0.02,
        drawdown=0.03,
        transaction_costs=0.01,
        weights=weights,
        risk_penalty=0.1,
        diversification_bonus=0.02,
    )
    expected = (
        0.1 - 0.05
        - 0.1 * 0.02
        - 0.0
        - 0.01
        + portfolio_diversification_reward(weights, diversification_bonus=0.02)
    )
    assert np.isclose(reward, expected)

