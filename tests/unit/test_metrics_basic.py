import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.utils import metrics

pytestmark = pytest.mark.unit


def test_calculate_sharpe_ratio_basic():
    returns = np.array([0.01, 0.02, 0.015])
    expected = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)
    assert np.isclose(metrics.calculate_sharpe_ratio(returns), expected)


def test_calculate_sharpe_ratio_zero_std():
    returns = np.array([0.0, 0.0, 0.0])
    assert np.isnan(metrics.calculate_sharpe_ratio(returns))


def test_calculate_max_drawdown():
    returns = np.array([0.0, -0.2, -0.25, 0.5, 0.333])
    assert np.isclose(metrics.calculate_max_drawdown(returns), -0.4)


def test_calculate_sortino_ratio():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    downside = np.clip(returns, -np.inf, 0)
    expected = returns.mean() * 252 / (np.sqrt(np.mean(downside**2)) * np.sqrt(252))
    result = metrics.calculate_sortino_ratio(returns)
    assert np.isclose(result, expected)


def test_calculate_risk_metrics_keys():
    returns = np.array([0.05, -0.02, 0.04, 0.03])
    m = metrics.calculate_risk_metrics(returns)
    expected_keys = {
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "profit_factor",
        "win_rate",
        "calmar_ratio",
        "total_return",
        "volatility",
        "num_trades",
    }
    assert expected_keys.issubset(m.keys())
