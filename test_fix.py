import numpy as np
import pandas as pd
import pytest

from src.utils import metrics

pytestmark = pytest.mark.unit


def test_calculate_sharpe_ratio_basic():
    returns = np.array([0.01, 0.02, 0.015])
    expected = np.mean(returns) / np.std(returns) * np.sqrt(252)
    assert np.isclose(metrics.calculate_sharpe_ratio(returns), expected)


def test_calculate_sharpe_ratio_zero_std():
    returns = np.array([0.0, 0.0, 0.0])
    assert metrics.calculate_sharpe_ratio(returns) == 0.0


def test_calculate_max_drawdown():
    equity = np.array([1.0, 0.8, 0.6, 0.9, 1.2])
    assert np.isclose(metrics.calculate_max_drawdown(equity), 0.4)


def test_calculate_sortino_ratio():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])
    excess = returns - 0.0
    downside = excess[excess < 0]
    expected = np.mean(excess) / np.sqrt(np.mean(downside**2)) * np.sqrt(252)
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
