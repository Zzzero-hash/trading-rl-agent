import numpy as np
import pandas as pd
import quantstats.stats as qs
import pytest

from trading_rl_agent.utils import metrics

pytestmark = pytest.mark.unit


def test_sharpe_ratio_with_risk_free():
    r = np.array([0.01, 0.02, -0.005])
    rf = 0.01
    expected = metrics._empyrical.sharpe_ratio(r, risk_free=rf)
    result = metrics.calculate_sharpe_ratio(r, risk_free_rate=rf)
    assert np.isclose(result, expected)


def test_drawdown_and_profit_factor():
    idx = pd.date_range("2020-01-01", periods=4)
    data = pd.Series([0.1, -0.05, 0.02, -0.02], index=idx)
    dd = metrics.calculate_max_drawdown(data)
    assert dd <= 0
    pf = metrics.calculate_profit_factor(data)
    assert np.isclose(pf, qs.profit_factor(data))
    win_rate = metrics.calculate_win_rate(data)
    assert np.isclose(win_rate, qs.win_rate(data))
