import numpy as np
import pandas as pd
import quantstats.stats as qs

from trading_rl_agent.utils import metrics


def _sample_series():
    data = [0.1, -0.05, 0.2, -0.1]
    idx = pd.date_range("2020-01-01", periods=len(data))
    return pd.Series(data, index=idx)


def test_profit_factor_matches_quantstats():
    series = _sample_series()
    expected = qs.profit_factor(series)
    assert np.isclose(metrics.calculate_profit_factor(series), expected)


def test_win_rate_matches_quantstats():
    series = _sample_series()
    expected = qs.win_rate(series)
    assert np.isclose(metrics.calculate_win_rate(series), expected)


def test_avg_win_loss_ratio_matches_quantstats():
    series = _sample_series()
    expected = qs.win_loss_ratio(series)
    assert np.isclose(metrics.calculate_average_win_loss_ratio(series), expected)
