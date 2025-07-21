import numpy as np
import pandas as pd

from trade_agent.utils import metrics


def test_var_and_expected_shortfall():
    returns = np.array([0.1, -0.2, 0.05, -0.1, 0.02])
    var_95 = metrics.calculate_var(returns, confidence=0.95)
    es_95 = metrics.calculate_expected_shortfall(returns, confidence=0.95)
    # Manual computation using numpy
    expected_var = -np.quantile(returns, 0.05)
    expected_es = -returns[returns <= -expected_var].mean()
    assert np.isclose(abs(var_95), expected_var)
    assert np.isclose(abs(es_95), expected_es)


def test_information_ratio_and_tracking_error():
    np.random.seed(0)
    returns = np.random.normal(0.01, 0.02, 100)
    benchmark = np.random.normal(0.008, 0.015, 100)
    info = metrics.calculate_information_ratio(returns, benchmark)
    te = metrics.calculate_tracking_error(returns, benchmark)
    assert te > 0
    # Information ratio should equal mean(excess)/std(excess)*sqrt(252)
    excess = returns - benchmark
    expected_info = excess.mean() / excess.std() * np.sqrt(metrics.TRADING_DAYS_PER_YEAR)
    assert np.isclose(info, expected_info)


def test_comprehensive_metrics_keys():
    np.random.seed(1)
    r = np.random.normal(0, 0.01, 50)
    b = np.random.normal(0, 0.008, 50)
    result = metrics.calculate_comprehensive_metrics(r, b)
    expected_keys = {
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
        "var_95",
        "expected_shortfall",
        "profit_factor",
        "win_rate",
        "average_win_loss_ratio",
        "information_ratio",
        "tracking_error",
        "beta",
    }
    assert expected_keys.issubset(result.keys())


def test_to_series_and_beta():
    data = [0.01, 0.02]
    series = metrics._to_series(data)
    assert isinstance(series.index, pd.DatetimeIndex)
    bench = [0.0, 0.01]
    beta = metrics.calculate_beta(data, bench)
    assert isinstance(beta, float)


def test_information_ratio_zero_te():
    r = [0.01, 0.01, 0.01]
    b = [0.01, 0.01, 0.01]
    assert metrics.calculate_information_ratio(r, b) == 0.0


def test_calculate_risk_metrics():
    returns = np.array([0.01, -0.02, 0.03])
    metrics_dict = metrics.calculate_risk_metrics(returns)
    assert metrics_dict["num_trades"] == len(returns)
    assert "total_return" in metrics_dict
