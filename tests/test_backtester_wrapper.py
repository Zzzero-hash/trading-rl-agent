import pandas as pd
import pytest

from src.backtesting import Backtester


def test_backtester_runs_basic_strategy():
    bt = Backtester()
    result = bt.run_backtest(prices=[1, 2, 3], policy=lambda p: "buy")
    assert isinstance(result, pd.Series)
    assert "Return [%]" in result


def test_backtester_slippage_and_latency():
    bt_no_slip = Backtester()
    res_no_slip = bt_no_slip.run_backtest(prices=[1, 2, 3], policy=lambda p: "buy")

    bt_slip = Backtester(slippage_pct=0.1)
    res_slip = bt_slip.run_backtest(prices=[1, 2, 3], policy=lambda p: "buy")

    bt_lat = Backtester(latency_seconds=1)
    res_lat = bt_lat.run_backtest(prices=[1, 2, 3, 4], policy=lambda p: "buy")

    assert res_slip["Return [%]"] < res_no_slip["Return [%]"]
    assert isinstance(res_lat, pd.Series)
