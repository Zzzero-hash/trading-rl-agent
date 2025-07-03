import pandas as pd
import pytest

from src.backtesting import Backtester


def test_backtester_runs_basic_strategy():
    bt = Backtester()
    result = bt.run_backtest(prices=[1, 2, 3], policy=lambda p: "buy")
    assert isinstance(result, pd.Series)
    assert "Return [%]" in result


def test_backtester_slippage_and_latency():
    bt = Backtester(slippage_pct=0.1, latency_seconds=1)
    assert bt.apply_slippage(100.0) == pytest.approx(110.0)
    assert bt.apply_latency(0.0) == pytest.approx(1.0)
