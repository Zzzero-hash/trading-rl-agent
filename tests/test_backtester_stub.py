import pytest

from src.eval import Backtester


def test_backtester_event_loop_without_adjustments():
    bt = Backtester()

    policy = lambda p: "buy" if p > 1 else "sell"
    result = bt.run_backtest(prices=[1, 2, 3], policy=policy)

    assert result == [
        {"price": 1, "action": "sell", "delay": 0.0},
        {"price": 2, "action": "buy", "delay": 0.0},
        {"price": 3, "action": "buy", "delay": 0.0},
    ]


def test_backtester_with_slippage_and_latency():
    bt = Backtester(slippage_pct=0.1, latency_seconds=0.5)

    result = bt.run_backtest(prices=[10, 20], policy=lambda p: "hold")

    assert result == [
        {"price": 11.0, "action": "hold", "delay": 0.5},
        {"price": 22.0, "action": "hold", "delay": 0.5},
    ]

    assert bt.apply_slippage(100.0) == pytest.approx(110.0)
    assert bt.apply_latency(0.0) == pytest.approx(0.5)
