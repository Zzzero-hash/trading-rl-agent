from src.eval import Backtester


def test_backtester_stub_methods():
    bt = Backtester()

    result = bt.run_backtest(prices=[1, 2, 3], policy=lambda p: "hold")
    assert result is None

    assert bt.apply_slippage(100.0) == 100.0
    assert bt.apply_latency(0.1) == 0.1
