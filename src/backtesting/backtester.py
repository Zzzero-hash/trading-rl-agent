from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from backtesting import Backtest, Strategy
import pandas as pd


class Backtester:
    """Thin wrapper around :class:`backtesting.Backtest` with slippage and latency."""

    def __init__(self, slippage_pct: float = 0.0, latency_seconds: float = 0.0) -> None:
        self.slippage_pct = slippage_pct
        self.latency_seconds = latency_seconds

    def run_backtest(self, prices: Sequence[float], policy: Callable[[float], str]):
        """Run a backtest over a sequence of prices using ``policy``.

        Parameters
        ----------
        prices : Sequence[float]
            Price series.
        policy : Callable[[float], str]
            Function mapping an observed price to an action: ``"buy"``, ``"sell"`` or ``"hold"``.

        Returns
        -------
        backtesting._stats._Stats
            Results returned by :meth:`backtesting.Backtest.run`.
        """

        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices,
                "Low": prices,
                "Close": prices,
            },
            index=pd.RangeIndex(len(prices)),
        )

        latency_steps = int(self.latency_seconds)
        slippage = self.slippage_pct
        policy_fn = policy

        class PolicyStrategy(Strategy):
            def init(self):
                pass

            def next(self):
                if latency_steps and self.i - latency_steps < 0:
                    return
                price_obs = (
                    self.data.Close[-latency_steps - 1]
                    if latency_steps
                    else self.data.Close[-1]
                )
                action = policy_fn(price_obs)
                if action == "buy":
                    self.buy()
                elif action == "sell":
                    self.sell()

        bt = Backtest(df, PolicyStrategy, commission=slippage)
        stats = bt.run()
        return pd.Series(stats._stats)

    def apply_slippage(self, price: float) -> float:
        """Apply the configured slippage percentage to ``price``."""
        return price * (1 + self.slippage_pct)

    def apply_latency(self, delay: float) -> float:
        """Apply the configured latency to ``delay``."""
        return delay + self.latency_seconds
