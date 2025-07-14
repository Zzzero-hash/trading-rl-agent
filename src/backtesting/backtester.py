from __future__ import annotations

from typing import TYPE_CHECKING

import backtrader as bt
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class Backtester:
    """Simple Backtrader-based wrapper supporting slippage and latency."""

    def __init__(self, slippage_pct: float = 0.0, latency_seconds: float = 0.0) -> None:
        self.slippage_pct = slippage_pct
        self.latency_seconds = latency_seconds

    def run_backtest(
        self,
        prices: Sequence[float],
        policy: Callable[[float], str],
    ) -> pd.Series:
        """Run a backtest over a sequence of prices using ``policy``.

        Parameters
        ----------
        prices : Sequence[float]
            Price series.
        policy : Callable[[float], str]
            Function mapping an observed price to an action: ``"buy"``, ``"sell"`` or ``"hold"``.

        Returns
        -------
        pandas.Series
            Series containing a ``"Return [%]"`` key with total return.
        """

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
            },
            index=pd.date_range("2000-01-01", periods=len(prices)),
        )

        latency_steps = round(self.latency_seconds)

        class PolicyStrategy(bt.Strategy):
            params = {"policy": None, "latency": 0}

            def __init__(self) -> None:
                self.policy = self.p.policy
                self.latency = self.p.latency

            def next(self) -> None:
                if len(self.data) <= self.latency:
                    return
                price_obs = self.data.close[-self.latency - 1] if self.latency else self.data.close[-1]
                action = self.policy(price_obs)
                if action == "buy":
                    self.buy()
                elif action == "sell":
                    self.sell()

        cerebro = bt.Cerebro()
        cerebro.adddata(bt.feeds.PandasData(dataname=df))
        cerebro.addstrategy(PolicyStrategy, policy=policy, latency=latency_steps)
        cerebro.broker.set_slippage_perc(self.slippage_pct)
        cerebro.broker.setcommission(commission=self.slippage_pct)
        if latency_steps == 0:
            cerebro.broker.set_coc(True)
        else:
            cerebro.broker.set_coc(False)

        initial_cash = 100.0
        cerebro.broker.setcash(initial_cash)
        cerebro.run()
        final_value = cerebro.broker.getvalue()
        return_pct = (final_value - initial_cash) / initial_cash * 100
        return pd.Series({"Return [%]": return_pct})
