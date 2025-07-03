from __future__ import annotations

from typing import Callable, Sequence

import pandas as pd
from backtesting import Backtest, Strategy


class Backtester:
    """Thin wrapper around :class:`backtesting.Backtest` with slippage and latency."""

    def __init__(self, slippage_pct: float = 0.0, latency_seconds: float = 0.0) -> None:
        """
        Initialize a Backtester instance with optional slippage and latency settings.
        
        Parameters:
            slippage_pct (float, optional): The slippage percentage to apply to each trade. Defaults to 0.0.
            latency_seconds (float, optional): The latency in seconds to simulate delayed trade execution. Defaults to 0.0.
        """
        self.slippage_pct = slippage_pct
        self.latency_seconds = latency_seconds

    def run_backtest(self, prices: Sequence[float], policy: Callable[[float], str]):
        """
        Run a backtest on a sequence of prices using a user-defined trading policy.
        
        The policy function receives observed prices (optionally delayed by the configured latency) and returns an action: "buy", "sell", or "hold". The method simulates slippage as a commission and latency as a delay in price observation. Returns the backtest statistics as a pandas Series.
         
        Parameters:
            prices (Sequence[float]): Sequence of price values to backtest over.
            policy (Callable[[float], str]): Function mapping an observed price to a trading action ("buy", "sell", or "hold").
        
        Returns:
            pd.Series: Backtest statistics as a pandas Series.
        """

        df = pd.DataFrame({
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
        }, index=pd.RangeIndex(len(prices)))

        latency_steps = int(self.latency_seconds)
        slippage = self.slippage_pct
        policy_fn = policy

        class PolicyStrategy(Strategy):
            def init(self):
                """
                Initializes the strategy. No custom initialization is performed.
                """
                pass

            def next(self):
                """
                Executes a trading action at each step based on the policy function's output, accounting for configured latency.
                
                If latency is set and insufficient data is available for the delayed observation, no action is taken. Otherwise, the policy function is called with the observed price (delayed by the latency), and the corresponding buy or sell action is executed.
                """
                if latency_steps and self.i - latency_steps < 0:
                    return
                price_obs = self.data.Close[-latency_steps - 1] if latency_steps else self.data.Close[-1]
                action = policy_fn(price_obs)
                if action == "buy":
                    self.buy()
                elif action == "sell":
                    self.sell()

        bt = Backtest(df, PolicyStrategy, commission=slippage)
        stats = bt.run()
        return pd.Series(stats._stats)

    def apply_slippage(self, price: float) -> float:
        """
        Return the input price adjusted by the configured slippage percentage.
        
        Parameters:
            price (float): The original price to which slippage should be applied.
        
        Returns:
            float: The price after applying the slippage percentage.
        """
        return price * (1 + self.slippage_pct)

    def apply_latency(self, delay: float) -> float:
        """
        Return the sum of the input delay and the configured latency in seconds.
        
        Parameters:
            delay (float): The initial delay value.
        
        Returns:
            float: The delay increased by the configured latency.
        """
        return delay + self.latency_seconds
