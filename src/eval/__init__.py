"""Utilities for evaluating trained policies.

This module currently exposes a simple :class:`Backtester` stub that
illustrates the interface for an event-driven backtesting engine.
"""

from __future__ import annotations

from typing import Any, Callable, List, Sequence


class Backtester:
    """Skeleton backtester for RL trading strategies.

    The implementation purposefully remains minimal to show the intended
    API surface.  Real logic can be added incrementally without breaking
    existing imports.

    Example
    -------
    >>> from src.eval import Backtester
    >>> bt = Backtester()
    >>> bt.run_backtest(prices=[1, 2, 3], policy=lambda price: "hold")
    >>> bt.apply_slippage(100.0)
    100.0
    >>> bt.apply_latency(0.5)
    0.5
    """

    def __init__(self, slippage_pct: float = 0.0, latency_seconds: float = 0.0) -> None:
        """Initialize backtester with slippage and latency models."""

        self.slippage_pct = slippage_pct
        self.latency_seconds = latency_seconds

    def run_backtest(
        self, prices: Sequence[float], policy: Callable[[float], Any]
    ) -> List[dict]:
        """Execute a backtest over ``prices`` using ``policy``.

        Parameters
        ----------
        prices : Sequence
            Iterable price series such as a list or ``pandas.Series``.
        policy : Callable
            Function that maps a market observation to an action.

        Returns
        -------
        list of dict
            List of executed actions with adjusted price and delay.
        """

        results = []
        for price in prices:
            action = policy(price)
            executed_price = self.apply_slippage(price)
            delay = self.apply_latency(0.0)
            results.append({"price": executed_price, "action": action, "delay": delay})

        return results

    def apply_slippage(self, price: float) -> float:
        """Apply a slippage model to ``price``.

        Parameters
        ----------
        price : float
            Executed price prior to slippage adjustment.

        Returns
        -------
        float
            Adjusted trade price after slippage.
        """

        return price * (1 + self.slippage_pct)

    def apply_latency(self, delay: float) -> float:
        """Apply a latency model to ``delay``.

        Parameters
        ----------
        delay : float
            Seconds between signal generation and execution.

        Returns
        -------
        float
            Adjusted delay after latency effects.
        """

        return delay + self.latency_seconds
