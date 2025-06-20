"""Utilities for evaluating trained policies.

This module currently exposes a simple :class:`Backtester` stub that
illustrates the interface for an event-driven backtesting engine.
"""

from __future__ import annotations


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

    def run_backtest(self, prices, policy):
        """Execute a backtest over ``prices`` using ``policy``.

        Parameters
        ----------
        prices : Sequence
            Iterable price series such as a list or ``pandas.Series``.
        policy : Callable
            Function that maps a market observation to an action.

        Returns
        -------
        Any
            Placeholder backtest results.  Currently returns ``None``.
        """

        # TODO: implement event-driven backtesting
        return None

    def apply_slippage(self, price: float) -> float:
        """Apply a slippage model to ``price``.

        Parameters
        ----------
        price : float
            Executed price prior to slippage adjustment.

        Returns
        -------
        float
            Adjusted trade price.  Currently returned unchanged.
        """

        # TODO: implement slippage logic
        return price

    def apply_latency(self, delay: float) -> float:
        """Apply a latency model to ``delay``.

        Parameters
        ----------
        delay : float
            Seconds between signal generation and execution.

        Returns
        -------
        float
            Adjusted delay after latency effects.  Currently returned unchanged.
        """

        # TODO: implement latency logic
        return delay
