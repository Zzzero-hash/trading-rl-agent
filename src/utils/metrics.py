"""
Trading Metrics Utilities

This module provides metrics for evaluating trading strategy performance.
Includes risk-adjusted returns, drawdown analysis, and portfolio metrics.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)


def calculate_max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Portfolio value over time

    Returns:
        Maximum drawdown as percentage
    """
    if len(equity_curve) == 0:
        return 0.0

    cumulative_returns = np.array(equity_curve)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return float(abs(np.min(drawdown)))


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray], target_return: float = 0.0
) -> float:
    """
    Calculate Sortino ratio (downside deviation).

    Args:
        returns: Series of periodic returns
        target_return: Target return rate

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - target_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float("inf") if np.mean(excess_returns) > 0 else 0.0

    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return 0.0

    return np.mean(excess_returns) / downside_deviation * np.sqrt(252)


def calculate_profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Series of periodic returns

    Returns:
        Profit factor
    """
    if len(returns) == 0:
        return 0.0

    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    gross_profit = np.sum(positive_returns) if len(positive_returns) > 0 else 0
    gross_loss = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Series of periodic returns

    Returns:
        Win rate as percentage (0-1)
    """
    if len(returns) == 0:
        return 0.0

    winning_trades = np.sum(returns > 0)
    total_trades = len(returns)

    return float(winning_trades / total_trades)


def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Series of periodic returns

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    annual_return = np.mean(returns) * 252
    equity_curve = np.cumprod(1 + returns)
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float("inf") if annual_return > 0 else 0.0

    return float(annual_return / max_dd)


def calculate_risk_metrics(
    returns: Union[pd.Series, np.ndarray],
    equity_curve: Optional[Union[pd.Series, np.ndarray]] = None,
) -> dict:
    """
    Calculate comprehensive risk metrics for trading strategy.

    Args:
        returns: Series of periodic returns
        equity_curve: Portfolio equity curve (optional, will calculate if not provided)

    Returns:
        Dictionary of risk metrics
    """
    if equity_curve is None:
        equity_curve = np.cumprod(1 + returns)

    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "sortino_ratio": calculate_sortino_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "profit_factor": calculate_profit_factor(returns),
        "win_rate": calculate_win_rate(returns),
        "calmar_ratio": calculate_calmar_ratio(returns),
        "total_return": (equity_curve[-1] - 1) if len(equity_curve) > 0 else 0.0,
        "volatility": np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0,
        "num_trades": len(returns),
    }

    return metrics


# Legacy compatibility
def compute_sharpe_ratio(returns: Union[pd.Series, np.ndarray]) -> float:
    """Legacy function name - use calculate_sharpe_ratio instead."""
    return calculate_sharpe_ratio(returns)


def compute_max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> float:
    """Legacy function name - use calculate_max_drawdown instead."""
    return calculate_max_drawdown(equity_curve)
