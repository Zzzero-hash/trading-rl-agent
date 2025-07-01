"""
Trading Metrics Utilities

This module provides metrics for evaluating trading strategy performance.
Includes risk-adjusted returns, drawdown analysis, and portfolio metrics.
"""

import empyrical as _empyrical
import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Sharpe ratio
    """
    return float(_empyrical.sharpe_ratio(returns, risk_free=risk_free_rate))


def calculate_max_drawdown(equity_curve) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Portfolio value over time

    Returns:
        Maximum drawdown as percentage
    """
    return float(_empyrical.max_drawdown(equity_curve))


def calculate_sortino_ratio(returns, target_return: float = 0.0) -> float:
    """
    Calculate Sortino ratio (downside deviation).

    Args:
        returns: Series of periodic returns
        target_return: Target return rate

    Returns:
        Sortino ratio
    """
    return float(_empyrical.sortino_ratio(returns, required_return=target_return))


def calculate_profit_factor(returns) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Series of periodic returns

    Returns:
        Profit factor
    """
    return float(_empyrical.profit_factor(returns))


def calculate_win_rate(returns) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Series of periodic returns

    Returns:
        Win rate as percentage (0-1)
    """
    return float(_empyrical.win_rate(returns))


def calculate_calmar_ratio(returns) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Series of periodic returns

    Returns:
        Calmar ratio
    """
    return float(_empyrical.calmar_ratio(returns))


def calculate_risk_metrics(returns, equity_curve=None):
    """Comprehensive risk metrics using empyrical"""
    if equity_curve is None:
        equity_curve = np.cumprod(1 + np.array(returns))
    return {
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "sortino_ratio": calculate_sortino_ratio(returns),
        "max_drawdown": calculate_max_drawdown(equity_curve),
        "profit_factor": calculate_profit_factor(returns),
        "win_rate": calculate_win_rate(returns),
        "calmar_ratio": calculate_calmar_ratio(returns),
        "total_return": float(equity_curve[-1] - 1) if len(equity_curve) > 0 else 0.0,
        "volatility": (
            float(np.std(returns) * np.sqrt(252)) if len(returns) > 0 else 0.0
        ),
        "num_trades": len(returns),
    }
