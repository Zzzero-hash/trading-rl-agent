"""
Trading Metrics Utilities

This module provides metrics for evaluating trading strategy performance.
Includes risk-adjusted returns, drawdown analysis, and portfolio metrics.
"""

"""Convenience wrappers around the ``empyrical`` statistics library."""

import numpy as np
import empyrical as _empyrical

TRADING_DAYS_PER_YEAR = 252


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


def calculate_max_drawdown(returns) -> float:
    """Calculate maximum drawdown from a series of returns."""
    return float(_empyrical.max_drawdown(np.asarray(returns)))


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
    """Calculate profit factor (gross profit / gross loss)."""
    returns = np.asarray(returns)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def calculate_win_rate(returns) -> float:
    """Calculate win rate (percentage of positive returns)."""
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def calculate_calmar_ratio(returns) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Series of periodic returns

    Returns:
        Calmar ratio
    """
    return float(_empyrical.calmar_ratio(returns))


def calculate_var(returns, confidence: float = 0.95) -> float:
    """Value at Risk (VaR) at the given confidence level."""
    cutoff = 1 - confidence
    return float(_empyrical.value_at_risk(np.asarray(returns), cutoff=cutoff))


def calculate_expected_shortfall(returns, confidence: float = 0.95) -> float:
    """Expected shortfall (Conditional VaR)."""
    cutoff = 1 - confidence
    return float(
        _empyrical.conditional_value_at_risk(np.asarray(returns), cutoff=cutoff)
    )


def calculate_information_ratio(returns, benchmark_returns) -> float:
    """Information ratio versus a benchmark."""
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    excess = returns - benchmark_returns
    te = np.std(excess)
    if te == 0:
        return 0.0
    return float(np.mean(excess) / te * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_tracking_error(returns, benchmark_returns) -> float:
    """Annualized tracking error."""
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)
    excess = returns - benchmark_returns
    return float(np.std(excess) * np.sqrt(TRADING_DAYS_PER_YEAR))


def calculate_beta(returns, benchmark_returns) -> float:
    """Beta of returns relative to benchmark."""
    return float(_empyrical.beta(np.asarray(returns), np.asarray(benchmark_returns)))


def calculate_average_win_loss_ratio(returns) -> float:
    """Average win/loss ratio."""
    returns = np.asarray(returns)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(losses) == 0:
        return float("inf") if len(wins) > 0 else 0.0
    return float(wins.mean() / -losses.mean())


def calculate_comprehensive_metrics(
    returns,
    benchmark_returns=None,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
):
    """Return a dictionary with common trading performance metrics."""
    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate),
        "calmar_ratio": calculate_calmar_ratio(returns),
        "max_drawdown": calculate_max_drawdown(returns),
        "var_95": calculate_var(returns, confidence),
        "expected_shortfall": calculate_expected_shortfall(returns, confidence),
        "profit_factor": calculate_profit_factor(returns),
        "win_rate": calculate_win_rate(returns),
        "average_win_loss_ratio": calculate_average_win_loss_ratio(returns),
    }

    if benchmark_returns is not None:
        metrics.update(
            {
                "information_ratio": calculate_information_ratio(
                    returns, benchmark_returns
                ),
                "tracking_error": calculate_tracking_error(
                    returns, benchmark_returns
                ),
                "beta": calculate_beta(returns, benchmark_returns),
            }
        )

    return metrics


def calculate_risk_metrics(returns, equity_curve=None):
    """Comprehensive risk metrics using empyrical"""
    if equity_curve is None:
        equity_curve = np.cumprod(1 + np.asarray(returns))
    return {
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "sortino_ratio": calculate_sortino_ratio(returns),
        "max_drawdown": calculate_max_drawdown(returns),
        "profit_factor": calculate_profit_factor(returns),
        "win_rate": calculate_win_rate(returns),
        "calmar_ratio": calculate_calmar_ratio(returns),
        "total_return": float(equity_curve[-1] - 1) if len(equity_curve) > 0 else 0.0,
        "volatility": (
            float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR))
            if len(returns) > 0
            else 0.0
        ),
        "num_trades": len(returns),
    }
