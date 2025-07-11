"""
Trading Metrics Utilities

This module provides metrics for evaluating trading strategy performance.
Includes risk-adjusted returns, drawdown analysis, and portfolio metrics.

Convenience wrappers around the ``empyrical`` statistics library.
"""

import importlib

import empyrical as _empyrical
import numpy as np
import pandas as pd

# QuantStats has compatibility issues with newer IPython versions.
# We'll try to import it, but provide fallback implementations if it fails.
try:
    qs_stats = importlib.import_module("quantstats.stats")
    QUANTSTATS_AVAILABLE = True
except (ImportError, AttributeError):
    # Fallback when quantstats is not available or has compatibility issues
    QUANTSTATS_AVAILABLE = False
    qs_stats = None

TRADING_DAYS_PER_YEAR = 252


def _to_series(returns) -> pd.Series:
    """Convert input to a pandas Series with a DatetimeIndex."""
    series = pd.Series(returns)
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.date_range("1970-01-01", periods=len(series))
    return series


def _calculate_profit_factor_fallback(returns) -> float:
    """Fallback implementation of profit factor when quantstats is not available."""
    returns = np.asarray(returns)
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    if len(negative_returns) == 0:
        return float("inf") if len(positive_returns) > 0 else 0.0

    gross_profit = np.sum(positive_returns)
    gross_loss = abs(np.sum(negative_returns))

    return float(gross_profit / gross_loss) if gross_loss > 0 else 0.0


def _calculate_win_rate_fallback(returns) -> float:
    """Fallback implementation of win rate when quantstats is not available."""
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    winning_trades = np.sum(returns > 0)
    return float(winning_trades / len(returns))


def _calculate_win_loss_ratio_fallback(returns) -> float:
    """Fallback implementation of win/loss ratio when quantstats is not available."""
    returns = np.asarray(returns)
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]

    if len(positive_returns) == 0 or len(negative_returns) == 0:
        return 0.0

    avg_win = np.mean(positive_returns)
    avg_loss = abs(np.mean(negative_returns))

    return float(avg_win / avg_loss) if avg_loss > 0 else 0.0


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
    """Calculate profit factor using QuantStats or fallback implementation."""
    if QUANTSTATS_AVAILABLE and qs_stats is not None:
        series = _to_series(returns)
        return float(qs_stats.profit_factor(series))
    return _calculate_profit_factor_fallback(returns)


def calculate_win_rate(returns) -> float:
    """Calculate win rate using QuantStats or fallback implementation."""
    if QUANTSTATS_AVAILABLE and qs_stats is not None:
        series = _to_series(returns)
        return float(qs_stats.win_rate(series))
    return _calculate_win_rate_fallback(returns)


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
        _empyrical.conditional_value_at_risk(np.asarray(returns), cutoff=cutoff),
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
    """Average win/loss ratio using QuantStats or fallback implementation."""
    if QUANTSTATS_AVAILABLE and qs_stats is not None:
        series = _to_series(returns)
        # QuantStats exposes this metric as ``win_loss_ratio`` / ``payoff_ratio``
        return float(qs_stats.win_loss_ratio(series))
    return _calculate_win_loss_ratio_fallback(returns)


def calculate_comprehensive_metrics(
    returns,
    benchmark_returns=None,
    risk_free_rate: float = 0.0,
    confidence: float = 0.95,
):
    """Return a dictionary with common trading performance metrics."""
    series = _to_series(returns)
    metrics = {
        "sharpe_ratio": calculate_sharpe_ratio(series, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(series, risk_free_rate),
        "calmar_ratio": calculate_calmar_ratio(series),
        "max_drawdown": calculate_max_drawdown(series),
        "var_95": calculate_var(series, confidence),
        "expected_shortfall": calculate_expected_shortfall(series, confidence),
        "profit_factor": calculate_profit_factor(series),
        "win_rate": calculate_win_rate(series),
        "average_win_loss_ratio": calculate_average_win_loss_ratio(series),
    }

    if benchmark_returns is not None:
        metrics.update(
            {
                "information_ratio": calculate_information_ratio(
                    returns,
                    benchmark_returns,
                ),
                "tracking_error": calculate_tracking_error(returns, benchmark_returns),
                "beta": calculate_beta(returns, benchmark_returns),
            },
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
        "volatility": (float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns) > 0 else 0.0),
        "num_trades": len(returns),
    }
