"""
Mock empyrical module for compatibility with Python 3.13.

This module provides basic implementations of empyrical functions
that are used in the trading RL agent.
"""

import numpy as np


def sharpe_ratio(returns: np.ndarray | list, risk_free: float = 0.0) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free: Risk-free rate

    Returns:
        Sharpe ratio
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free
    if np.std(excess_returns) == 0:
        return 0.0

    return float(np.mean(excess_returns) / np.std(excess_returns))


def max_drawdown(returns: np.ndarray | list) -> float:
    """
    Calculate maximum drawdown.

    Args:
        returns: Array of returns

    Returns:
        Maximum drawdown as a negative value
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))


def sortino_ratio(returns: np.ndarray | list, required_return: float = 0.0) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Array of returns
        required_return: Required return rate

    Returns:
        Sortino ratio
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - required_return
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    return float(np.mean(excess_returns) / downside_std)


def calmar_ratio(returns: np.ndarray | list) -> float:
    """
    Calculate Calmar ratio.

    Args:
        returns: Array of returns

    Returns:
        Calmar ratio
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    max_dd = max_drawdown(returns)
    if max_dd == 0:
        return 0.0

    annual_return = np.mean(returns) * 252  # Assuming daily returns
    return float(annual_return / abs(max_dd))


def value_at_risk(returns: np.ndarray | list, cutoff: float = 0.05) -> float:
    """
    Calculate Value at Risk.

    Args:
        returns: Array of returns
        cutoff: Confidence level

    Returns:
        Value at Risk
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    return float(np.percentile(returns, cutoff * 100))


def conditional_value_at_risk(returns: np.ndarray | list, cutoff: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (Expected Shortfall).

    Args:
        returns: Array of returns
        cutoff: Confidence level

    Returns:
        Conditional Value at Risk
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    var = value_at_risk(returns, cutoff)
    return float(np.mean(returns[returns <= var]))


def beta(returns: np.ndarray | list, benchmark_returns: np.ndarray | list) -> float:
    """
    Calculate beta relative to benchmark.

    Args:
        returns: Array of returns
        benchmark_returns: Array of benchmark returns

    Returns:
        Beta value
    """
    returns = np.asarray(returns)
    benchmark_returns = np.asarray(benchmark_returns)

    if len(returns) == 0 or len(benchmark_returns) == 0:
        return 0.0

    # Ensure same length
    min_len = min(len(returns), len(benchmark_returns))
    returns = returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]

    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)

    if benchmark_variance == 0:
        return 0.0

    return float(covariance / benchmark_variance)
