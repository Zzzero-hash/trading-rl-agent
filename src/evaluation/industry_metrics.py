"""
Industry-Standard Trading Performance Metrics

Comprehensive evaluation metrics used by professional trading firms
and institutional investors for performance assessment.
"""

from typing import Optional, Union
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class IndustryStandardEvaluator:
    """
    Industry-standard trading performance evaluator.

    Implements metrics commonly used by hedge funds, asset managers,
    and institutional trading firms for performance attribution and
    risk-adjusted return analysis.
    """

    def __init__(self, trading_days_per_year: int = 252):
        """
        Initialize evaluator.

        Args:
            trading_days_per_year: Number of trading days per year for annualization
        """
        self.trading_days_per_year = trading_days_per_year

    def calculate_comprehensive_metrics(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
        risk_free_rate: float = 0.02,
    ) -> dict:
        """
        Calculate comprehensive industry-standard metrics.

        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns for comparison
            risk_free_rate: Risk-free rate (annualized)

        Returns:
            Dictionary of performance metrics
        """
        returns = self._ensure_series(returns)

        if benchmark_returns is not None:
            benchmark_returns = self._ensure_series(benchmark_returns)

        metrics = {}

        # Risk-adjusted returns
        metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics["sortino_ratio"] = self.calculate_sortino_ratio(returns, risk_free_rate)
        metrics["calmar_ratio"] = self.calculate_calmar_ratio(returns)

        # Risk metrics
        metrics["max_drawdown"] = self.calculate_max_drawdown(returns)
        metrics["var_95"] = self.calculate_var(returns, confidence=0.95)
        metrics["expected_shortfall"] = self.calculate_expected_shortfall(
            returns, confidence=0.95
        )
        metrics["volatility"] = self.calculate_volatility(returns)

        # Return metrics
        metrics["total_return"] = self.calculate_total_return(returns)
        metrics["annualized_return"] = self.calculate_annualized_return(returns)
        metrics["skewness"] = self.calculate_skewness(returns)
        metrics["kurtosis"] = self.calculate_kurtosis(returns)

        # Trading metrics
        metrics["profit_factor"] = self.calculate_profit_factor(returns)
        metrics["win_rate"] = self.calculate_win_rate(returns)
        metrics["average_win_loss_ratio"] = self.calculate_avg_win_loss_ratio(returns)

        # Benchmark comparison (if provided)
        if benchmark_returns is not None:
            metrics["information_ratio"] = self.calculate_information_ratio(
                returns, benchmark_returns
            )
            metrics["tracking_error"] = self.calculate_tracking_error(
                returns, benchmark_returns
            )
            metrics["beta"] = self.calculate_beta(returns, benchmark_returns)
            metrics["alpha"] = self.calculate_alpha(
                returns, benchmark_returns, risk_free_rate
            )
            metrics["treynor_ratio"] = self.calculate_treynor_ratio(
                returns, benchmark_returns, risk_free_rate
            )

        return metrics

    def calculate_sharpe_ratio(
        self, returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio (risk-adjusted return)."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / self.trading_days_per_year

        if np.std(excess_returns) == 0:
            return 0.0

        return (
            np.mean(excess_returns)
            / np.std(excess_returns)
            * np.sqrt(self.trading_days_per_year)
        )

    def calculate_sortino_ratio(
        self, returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return)."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / self.trading_days_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return np.inf if np.mean(excess_returns) > 0 else 0.0

        downside_deviation = np.std(downside_returns)
        return (
            np.mean(excess_returns)
            / downside_deviation
            * np.sqrt(self.trading_days_per_year)
        )

    def calculate_calmar_ratio(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate Calmar ratio (return/max drawdown)."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        annualized_return = self.calculate_annualized_return(returns)
        max_drawdown = abs(self.calculate_max_drawdown(returns))

        if max_drawdown == 0:
            return np.inf if annualized_return > 0 else 0.0

        return annualized_return / max_drawdown

    def calculate_max_drawdown(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate maximum drawdown."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        cumulative_returns = (1 + returns).cumprod()
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max

        return np.min(drawdowns)

    def calculate_var(
        self, returns: Union[pd.Series, np.ndarray], confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_expected_shortfall(
        self, returns: Union[pd.Series, np.ndarray], confidence: float = 0.95
    ) -> float:
        """Calculate Expected Shortfall (Conditional VaR)."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        var = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= var]

        if len(tail_returns) == 0:
            return var

        return np.mean(tail_returns)

    def calculate_volatility(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate annualized volatility."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        return np.std(returns) * np.sqrt(self.trading_days_per_year)

    def calculate_total_return(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate total return."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        return (1 + returns).prod() - 1

    def calculate_annualized_return(
        self, returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """Calculate annualized return."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        total_return = self.calculate_total_return(returns)
        periods = len(returns) / self.trading_days_per_year

        if periods <= 0:
            return 0.0

        return (1 + total_return) ** (1 / periods) - 1

    def calculate_skewness(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate skewness of returns."""
        returns = self._ensure_array(returns)

        if len(returns) < 3:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness

    def calculate_kurtosis(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate kurtosis of returns."""
        returns = self._ensure_array(returns)

        if len(returns) < 4:
            return 0.0

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0.0

        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis

    def calculate_profit_factor(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        profits = returns[returns > 0]
        losses = returns[returns < 0]

        gross_profit = np.sum(profits) if len(profits) > 0 else 0
        gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0

        return gross_profit / gross_loss

    def calculate_win_rate(self, returns: Union[pd.Series, np.ndarray]) -> float:
        """Calculate win rate (percentage of profitable trades)."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        winning_trades = np.sum(returns > 0)
        total_trades = len(returns)

        return winning_trades / total_trades

    def calculate_avg_win_loss_ratio(
        self, returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """Calculate average win/loss ratio."""
        returns = self._ensure_array(returns)

        if len(returns) == 0:
            return 0.0

        profits = returns[returns > 0]
        losses = returns[returns < 0]

        avg_profit = np.mean(profits) if len(profits) > 0 else 0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0

        if avg_loss == 0:
            return np.inf if avg_profit > 0 else 0.0

        return avg_profit / avg_loss

    def calculate_information_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
    ) -> float:
        """Calculate information ratio."""
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align arrays to same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

        excess_returns = returns - benchmark_returns
        tracking_error = np.std(excess_returns)

        if tracking_error == 0:
            return 0.0

        return (
            np.mean(excess_returns)
            / tracking_error
            * np.sqrt(self.trading_days_per_year)
        )

    def calculate_tracking_error(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
    ) -> float:
        """Calculate tracking error (annualized)."""
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align arrays to same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

        excess_returns = returns - benchmark_returns
        return np.std(excess_returns) * np.sqrt(self.trading_days_per_year)

    def calculate_beta(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
    ) -> float:
        """Calculate beta (sensitivity to benchmark)."""
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align arrays to same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance == 0:
            return 0.0

        covariance = np.cov(returns, benchmark_returns)[0, 1]
        return covariance / benchmark_variance

    def calculate_alpha(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate alpha (excess return over CAPM prediction)."""
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align arrays to same length
        min_length = min(len(returns), len(benchmark_returns))
        returns = returns[:min_length]
        benchmark_returns = benchmark_returns[:min_length]

        beta = self.calculate_beta(returns, benchmark_returns)

        portfolio_return = np.mean(returns) * self.trading_days_per_year
        benchmark_return = np.mean(benchmark_returns) * self.trading_days_per_year

        expected_return = risk_free_rate + beta * (benchmark_return - risk_free_rate)
        alpha = portfolio_return - expected_return

        return alpha

    def calculate_treynor_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        benchmark_returns: Union[pd.Series, np.ndarray],
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Treynor ratio (excess return per unit of systematic risk)."""
        returns = self._ensure_array(returns)
        benchmark_returns = self._ensure_array(benchmark_returns)

        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        beta = self.calculate_beta(returns, benchmark_returns)

        if beta == 0:
            return 0.0

        annualized_return = self.calculate_annualized_return(returns)
        excess_return = annualized_return - risk_free_rate

        return excess_return / beta

    def _ensure_series(self, data: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Ensure data is a pandas Series."""
        if isinstance(data, np.ndarray):
            return pd.Series(data)
        return data

    def _ensure_array(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Ensure data is a numpy array."""
        if isinstance(data, pd.Series):
            return data.values
        return np.asarray(data)


def calculate_performance_summary(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.02,
) -> dict:
    """
    Calculate comprehensive performance summary using industry standards.

    Args:
        returns: Strategy returns
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Risk-free rate (annualized)

    Returns:
        Dictionary with performance metrics and interpretation
    """
    evaluator = IndustryStandardEvaluator()
    metrics = evaluator.calculate_comprehensive_metrics(
        returns, benchmark_returns, risk_free_rate
    )

    # Add performance interpretation
    summary = {"metrics": metrics, "interpretation": _interpret_metrics(metrics)}

    return summary


def _interpret_metrics(metrics: dict) -> dict:
    """Provide interpretation of performance metrics."""
    interpretation = {}

    # Sharpe ratio interpretation
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe > 2.0:
        interpretation["sharpe_quality"] = "Excellent"
    elif sharpe > 1.0:
        interpretation["sharpe_quality"] = "Good"
    elif sharpe > 0.5:
        interpretation["sharpe_quality"] = "Acceptable"
    else:
        interpretation["sharpe_quality"] = "Poor"

    # Maximum drawdown interpretation
    max_dd = abs(metrics.get("max_drawdown", 0))
    if max_dd < 0.05:
        interpretation["risk_level"] = "Low"
    elif max_dd < 0.15:
        interpretation["risk_level"] = "Moderate"
    elif max_dd < 0.25:
        interpretation["risk_level"] = "High"
    else:
        interpretation["risk_level"] = "Very High"

    # Win rate interpretation
    win_rate = metrics.get("win_rate", 0)
    if win_rate > 0.6:
        interpretation["consistency"] = "Highly Consistent"
    elif win_rate > 0.5:
        interpretation["consistency"] = "Consistent"
    elif win_rate > 0.4:
        interpretation["consistency"] = "Moderate"
    else:
        interpretation["consistency"] = "Inconsistent"

    return interpretation
