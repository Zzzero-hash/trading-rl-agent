"""
Comprehensive risk management system.

Handles portfolio risk monitoring, position sizing, and risk controls
for the trading system.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import numpy as np
import pandas as pd

try:
    # Future scipy support for advanced statistics
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    # Portfolio level limits
    max_portfolio_var: float = 0.02  # 2% VaR limit
    max_drawdown: float = 0.10  # 10% max drawdown
    max_leverage: float = 1.0  # No leverage
    max_correlation: float = 0.8  # Max correlation between positions

    # Position level limits
    max_position_size: float = 0.1  # 10% max position size
    max_sector_exposure: float = 0.3  # 30% max sector exposure

    # Trading limits
    max_daily_trades: int = 100
    max_daily_volume: float = 1000000  # $1M daily volume limit

    # Stop loss and take profit
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.15  # 15% take profit


@dataclass
class RiskMetrics:
    """Risk metrics container."""

    portfolio_var: float
    portfolio_cvar: float
    max_drawdown: float
    current_drawdown: float
    leverage: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    timestamp: datetime


class RiskManager:
    """
    Comprehensive risk management system.

    Features:
    - Real-time risk monitoring
    - VaR and CVaR calculations
    - Position sizing optimization
    - Correlation and concentration analysis
    - Automated risk controls
    """

    def __init__(self, risk_limits: RiskLimits | None = None):
        """
        Initialize risk manager.

        Args:
            risk_limits: Risk limits configuration
        """
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = get_logger(self.__class__.__name__)

        # Risk state
        self.current_metrics: RiskMetrics | None = None
        self.risk_alerts: list[dict] = []
        self.historical_metrics: list[RiskMetrics] = []

        # Market data for calculations
        self._returns_data: dict[str, pd.Series] = {}
        self._benchmark_returns: pd.Series | None = None

        if not SCIPY_AVAILABLE:
            self.logger.warning(
                "SciPy not available, some risk calculations may be limited",
            )

    def update_returns_data(
        self,
        returns_data: dict[str, pd.Series],
        benchmark_returns: pd.Series | None = None,
    ) -> None:
        """
        Update returns data for risk calculations.

        Args:
            returns_data: Dictionary of asset returns series
            benchmark_returns: Benchmark returns for beta calculation
        """
        self._returns_data = returns_data.copy()
        self._benchmark_returns = benchmark_returns
        self.logger.debug(f"Updated returns data for {len(returns_data)} assets")

    def calculate_portfolio_var(
        self,
        weights: dict[str, float],
        confidence_level: float = 0.05,
        time_horizon: int = 1,
    ) -> float:
        """
        Calculate portfolio Value at Risk (VaR).

        Args:
            weights: Portfolio weights
            confidence_level: VaR confidence level (default 5%)
            time_horizon: Time horizon in days

        Returns:
            Portfolio VaR value
        """
        try:
            # Filter weights and returns for available assets
            available_assets = set(weights.keys()) & set(self._returns_data.keys())
            if not available_assets:
                self.logger.warning("No overlapping assets for VaR calculation")
                return 0.0

            # Prepare returns matrix
            returns_matrix = pd.DataFrame()
            filtered_weights = {}

            for asset in available_assets:
                if weights[asset] != 0:  # Only include non-zero weights
                    returns_matrix[asset] = self._returns_data[asset]
                    filtered_weights[asset] = weights[asset]

            if returns_matrix.empty:
                return 0.0

            # Normalize weights
            total_weight = sum(filtered_weights.values())
            if total_weight == 0:
                return 0.0

            normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}

            # Calculate portfolio returns
            weight_series = pd.Series(normalized_weights)
            portfolio_returns = (returns_matrix * weight_series).sum(axis=1)
            portfolio_returns = portfolio_returns.dropna()

            if len(portfolio_returns) < 30:  # Need sufficient data
                self.logger.warning("Insufficient data for reliable VaR calculation")
                return float(portfolio_returns.std() * 2.33)  # Approximate using normal distribution

            # Calculate VaR using historical simulation
            var_value = np.percentile(portfolio_returns, confidence_level * 100)

            # Scale for time horizon
            var_scaled = var_value * np.sqrt(time_horizon)

            return float(abs(var_scaled))

        except Exception as e:
            self.logger.exception(f"VaR calculation failed: {e}")
            return 0.0

    def calculate_portfolio_cvar(
        self,
        weights: dict[str, float],
        confidence_level: float = 0.05,
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).

        Args:
            weights: Portfolio weights
            confidence_level: CVaR confidence level

        Returns:
            Portfolio CVaR value
        """
        try:
            var_value = self.calculate_portfolio_var(weights, confidence_level)

            # Get portfolio returns
            available_assets = set(weights.keys()) & set(self._returns_data.keys())
            if not available_assets:
                return var_value * 1.3  # Conservative estimate

            returns_matrix = pd.DataFrame()
            filtered_weights = {}

            for asset in available_assets:
                if weights[asset] != 0:
                    returns_matrix[asset] = self._returns_data[asset]
                    filtered_weights[asset] = weights[asset]

            if returns_matrix.empty:
                return var_value * 1.3

            # Normalize weights
            total_weight = sum(filtered_weights.values())
            normalized_weights = {k: v / total_weight for k, v in filtered_weights.items()}
            weight_series = pd.Series(normalized_weights)

            portfolio_returns = (returns_matrix * weight_series).sum(axis=1).dropna()

            # Calculate CVaR as expected value of returns below VaR
            var_threshold = np.percentile(portfolio_returns, confidence_level * 100)
            tail_returns = portfolio_returns[portfolio_returns <= var_threshold]

            if len(tail_returns) > 0:
                cvar_value = abs(tail_returns.mean())
            else:
                cvar_value = var_value * 1.3  # Conservative estimate

            return float(cvar_value)

        except Exception as e:
            self.logger.exception(f"CVaR calculation failed: {e}")
            return self.calculate_portfolio_var(weights, confidence_level) * 1.3

    def calculate_portfolio_drawdown(self, weights: dict[str, float]) -> float:
        """Calculate maximum drawdown for a weighted portfolio."""
        try:
            available_assets = set(weights.keys()) & set(self._returns_data.keys())
            if not available_assets:
                return 0.0

            returns_df = pd.DataFrame()
            filtered_weights = {}
            for asset in available_assets:
                if weights[asset] != 0:
                    returns_df[asset] = self._returns_data[asset]
                    filtered_weights[asset] = weights[asset]

            if returns_df.empty:
                return 0.0

            total_weight = sum(filtered_weights.values())
            if total_weight == 0:
                return 0.0

            weight_series = pd.Series(
                {k: v / total_weight for k, v in filtered_weights.items()},
            )
            portfolio_returns = (returns_df * weight_series).sum(axis=1).dropna()

            if portfolio_returns.empty:
                return 0.0

            cumulative_returns = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            return float(abs(drawdowns.min()))
        except Exception as e:
            self.logger.exception(f"Drawdown calculation failed: {e}")
            return 0.0

    def calculate_sharpe_ratio(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Sharpe ratio."""
        try:
            if len(portfolio_returns) < 2:
                return 0.0

            excess_returns = portfolio_returns.mean() - risk_free_rate / 252  # Daily risk-free rate
            volatility = portfolio_returns.std()

            if volatility == 0:
                return 0.0

            return float((excess_returns / volatility) * np.sqrt(252))  # Annualized

        except Exception as e:
            self.logger.exception(f"Sharpe ratio calculation failed: {e}")
            return 0.0

    def calculate_sortino_ratio(
        self,
        portfolio_returns: pd.Series,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        try:
            if len(portfolio_returns) < 2:
                return 0.0

            excess_returns = portfolio_returns.mean() - risk_free_rate / 252
            downside_returns = portfolio_returns[portfolio_returns < 0]

            if len(downside_returns) == 0:
                return float("inf")

            downside_deviation = downside_returns.std()

            if downside_deviation == 0:
                return float("inf")

            return float((excess_returns / downside_deviation) * np.sqrt(252))

        except Exception as e:
            self.logger.exception(f"Sortino ratio calculation failed: {e}")
            return 0.0

    def calculate_beta(self, portfolio_returns: pd.Series) -> float:
        """Calculate portfolio beta relative to benchmark."""
        try:
            if self._benchmark_returns is None or len(portfolio_returns) < 2:
                return 1.0  # Default to market beta

            # Align returns
            aligned_data = pd.DataFrame(
                {"portfolio": portfolio_returns, "benchmark": self._benchmark_returns},
            ).dropna()

            if len(aligned_data) < 10:  # Need sufficient data
                return 1.0

            # Calculate beta using covariance
            covariance = aligned_data["portfolio"].cov(aligned_data["benchmark"])
            benchmark_variance = aligned_data["benchmark"].var()

            if benchmark_variance == 0:
                return 1.0

            return float(covariance / benchmark_variance)

        except Exception as e:
            self.logger.exception(f"Beta calculation failed: {e}")
            return 1.0

    def calculate_correlation_risk(self, weights: dict[str, float]) -> float:
        """Calculate portfolio correlation risk."""
        try:
            # Get returns for weighted assets
            weighted_assets = [
                asset for asset, weight in weights.items() if weight != 0 and asset in self._returns_data
            ]

            if len(weighted_assets) < 2:
                return 0.0  # No correlation risk with single asset

            # Build correlation matrix
            returns_df = pd.DataFrame()
            for asset in weighted_assets:
                returns_df[asset] = self._returns_data[asset]

            correlation_matrix = returns_df.corr()

            # Calculate weighted average correlation
            total_correlation = 0.0
            total_weight_pairs = 0.0

            for i, asset1 in enumerate(weighted_assets):
                for j, asset2 in enumerate(weighted_assets):
                    if i != j:
                        weight_product = weights[asset1] * weights[asset2]
                        correlation = correlation_matrix.loc[asset1, asset2]
                        total_correlation += weight_product * abs(correlation)
                        total_weight_pairs += weight_product

            if total_weight_pairs == 0:
                return 0.0

            return total_correlation / total_weight_pairs

        except Exception as e:
            self.logger.exception(f"Correlation risk calculation failed: {e}")
            return 0.0

    def calculate_concentration_risk(self, weights: dict[str, float]) -> float:
        """Calculate portfolio concentration risk using Herfindahl index."""
        try:
            # Filter non-zero weights
            non_zero_weights = {k: v for k, v in weights.items() if v != 0}

            if not non_zero_weights:
                return 0.0

            # Normalize weights
            total_weight = sum(non_zero_weights.values())
            normalized_weights = {k: v / total_weight for k, v in non_zero_weights.items()}

            # Calculate Herfindahl index
            herfindahl = sum(w**2 for w in normalized_weights.values())

            # Convert to concentration risk (0 = fully diversified, 1 = fully concentrated)
            n_assets = len(normalized_weights)

            if n_assets == 1:
                return 1.0

            return herfindahl

        except Exception as e:
            self.logger.exception(f"Concentration risk calculation failed: {e}")
            return 0.0

    def check_risk_limits(
        self,
        portfolio_weights: dict[str, float],
        portfolio_value: float,
    ) -> list[dict[str, Any]]:
        """
        Check portfolio against risk limits.

        Args:
            portfolio_weights: Current portfolio weights
            portfolio_value: Current portfolio value

        Returns:
            List of risk violations
        """
        violations = []

        try:
            # Calculate current metrics
            current_var = self.calculate_portfolio_var(portfolio_weights)
            # current_cvar = self.calculate_portfolio_cvar(portfolio_weights)  # Not used currently
            correlation_risk = self.calculate_correlation_risk(portfolio_weights)
            # concentration_risk = self.calculate_concentration_risk(portfolio_weights)  # Not used currently
            drawdown = self.calculate_portfolio_drawdown(portfolio_weights)

            # Check VaR limit
            if current_var > self.risk_limits.max_portfolio_var:
                violations.append(
                    {
                        "type": "var_limit",
                        "current_value": current_var,
                        "limit": self.risk_limits.max_portfolio_var,
                        "severity": "high",
                    },
                )

            # Check position size limits
            for asset, weight in portfolio_weights.items():
                if abs(weight) > self.risk_limits.max_position_size:
                    violations.append(
                        {
                            "type": "position_size",
                            "asset": asset,
                            "current_value": abs(weight),
                            "limit": self.risk_limits.max_position_size,
                            "severity": "medium",
                        },
                    )

            # Check correlation risk
            if correlation_risk > self.risk_limits.max_correlation:
                violations.append(
                    {
                        "type": "correlation_risk",
                        "current_value": correlation_risk,
                        "limit": self.risk_limits.max_correlation,
                        "severity": "medium",
                    },
                )

            # Check drawdown limit
            if drawdown > self.risk_limits.max_drawdown:
                violations.append(
                    {
                        "type": "drawdown_limit",
                        "current_value": drawdown,
                        "limit": self.risk_limits.max_drawdown,
                        "severity": "high",
                    },
                )

            return violations

        except Exception as e:
            self.logger.exception(f"Risk limit check failed: {e}")
            return []

    def calculate_kelly_position_size(
        self,
        expected_return: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_kelly_fraction: float = 0.25,
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion.

        Args:
            expected_return: Expected return of the strategy
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive value)
            max_kelly_fraction: Maximum Kelly fraction to use

        Returns:
            Optimal position size fraction
        """
        try:
            if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
                return 0.0

            # Kelly fraction formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate

            kelly_fraction = (b * p - q) / b

            # Apply safety constraints
            kelly_fraction = max(0, kelly_fraction)  # No negative positions
            return min(kelly_fraction, max_kelly_fraction)  # Cap at max

        except Exception as e:
            self.logger.exception(f"Kelly position size calculation failed: {e}")
            return 0.0

    def generate_risk_report(
        self,
        portfolio_weights: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, Any]:
        """Generate comprehensive risk report."""
        try:
            # Build portfolio returns if available
            portfolio_returns = None
            if self._returns_data:
                available_assets = set(portfolio_weights.keys()) & set(
                    self._returns_data.keys(),
                )
                if available_assets:
                    returns_df = pd.DataFrame()
                    for asset in available_assets:
                        if portfolio_weights[asset] != 0:
                            returns_df[asset] = self._returns_data[asset]

                    if not returns_df.empty:
                        weight_series = pd.Series(
                            {asset: portfolio_weights[asset] for asset in returns_df.columns},
                        )
                        portfolio_returns = (returns_df * weight_series).sum(axis=1)

            # Calculate metrics
            var = self.calculate_portfolio_var(portfolio_weights)
            cvar = self.calculate_portfolio_cvar(portfolio_weights)
            correlation_risk = self.calculate_correlation_risk(portfolio_weights)
            concentration_risk = self.calculate_concentration_risk(portfolio_weights)
            drawdown = self.calculate_portfolio_drawdown(portfolio_weights)

            report = {
                "timestamp": datetime.now(),
                "portfolio_value": portfolio_value,
                "risk_metrics": {
                    "var_5pct": var,
                    "cvar_5pct": cvar,
                    "correlation_risk": correlation_risk,
                    "concentration_risk": concentration_risk,
                    "max_drawdown": drawdown,
                },
                "risk_limits": {
                    "var_limit": self.risk_limits.max_portfolio_var,
                    "position_size_limit": self.risk_limits.max_position_size,
                    "correlation_limit": self.risk_limits.max_correlation,
                },
                "risk_violations": self.check_risk_limits(
                    portfolio_weights,
                    portfolio_value,
                ),
            }

            # Add performance metrics if portfolio returns available
            if portfolio_returns is not None and len(portfolio_returns) > 1:
                report["performance_metrics"] = {
                    "sharpe_ratio": self.calculate_sharpe_ratio(portfolio_returns),
                    "sortino_ratio": self.calculate_sortino_ratio(portfolio_returns),
                    "beta": self.calculate_beta(portfolio_returns),
                    "volatility": portfolio_returns.std() * np.sqrt(252),
                    "max_drawdown": self._calculate_max_drawdown(portfolio_returns),
                }

            return report

        except Exception as e:
            self.logger.exception(f"Risk report generation failed: {e}")
            return {"timestamp": datetime.now(), "error": str(e)}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        try:
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            return abs(cast(float, drawdowns.min()))
        except Exception:
            return 0.0
