"""
Riskfolio-specific risk management components.

This module contains risk management classes specifically designed for
integration with the Riskfolio portfolio optimization library.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskfolioConfig:
    """Configuration for Riskfolio risk management."""

    max_position: float
    min_position: float
    var_limit: float  # maximum acceptable Value at Risk


class RiskfolioRiskManager:
    """
    Risk manager that enforces position limits and Value at Risk limit.

    This class provides basic risk management functionality for integration
    with Riskfolio-based portfolio optimization strategies.
    """

    def __init__(self, config: RiskfolioConfig) -> None:
        """Initialize the risk manager with configuration."""
        self.config = config

    def calculate_risk(self, returns: np.ndarray) -> dict[str, float]:
        """
        Calculate risk metrics from returns.

        Args:
            returns: Array of portfolio returns

        Returns:
            Dictionary containing risk metrics including VaR at 5% level
        """
        # Value at Risk at 5%: negative quantile
        if returns.size == 0:
            var = 0.0
        else:
            # Ensure var is native float
            var = float(abs(np.percentile(returns, 5)))
        return {"var": var}

    def validate_action(self, action: float, returns: np.ndarray) -> bool:
        """
        Validate if an action is within position limits and risk limits.

        Args:
            action: Trading action value (position size)
            returns: Historical returns for risk calculation

        Returns:
            True if action is valid, False otherwise
        """
        # Check position bounds
        if action < self.config.min_position or action > self.config.max_position:
            return False

        # Check risk limit
        risk = self.calculate_risk(returns)
        return not risk.get("var", 0.0) > self.config.var_limit

    def risk_adjusted_action(self, action: float, returns: np.ndarray) -> float:
        """
        Return risk-adjusted action.

        Args:
            action: Original trading action
            returns: Historical returns for risk calculation

        Returns:
            Original action if valid, otherwise 0.0 to halt trading
        """
        if self.validate_action(action, returns):
            return action
        return 0.0


__all__ = ["RiskfolioConfig", "RiskfolioRiskManager"]
