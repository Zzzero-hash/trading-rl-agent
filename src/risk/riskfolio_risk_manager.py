from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np
from riskfolio import RiskFunctions


@dataclass
class RiskfolioConfig:
    """Configuration for :class:`RiskfolioRiskManager`."""

    max_position: float = 1.0
    min_position: float = -1.0
    var_limit: float = 0.02
    var_alpha: float = 0.05


class RiskfolioRiskManager:
    """Risk management wrapper using ``riskfolio-lib`` for VaR calculations."""

    def __init__(self, config: RiskfolioConfig) -> None:
        self.config = config

    def calculate_risk(self, returns: Sequence[float]) -> Dict[str, float]:
        """Calculate portfolio risk metrics using historical VaR."""
        if len(returns) == 0:
            raise ValueError("Returns sequence cannot be empty")
        arr = np.asarray(returns, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            raise ValueError("Returns must contain valid numerical data")
        var = RiskFunctions.VaR_Hist(arr, alpha=self.config.var_alpha)
        return {"var": float(var)}

    def validate_action(self, action: float, returns: Sequence[float]) -> bool:
        """Check whether an action is within limits and risk thresholds."""
        if not (self.config.min_position <= action <= self.config.max_position):
            return False
        risk = self.calculate_risk(returns)
        return risk["var"] <= self.config.var_limit

    def risk_adjusted_action(self, action: float, returns: Sequence[float]) -> float:
        """Return zero if the action violates risk policies."""
        return action if self.validate_action(action, returns) else 0.0

