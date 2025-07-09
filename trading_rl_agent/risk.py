"""Risk management modules for Trading RL Agent."""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RiskfolioConfig:
    max_position: float
    min_position: float
    var_limit: float


class RiskfolioRiskManager:
    """Risk manager that enforces position limits and Value at Risk limit."""

    def __init__(self, config: RiskfolioConfig) -> None:
        self.config = config

    def calculate_risk(self, returns: np.ndarray) -> dict[str, float]:
        if returns.size == 0:
            var = 0.0
        else:
            var = float(abs(np.percentile(returns, 5)))
        return {"var": var}

    def validate_action(self, action: float, returns: np.ndarray) -> bool:
        if action < self.config.min_position or action > self.config.max_position:
            return False
        risk = self.calculate_risk(returns)
        if risk.get("var", 0.0) > self.config.var_limit:
            return False
        return True

    def risk_adjusted_action(self, action: float, returns: np.ndarray) -> float:
        return action if self.validate_action(action, returns) else 0.0
