"""
Risk management module for trading system.

Provides comprehensive risk management including:
- Value at Risk (VaR) calculations
- Position sizing using Kelly criterion
- Drawdown monitoring and controls
- Real-time risk monitoring
- Riskfolio portfolio optimization integration
"""

from .manager import RiskManager
from .position_sizer import kelly_position_size
from .riskfolio import RiskfolioConfig, RiskfolioRiskManager

__all__ = [
    "RiskManager",
    "RiskfolioConfig",
    "RiskfolioRiskManager",
    "kelly_position_size",
]
