"""
Risk management module for trading system.

Provides comprehensive risk management including:
- Value at Risk (VaR) calculations
- Position sizing using Kelly criterion
- Drawdown monitoring and controls
- Real-time risk monitoring
"""

from .manager import RiskManager
from .position_sizer import kelly_position_size

__all__ = [
    "RiskManager",
    "kelly_position_size",
]
