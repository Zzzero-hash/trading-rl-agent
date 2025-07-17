"""
Risk management module for trading system.

Provides comprehensive risk management including:
- Value at Risk (VaR) calculations
- Advanced Monte Carlo VaR with multiple simulation methods
- Historical simulation with bootstrapping
- Parametric VaR with multiple distribution assumptions
- Monte Carlo simulation with correlated asset movements
- Stress testing scenarios for extreme market conditions
- Parallel processing for large simulations
- VaR backtesting and validation methods
- Position sizing using Kelly criterion
- Drawdown monitoring and controls
- Real-time risk monitoring
- Riskfolio portfolio optimization integration
"""

from .manager import RiskManager
from .monte_carlo_var import MonteCarloVaR, MonteCarloVaRConfig, VaRResult
from .parallel_var import ParallelVaRCalculator, ParallelVaRConfig
from .position_sizer import kelly_position_size
from .riskfolio import RiskfolioConfig, RiskfolioRiskManager

__all__ = [
    "MonteCarloVaR",
    "MonteCarloVaRConfig",
    "ParallelVaRCalculator",
    "ParallelVaRConfig",
    "RiskManager",
    "RiskfolioConfig",
    "RiskfolioRiskManager",
    "VaRResult",
    "kelly_position_size",
]
