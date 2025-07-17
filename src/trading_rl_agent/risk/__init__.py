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
from .position_sizer import kelly_position_size
from .riskfolio import RiskfolioConfig, RiskfolioRiskManager
from .monte_carlo_var import (
    MonteCarloVaR,
    MonteCarloVaRConfig,
    VaRResult
)
from .parallel_var import (
    ParallelVaRCalculator,
    ParallelVaRConfig
)

__all__ = [
    "RiskManager",
    "RiskfolioConfig",
    "RiskfolioRiskManager",
    "kelly_position_size",
    "MonteCarloVaR",
    "MonteCarloVaRConfig",
    "VaRResult",
    "ParallelVaRCalculator",
    "ParallelVaRConfig",
]
