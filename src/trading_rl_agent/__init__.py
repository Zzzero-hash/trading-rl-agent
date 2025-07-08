"""
Trading RL Agent - Production-grade hybrid reinforcement learning trading system.

This package provides a comprehensive framework for algorithmic trading that combines:
- CNN+LSTM supervised learning for market pattern recognition
- Deep reinforcement learning for trading decision optimization
- Risk management and portfolio optimization
- Real-time data processing and execution

Architecture:
    agents/         - RL agents and ensemble methods
    data/           - Data ingestion, processing, and feature engineering
    features/       - Feature engineering and technical indicators
    portfolio/      - Portfolio management and optimization
    risk/           - Risk management and compliance
    execution/      - Order execution and broker integration
    monitoring/     - Performance monitoring and alerting
    utils/          - Shared utilities and helpers
"""

__version__ = "2.0.0"
__author__ = "Trading RL Team"

# Core imports for easy access
from .core.config import ConfigManager, SystemConfig
from .core.logging import setup_logging, get_logger
from .core.exceptions import TradingSystemError, DataValidationError, ModelError

# Main components are imported lazily to avoid heavy dependencies at import time.
# These modules rely on ML frameworks such as ``torch`` and ``ray``.  Importing
# them here would make ``import trading_rl_agent`` fail in lightweight
# environments.  The classes are therefore loaded on demand via ``__getattr__``.

__all__ = [
    # Core
    "ConfigManager",
    "SystemConfig",
    "setup_logging",
    "get_logger",
    "TradingSystemError",
    "DataValidationError",
    "ModelError",
    # Main components
    "Agent",
    "EnsembleAgent",
    "DataPipeline",
    "MarketDataLoader",
    "PortfolioManager",
    "RiskManager",
    "ExecutionEngine",
]


def __dir__():
    """Return a list of attributes for tab completion."""
    lazy_keys = [
        "Agent",
        "EnsembleAgent",
        "Trainer",
        "DataPipeline",
        "MarketDataLoader",
        "PortfolioManager",
        "RiskManager",
        "ExecutionEngine",
    ]
    return sorted(__all__ + lazy_keys)
def __getattr__(name: str):
    """Lazily import heavy optional components when accessed."""
    import importlib

    lazy_map = {
        # Agents and trainer related components
        "Agent": (".agents", "Agent"),
        "EnsembleAgent": (".agents", "EnsembleAgent"),
        "Trainer": (".agents.trainer", "Trainer"),
        # Data handling
        "DataPipeline": (".data", "DataPipeline"),
        "MarketDataLoader": (".data", "MarketDataLoader"),
        # Portfolio, risk and execution modules
        "PortfolioManager": (".portfolio", "PortfolioManager"),
        "RiskManager": (".risk", "RiskManager"),
        "ExecutionEngine": (".execution", "ExecutionEngine"),
    }

    if name in lazy_map:
        module_name, attr = lazy_map[name]
        try:
            module = importlib.import_module(f"{__name__}{module_name}")
            obj = getattr(module, attr)
        except Exception as exc:  # pragma: no cover - just in case
            raise ImportError(
                f"{name} requires optional ML dependencies. Install them to use this feature."
            ) from exc

        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
