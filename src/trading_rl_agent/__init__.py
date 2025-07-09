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
from .core.exceptions import DataValidationError, ModelError, TradingSystemError
from .core.logging import get_logger, setup_logging

# Main components are imported lazily to avoid heavy dependencies at import time.
# These modules rely on ML frameworks such as ``torch`` and ``ray``. Importing
# them here would make ``import trading_rl_agent`` fail in lightweight
# environments. The classes are therefore loaded on demand via ``__getattr__``.

__all__ = [
    "ConfigManager",
    "SystemConfig",
    "setup_logging",
    "get_logger",
    "TradingSystemError",
    "DataValidationError",
    "ModelError",
]

# Optional heavy components mapped to their import paths.  These are imported
# lazily to keep ``import trading_rl_agent`` lightweight.
_OPTIONAL_IMPORTS = {
    "Trainer": (".agents.trainer", "Trainer"),
    "WeightedEnsembleAgent": (".agents.policy_utils", "WeightedEnsembleAgent"),
    "EnsembleAgent": (".agents.policy_utils", "EnsembleAgent"),
    "CallablePolicy": (".agents.policy_utils", "CallablePolicy"),
    "weighted_policy_mapping": (".agents.policy_utils", "weighted_policy_mapping"),
    "PortfolioManager": (".portfolio.manager", "PortfolioManager"),
    "ExecutionEngine": (".execution", "ExecutionEngine"),
}


def __dir__():
    """Return a list of attributes for tab completion."""
    return sorted(__all__ + list(_OPTIONAL_IMPORTS))


# Mapping of lazily imported components to their modules and attributes
lazy_map = _OPTIONAL_IMPORTS


def __getattr__(name: str):
    """Lazily import heavy optional components when accessed."""
    import importlib

    if name in lazy_map:
        module_name, attr = lazy_map[name]
        try:
            module = importlib.import_module(f"{__name__}{module_name}")
            obj = getattr(module, attr)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                f"{name} requires optional ML dependencies. Install them to use this feature."
            ) from exc
        except AttributeError as exc:
            raise ImportError(
                f"The module '{module_name}' was found, but it does not contain the attribute '{attr}'. "
                f"Ensure that all dependencies are installed and the module is correctly implemented."
            ) from exc

        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
