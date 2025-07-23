"""
Trading RL Agent - Production-grade live trading system.

CORE PRIORITIES:
1. Live Trading Framework - Real-time trading execution
2. Dataset Pipeline - Data ingestion and processing
3. CNN+LSTM Training - Supervised learning for pattern recognition
4. CNN+LSTM Integration - Hybrid RL agents with supervised components
5. Testing - Comprehensive test suite
6. Live Inference - Production model serving

ARCHITECTURE:
    agents/         - RL agents with CNN+LSTM integration
    data/           - Live data pipeline and feature engineering
    models/         - CNN+LSTM models and architectures
    training/       - Training pipeline for both supervised and RL
    envs/           - Trading environments for RL training
    core/           - Core trading framework and utilities
    utils/          - Shared utilities and helpers
"""

__version__ = "0.2.0"
__author__ = "Trade Agent Team"

# Core imports for easy access
# Import functions from root config.py
import sys
from pathlib import Path
from typing import Any

from .core.config import ConfigManager, SystemConfig
from .core.exceptions import DataValidationError, ModelError, TradingSystemError
from .core.logging import get_logger, setup_logging

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_settings, load_settings

# Main components are imported lazily to avoid heavy dependencies at import time.
# These modules rely on ML frameworks such as ``torch`` and ``ray``. Importing
# them here would make ``import trade_agent`` fail in lightweight
# environments. The classes are therefore loaded on demand via ``__getattr__``.

__all__ = [
    "ConfigManager",
    "DataValidationError",
    "ModelError",
    "SystemConfig",
    "TradingSystemError",
    "get_logger",
    "get_settings",
    "load_settings",
    "setup_logging",
]

# Optional heavy components mapped to their import paths. These are imported
# lazily to keep ``import trade_agent`` lightweight.
_OPTIONAL_IMPORTS = {
    # Live Trading Framework
    "LiveTradingEngine": (".core.live_trading", "LiveTradingEngine"),
    "TradingSession": (".core.live_trading", "TradingSession"),
    # Dataset Pipeline
    "DataPipeline": (".data.pipeline", "DataPipeline"),
    "LiveDataFeed": (".data.live_feed", "LiveDataFeed"),
    # CNN+LSTM Models
    "CNNLSTMModel": (".models.cnn_lstm", "CNNLSTMModel"),
    "HybridAgent": (".agents.hybrid", "HybridAgent"),
    # Training
    "Trainer": (".training.trainer", "Trainer"),
    "OptimizedTrainingManager": (
        ".training.optimized_trainer",
        "OptimizedTrainingManager",
    ),
    # RL Agents
    "PPOAgent": (".agents.ppo_agent", "PPOAgent"),
    "SACAgent": (".agents.sac_agent", "SACAgent"),
    "TD3Agent": (".agents.td3_agent", "TD3Agent"),
    # Risk Management
    "RiskManager": (".risk.manager", "RiskManager"),
    "PortfolioManager": (".portfolio.manager", "PortfolioManager"),
}


def __dir__() -> list[str]:
    """Return a list of attributes for tab completion."""
    return sorted(__all__ + list(_OPTIONAL_IMPORTS))


# Mapping of lazily imported components to their modules and attributes
lazy_map = _OPTIONAL_IMPORTS


def __getattr__(name: str) -> Any:
    """Lazily import heavy optional components when accessed."""
    import importlib

    if name in lazy_map:
        module_name, attr = lazy_map[name]
        try:
            module = importlib.import_module(f"{__name__}{module_name}")
            obj = getattr(module, attr)
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                f"{name} requires optional ML dependencies. Install them to use this feature.",
            ) from exc
        except AttributeError as exc:
            raise ImportError(
                f"The module '{module_name}' was found, but it does not contain the attribute '{attr}'. "
                f"Ensure that all dependencies are installed and the module is correctly implemented.",
            ) from exc

        globals()[name] = obj
        return obj

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
