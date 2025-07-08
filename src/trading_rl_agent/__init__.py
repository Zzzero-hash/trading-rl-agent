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

# Main components
from .agents import Agent, EnsembleAgent
from .data import DataPipeline, MarketDataLoader
from .portfolio import PortfolioManager
from .risk import RiskManager
from .execution import ExecutionEngine

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
