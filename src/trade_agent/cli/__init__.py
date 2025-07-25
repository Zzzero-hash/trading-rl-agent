"""
Modular CLI package for Trading RL Agent.

This package provides a well-structured command-line interface split into domain-specific modules:
- cli_main.py: Main app and shared utilities
- cli_data.py: Data pipeline operations
- cli_train.py: Model training operations
- cli_backtest.py: Backtesting and evaluation
- cli_trade.py: Live trading operations
"""

# Import main app after all modules are loaded
from .cli_main import app, backtest_app, data_app, trade_app, train_app

__all__ = ["app", "backtest_app", "data_app", "trade_app", "train_app"]
