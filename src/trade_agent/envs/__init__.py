"""
Trading environment module for reinforcement learning agents.

This module provides the trading environment implementation for RL agents
to interact with financial markets.
"""

from .finrl_trading_env import TradingEnv, register_env

__all__ = ["TradingEnv", "register_env"]
