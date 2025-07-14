"""
Order execution engine for trading system.

Handles order routing, execution, and broker integration with
sophisticated order management capabilities.
"""

from .broker_interface import BrokerInterface
from .engine import ExecutionEngine
from .order_manager import OrderManager

__all__ = [
    "BrokerInterface",
    "ExecutionEngine",
    "OrderManager",
]
