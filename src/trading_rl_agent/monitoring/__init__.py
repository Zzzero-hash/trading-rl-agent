"""
Performance monitoring and alerting system.

Provides real-time monitoring of trading system performance including:
- Performance metrics tracking
- Risk monitoring and alerts
- Real-time dashboards
- Integration with monitoring tools
"""

from .alerts import AlertManager
from .dashboard import Dashboard
from .metrics import MetricsCollector

__all__ = [
    "AlertManager",
    "Dashboard",
    "MetricsCollector",
]
