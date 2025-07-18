"""
Monitoring and alerting system for the trading RL agent.

This module provides real-time monitoring capabilities including:
- Performance metrics collection and tracking
- Real-time dashboards for system health
- Automated alerting for risk violations and system issues
- Integration with external monitoring systems
"""

from .alert_manager import AlertManager, AlertSeverity, AlertStatus
from .dashboard import Dashboard
from .metrics_collector import MetricsCollector
from .performance_dashboard import PerformanceDashboard, run_performance_dashboard
from .streaming_dashboard import StreamingDashboard, WebSocketClient, run_streaming_dashboard

__all__ = [
    "AlertManager",
    "AlertSeverity",
    "AlertStatus",
    "Dashboard",
    "MetricsCollector",
    "PerformanceDashboard",
    "run_performance_dashboard",
    "StreamingDashboard",
    "WebSocketClient",
    "run_streaming_dashboard",
]
