"""
Alert management system for the trading RL agent.

This module provides comprehensive alerting capabilities for:
- Risk threshold violations
- System health issues
- Performance degradation
- Trading anomalies
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


@dataclass
class Alert:
    """Represents a single alert."""

    id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None
    resolved_at: float | None = None


class AlertManager:
    """Manages alerts and notifications for the trading system."""

    def __init__(self, max_alerts: int = 1000):
        """Initialize the alert manager.

        Args:
            max_alerts: Maximum number of alerts to keep in memory
        """
        self.max_alerts = max_alerts
        self.alerts: list[Alert] = []
        self.alert_handlers: dict[str, Callable[[Alert], None]] = {}
        self.thresholds: dict[str, dict[str, float | int]] = {}
        self.alert_counters: dict[str, int] = {}
        self._alert_id_counter = 0

    def add_alert_handler(self, alert_type: str, handler: Callable[[Alert], None]) -> None:
        """Add a handler for a specific alert type.

        Args:
            alert_type: Type of alert to handle
            handler: Function to call when alert is triggered
        """
        self.alert_handlers[alert_type] = handler

    def set_threshold(self, metric_name: str, threshold_type: str, value: float | int) -> None:
        """Set a threshold for a metric.

        Args:
            metric_name: Name of the metric to monitor
            threshold_type: Type of threshold (min, max, etc.)
            value: Threshold value
        """
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        self.thresholds[metric_name][threshold_type] = value

    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        source: str,
        alert_type: str = "general",
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        """Create and register a new alert.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity
            source: Source of the alert
            alert_type: Type of alert
            metadata: Optional metadata

        Returns:
            Created alert object
        """
        self._alert_id_counter += 1
        alert_id = f"{alert_type}_{int(time.time() * 1000)}_{self._alert_id_counter}"

        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        self.alerts.append(alert)

        # Keep only the most recent alerts
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts :]

        # Increment counter
        self.alert_counters[alert_type] = self.alert_counters.get(alert_type, 0) + 1

        # Call handler if registered
        if alert_type in self.alert_handlers:
            try:
                self.alert_handlers[alert_type](alert)
            except Exception as e:
                # Log error but don't fail
                print(f"Error in alert handler for {alert_type}: {e}")

        return alert

    def check_threshold(self, metric_name: str, value: float | int) -> list[Alert]:
        """Check if a metric value violates any thresholds.

        Args:
            metric_name: Name of the metric
            value: Current metric value

        Returns:
            List of alerts created for threshold violations
        """
        alerts: list[Any] = []

        if metric_name not in self.thresholds:
            return alerts

        thresholds = self.thresholds[metric_name]

        for threshold_type, threshold_value in thresholds.items():
            if threshold_type == "min" and value < threshold_value:
                alert = self.create_alert(
                    title=f"{metric_name} below minimum threshold",
                    message=f"{metric_name} = {value} is below minimum threshold of {threshold_value}",
                    severity=AlertSeverity.WARNING,
                    source="threshold_monitor",
                    alert_type=f"{metric_name}_threshold",
                    metadata={"metric": metric_name, "value": value, "threshold": threshold_value},
                )
                alerts.append(alert)

            elif threshold_type == "max" and value > threshold_value:
                alert = self.create_alert(
                    title=f"{metric_name} above maximum threshold",
                    message=f"{metric_name} = {value} is above maximum threshold of {threshold_value}",
                    severity=AlertSeverity.WARNING,
                    source="threshold_monitor",
                    alert_type=f"{metric_name}_threshold",
                    metadata={"metric": metric_name, "value": value, "threshold": threshold_value},
                )
                alerts.append(alert)

        return alerts

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Name of person acknowledging the alert

        Returns:
            True if alert was found and acknowledged, False otherwise
        """
        for alert in self.alerts:
            if alert.id == alert_id and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: ID of the alert to resolve

        Returns:
            True if alert was found and resolved, False otherwise
        """
        for alert in self.alerts:
            if alert.id == alert_id and alert.status != AlertStatus.RESOLVED:
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                return True
        return False

    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert.

        Args:
            alert_id: ID of the alert to dismiss

        Returns:
            True if alert was found and dismissed, False otherwise
        """
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.status = AlertStatus.DISMISSED
                return True
        return False

    def get_active_alerts(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get all active alerts.

        Args:
            severity: Optional severity filter

        Returns:
            List of active alerts
        """
        active_alerts = [alert for alert in self.alerts if alert.status == AlertStatus.ACTIVE]

        if severity is not None:
            active_alerts = [alert for alert in active_alerts if alert.severity == severity]

        return active_alerts

    def get_alerts_by_source(self, source: str) -> list[Alert]:
        """Get all alerts from a specific source.

        Args:
            source: Source name

        Returns:
            List of alerts from the source
        """
        return [alert for alert in self.alerts if alert.source == source]

    def get_alerts_by_type(self, alert_type: str) -> list[Alert]:
        """Get all alerts of a specific type.

        Args:
            alert_type: Alert type

        Returns:
            List of alerts of the specified type
        """
        return [alert for alert in self.alerts if alert.id.startswith(f"{alert_type}_")]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get a summary of all alerts.

        Returns:
            Dictionary with alert summary statistics
        """
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())

        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([alert for alert in self.alerts if alert.severity == severity])

        source_counts: dict[str, int] = {}
        for alert in self.alerts:
            source_counts[alert.source] = source_counts.get(alert.source, 0) + 1

        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "severity_counts": severity_counts,
            "source_counts": source_counts,
            "type_counts": dict(self.alert_counters),
            "timestamp": time.time(),
        }

    def clear_alerts(self, status: AlertStatus | None = None) -> int:
        """Clear alerts.

        Args:
            status: Only clear alerts with this status, or None to clear all

        Returns:
            Number of alerts cleared
        """
        if status is None:
            cleared_count = len(self.alerts)
            self.alerts.clear()
            return cleared_count
        original_count = len(self.alerts)
        self.alerts = [alert for alert in self.alerts if alert.status != status]
        return original_count - len(self.alerts)
