"""
Real-time dashboard for the trading system.

This module provides a dashboard interface for monitoring:
- System health and performance
- Trading metrics and P&L
- Risk metrics and alerts
- Model performance
"""

import time
from typing import Any


class Dashboard:
    """Real-time dashboard for monitoring the trading system."""

    def __init__(self, metrics_collector: Any = None, alert_manager: Any = None) -> None:
        """Initialize the dashboard.

        Args:
            metrics_collector: Optional MetricsCollector instance
            alert_manager: Optional AlertManager instance
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.last_update = time.time()
        self.update_interval = 1.0  # seconds

    def get_system_overview(self) -> dict[str, Any]:
        """Get system overview information.

        Returns:
            Dictionary with system overview data
        """
        overview = {
            "timestamp": time.time(),
            "last_update": self.last_update,
            "system_status": "healthy",
            "uptime": self._get_uptime(),
        }

        if self.metrics_collector:
            metrics = self.metrics_collector.export_metrics()
            overview["metrics_summary"] = {
                "total_metrics": len(metrics["metrics"]),
                "total_counters": len(metrics["counters"]),
                "total_gauges": len(metrics["gauges"]),
                "total_histograms": len(metrics["histograms"]),
            }

        if self.alert_manager:
            alert_summary = self.alert_manager.get_alert_summary()
            overview["alerts_summary"] = alert_summary
            overview["system_status"] = self._determine_system_status(alert_summary)

        return overview

    def get_trading_metrics(self) -> dict[str, Any]:
        """Get trading-specific metrics.

        Returns:
            Dictionary with trading metrics
        """
        trading_metrics = {
            "timestamp": time.time(),
            "pnl": 0.0,
            "daily_pnl": 0.0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "open_positions": 0,
        }

        if self.metrics_collector:
            # Get trading-specific metrics
            for metric_name in ["pnl", "daily_pnl", "total_return", "sharpe_ratio"]:
                summary = self.metrics_collector.get_metric_summary(metric_name)
                if summary["latest"] is not None:
                    trading_metrics[metric_name] = summary["latest"]

            # Get counters
            trading_metrics["total_trades"] = self.metrics_collector.get_counter_value("total_trades")
            trading_metrics["open_positions"] = self.metrics_collector.get_gauge_value("open_positions") or 0

        return trading_metrics

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get risk-related metrics.

        Returns:
            Dictionary with risk metrics
        """
        risk_metrics = {
            "timestamp": time.time(),
            "var_95": 0.0,
            "cvar_95": 0.0,
            "current_exposure": 0.0,
            "position_concentration": 0.0,
            "volatility": 0.0,
            "beta": 0.0,
        }

        if self.metrics_collector:
            # Get risk metrics
            for metric_name in ["var_95", "cvar_95", "current_exposure", "volatility", "beta"]:
                summary = self.metrics_collector.get_metric_summary(metric_name)
                if summary["latest"] is not None:
                    risk_metrics[metric_name] = summary["latest"]

        return risk_metrics

    def get_model_metrics(self) -> dict[str, Any]:
        """Get model performance metrics.

        Returns:
            Dictionary with model metrics
        """
        model_metrics = {
            "timestamp": time.time(),
            "model_accuracy": 0.0,
            "model_loss": 0.0,
            "prediction_latency": 0.0,
            "model_confidence": 0.0,
            "training_status": "idle",
        }

        if self.metrics_collector:
            # Get model metrics
            for metric_name in ["model_accuracy", "model_loss", "prediction_latency", "model_confidence"]:
                summary = self.metrics_collector.get_metric_summary(metric_name)
                if summary["latest"] is not None:
                    model_metrics[metric_name] = summary["latest"]

            # Get training status
            training_status = self.metrics_collector.get_gauge_value("training_status")
            if training_status is not None:
                model_metrics["training_status"] = "training" if training_status > 0 else "idle"

        return model_metrics

    def get_system_health(self) -> dict[str, Any]:
        """Get system health metrics.

        Returns:
            Dictionary with system health data
        """
        health_metrics = {
            "timestamp": time.time(),
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_latency": 0.0,
            "error_rate": 0.0,
            "response_time": 0.0,
        }

        if self.metrics_collector:
            # Get system metrics
            for metric_name in [
                "cpu_usage",
                "memory_usage",
                "disk_usage",
                "network_latency",
                "error_rate",
                "response_time",
            ]:
                summary = self.metrics_collector.get_metric_summary(metric_name)
                if summary["latest"] is not None:
                    health_metrics[metric_name] = summary["latest"]

        return health_metrics

    def get_recent_alerts(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of recent alerts
        """
        if not self.alert_manager:
            return []

        active_alerts = self.alert_manager.get_active_alerts()

        # Sort by timestamp (most recent first)
        active_alerts.sort(key=lambda x: x.timestamp, reverse=True)

        # Convert to dictionary format
        alerts_data = []
        for alert in active_alerts[:limit]:
            alerts_data.append(
                {
                    "id": alert.id,
                    "title": alert.title,
                    "message": alert.message,
                    "severity": alert.severity.value,
                    "source": alert.source,
                    "timestamp": alert.timestamp,
                    "status": alert.status.value,
                },
            )

        return alerts_data

    def get_metric_trends(self, metric_name: str, hours: int = 24) -> dict[str, Any]:
        """Get trend data for a specific metric.

        Args:
            metric_name: Name of the metric
            hours: Number of hours to look back

        Returns:
            Dictionary with trend data
        """
        if not self.metrics_collector:
            return {"data": [], "summary": {}}

        end_time = time.time()
        start_time = end_time - (hours * 3600)

        history = self.metrics_collector.get_metric_history(metric_name, start_time, end_time)

        # Convert to time series data
        data = []
        for point in history:
            data.append(
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                },
            )

        # Calculate summary statistics
        values = [p.value for p in history if isinstance(p.value, (int, float))]
        summary = {}
        if values:
            summary = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "latest": values[-1] if values else None,
            }

        return {
            "metric_name": metric_name,
            "data": data,
            "summary": summary,
            "time_range": {
                "start": start_time,
                "end": end_time,
                "hours": hours,
            },
        }

    def update(self) -> None:
        """Update the dashboard data."""
        self.last_update = time.time()

    def should_update(self) -> bool:
        """Check if dashboard should be updated.

        Returns:
            True if dashboard should be updated
        """
        return (time.time() - self.last_update) >= self.update_interval

    def set_update_interval(self, interval: float) -> None:
        """Set the update interval.

        Args:
            interval: Update interval in seconds
        """
        self.update_interval = interval

    def _get_uptime(self) -> float:
        """Get system uptime.

        Returns:
            Uptime in seconds
        """
        # This is a simplified implementation
        # In a real system, you'd track the actual start time
        return time.time() - (24 * 3600)  # Assume 24 hours for demo

    def _determine_system_status(self, alert_summary: dict[str, Any]) -> str:
        """Determine system status based on alerts.

        Args:
            alert_summary: Alert summary from AlertManager

        Returns:
            System status string
        """
        if not alert_summary:
            return "healthy"

        severity_counts = alert_summary.get("severity_counts", {})

        if severity_counts.get("critical", 0) > 0:
            return "critical"
        if severity_counts.get("error", 0) > 0:
            return "error"
        if severity_counts.get("warning", 0) > 0:
            return "warning"
        return "healthy"
