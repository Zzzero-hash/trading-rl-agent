"""
Tests for the monitoring module.
"""

import time

from trade_agent.monitoring import (
    AlertManager,
    AlertSeverity,
    AlertStatus,
    Dashboard,
    MetricsCollector,
)


class TestMetricsCollector:
    """Test suite for MetricsCollector."""

    def test_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector(max_history=1000)

        assert collector.max_history == 1000
        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
        assert len(collector.histograms) == 0

    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()

        collector.record_metric("test_metric", 42.5, {"source": "test"})

        assert "test_metric" in collector.metrics
        assert len(collector.metrics["test_metric"]) == 1

        metric_point = collector.metrics["test_metric"][0]
        assert metric_point.value == 42.5
        assert metric_point.metadata["source"] == "test"
        assert isinstance(metric_point.timestamp, float)

    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()

        collector.increment_counter("test_counter")
        assert collector.get_counter_value("test_counter") == 1

        collector.increment_counter("test_counter", 5)
        assert collector.get_counter_value("test_counter") == 6

    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()

        collector.set_gauge("test_gauge", 100.0)
        assert collector.get_gauge_value("test_gauge") == 100.0

        collector.set_gauge("test_gauge", 200.0)
        assert collector.get_gauge_value("test_gauge") == 200.0

    def test_record_histogram(self):
        """Test histogram recording."""
        collector = MetricsCollector()

        collector.record_histogram("test_hist", 1.0)
        collector.record_histogram("test_hist", 2.0)
        collector.record_histogram("test_hist", 3.0)

        summary = collector.get_histogram_summary("test_hist")
        assert summary["count"] == 3
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["mean"] == 2.0

    def test_get_metric_summary(self):
        """Test metric summary generation."""
        collector = MetricsCollector()

        # Record some metrics
        collector.record_metric("test_metric", 10.0)
        collector.record_metric("test_metric", 20.0)
        collector.record_metric("test_metric", 30.0)

        summary = collector.get_metric_summary("test_metric")
        assert summary["count"] == 3
        assert summary["min"] == 10.0
        assert summary["max"] == 30.0
        assert summary["mean"] == 20.0
        assert summary["latest"] == 30.0

    def test_export_metrics(self):
        """Test metrics export."""
        collector = MetricsCollector()

        # Add some data
        collector.record_metric("test_metric", 42.0)
        collector.increment_counter("test_counter")
        collector.set_gauge("test_gauge", 100.0)
        collector.record_histogram("test_hist", 1.0)

        exported = collector.export_metrics()

        assert "metrics" in exported
        assert "counters" in exported
        assert "gauges" in exported
        assert "histograms" in exported
        assert "timestamp" in exported
        assert exported["counters"]["test_counter"] == 1
        assert exported["gauges"]["test_gauge"] == 100.0


class TestAlertManager:
    """Test suite for AlertManager."""

    def test_initialization(self):
        """Test AlertManager initialization."""
        manager = AlertManager(max_alerts=500)

        assert manager.max_alerts == 500
        assert len(manager.alerts) == 0
        assert len(manager.alert_handlers) == 0
        assert len(manager.thresholds) == 0

    def test_create_alert(self):
        """Test alert creation."""
        manager = AlertManager()

        alert = manager.create_alert(
            title="Test Alert",
            message="This is a test alert",
            severity=AlertSeverity.WARNING,
            source="test_system",
            alert_type="test_type",
        )

        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "test_system"
        assert alert.status == AlertStatus.ACTIVE
        assert len(manager.alerts) == 1

    def test_set_threshold(self):
        """Test threshold setting."""
        manager = AlertManager()

        manager.set_threshold("cpu_usage", "max", 90.0)
        manager.set_threshold("memory_usage", "min", 10.0)

        assert "cpu_usage" in manager.thresholds
        assert "memory_usage" in manager.thresholds
        assert manager.thresholds["cpu_usage"]["max"] == 90.0
        assert manager.thresholds["memory_usage"]["min"] == 10.0

    def test_check_threshold(self):
        """Test threshold checking."""
        manager = AlertManager()

        # Set thresholds
        manager.set_threshold("cpu_usage", "max", 80.0)
        manager.set_threshold("memory_usage", "min", 20.0)

        # Check within limits
        alerts = manager.check_threshold("cpu_usage", 70.0)
        assert len(alerts) == 0

        # Check above max threshold
        alerts = manager.check_threshold("cpu_usage", 90.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

        # Check below min threshold
        alerts = manager.check_threshold("memory_usage", 10.0)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        manager = AlertManager()

        alert = manager.create_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.ERROR,
            source="test",
        )

        success = manager.acknowledge_alert(alert.id, "test_user")
        assert success is True
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"

    def test_resolve_alert(self):
        """Test alert resolution."""
        manager = AlertManager()

        alert = manager.create_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.ERROR,
            source="test",
        )

        success = manager.resolve_alert(alert.id)
        assert success is True
        assert alert.status == AlertStatus.RESOLVED

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        manager = AlertManager()

        # Create alerts with different statuses
        alert1 = manager.create_alert(
            title="Active Alert",
            message="Active message",
            severity=AlertSeverity.WARNING,
            source="test",
        )

        alert2 = manager.create_alert(
            title="Another Alert",
            message="Another message",
            severity=AlertSeverity.ERROR,
            source="test",
        )

        # Verify both alerts are initially active
        assert len(manager.get_active_alerts()) == 2

        manager.resolve_alert(alert2.id)

        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].id == alert1.id

        warning_alerts = manager.get_active_alerts(AlertSeverity.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].severity == AlertSeverity.WARNING

    def test_get_alert_summary(self):
        """Test alert summary generation."""
        manager = AlertManager()

        # Create alerts with different severities
        manager.create_alert(
            title="Info Alert",
            message="Info message",
            severity=AlertSeverity.INFO,
            source="test1",
        )

        manager.create_alert(
            title="Warning Alert",
            message="Warning message",
            severity=AlertSeverity.WARNING,
            source="test2",
        )

        manager.create_alert(
            title="Error Alert",
            message="Error message",
            severity=AlertSeverity.ERROR,
            source="test1",
        )

        summary = manager.get_alert_summary()

        assert summary["total_alerts"] == 3
        assert summary["active_alerts"] == 3
        assert summary["severity_counts"]["info"] == 1
        assert summary["severity_counts"]["warning"] == 1
        assert summary["severity_counts"]["error"] == 1
        assert summary["source_counts"]["test1"] == 2
        assert summary["source_counts"]["test2"] == 1


class TestDashboard:
    """Test suite for Dashboard."""

    def test_initialization(self):
        """Test Dashboard initialization."""
        dashboard = Dashboard()

        assert dashboard.metrics_collector is None
        assert dashboard.alert_manager is None
        assert dashboard.update_interval == 1.0
        assert isinstance(dashboard.last_update, float)

    def test_get_system_overview(self):
        """Test system overview generation."""
        dashboard = Dashboard()

        overview = dashboard.get_system_overview()

        assert "timestamp" in overview
        assert "last_update" in overview
        assert "system_status" in overview
        assert "uptime" in overview
        assert overview["system_status"] == "healthy"

    def test_get_trading_metrics(self):
        """Test trading metrics generation."""
        dashboard = Dashboard()

        metrics = dashboard.get_trading_metrics()

        assert "timestamp" in metrics
        assert "pnl" in metrics
        assert "daily_pnl" in metrics
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "total_trades" in metrics
        assert "open_positions" in metrics

    def test_get_risk_metrics(self):
        """Test risk metrics generation."""
        dashboard = Dashboard()

        metrics = dashboard.get_risk_metrics()

        assert "timestamp" in metrics
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        assert "current_exposure" in metrics
        assert "position_concentration" in metrics
        assert "volatility" in metrics
        assert "beta" in metrics

    def test_get_model_metrics(self):
        """Test model metrics generation."""
        dashboard = Dashboard()

        metrics = dashboard.get_model_metrics()

        assert "timestamp" in metrics
        assert "model_accuracy" in metrics
        assert "model_loss" in metrics
        assert "prediction_latency" in metrics
        assert "model_confidence" in metrics
        assert "training_status" in metrics

    def test_get_system_health(self):
        """Test system health metrics generation."""
        dashboard = Dashboard()

        health = dashboard.get_system_health()

        assert "timestamp" in health
        assert "cpu_usage" in health
        assert "memory_usage" in health
        assert "disk_usage" in health
        assert "network_latency" in health
        assert "error_rate" in health
        assert "response_time" in health

    def test_get_recent_alerts(self):
        """Test recent alerts retrieval."""
        alert_manager = AlertManager()
        dashboard = Dashboard(alert_manager=alert_manager)

        # Create some alerts
        alert_manager.create_alert(
            title="Test Alert 1",
            message="First alert",
            severity=AlertSeverity.WARNING,
            source="test",
        )

        alert_manager.create_alert(
            title="Test Alert 2",
            message="Second alert",
            severity=AlertSeverity.ERROR,
            source="test",
        )

        recent_alerts = dashboard.get_recent_alerts(limit=1)
        assert len(recent_alerts) == 1
        assert recent_alerts[0]["title"] == "Test Alert 2"  # Most recent first

    def test_should_update(self):
        """Test update interval checking."""
        dashboard = Dashboard()

        # Should not update immediately
        assert not dashboard.should_update()

        # Wait and then should update
        time.sleep(1.1)
        assert dashboard.should_update()

    def test_update(self):
        """Test dashboard update."""
        dashboard = Dashboard()

        old_update_time = dashboard.last_update
        time.sleep(0.1)  # Small delay

        dashboard.update()

        assert dashboard.last_update > old_update_time
