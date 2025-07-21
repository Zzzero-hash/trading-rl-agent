"""
Test suite for the Risk Alert System.

Tests all major functionality including:
- Alert threshold monitoring
- Circuit breaker functionality
- Escalation procedures
- Notification systems
- Reporting and audit logging
"""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from src.trading_rl_agent.monitoring.alert_manager import AlertManager, AlertSeverity
from src.trading_rl_agent.risk.alert_system import (
    AlertThreshold,
    CircuitBreakerRule,
    CircuitBreakerStatus,
    EscalationLevel,
    NotificationConfig,
    RiskAlertConfig,
    RiskAlertSystem,
)
from src.trading_rl_agent.risk.manager import RiskLimits, RiskManager, RiskMetrics


class TestRiskAlertConfig:
    """Test RiskAlertConfig validation and functionality."""

    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = RiskAlertConfig(
            monitoring_interval_seconds=60,
            real_time_monitoring=True,
            alert_thresholds=[],
            circuit_breaker_rules=[],
            notifications=NotificationConfig(),
            escalation_enabled=True,
            escalation_timeout_minutes=30,
            audit_log_enabled=True,
            audit_log_path="logs/test.log",
            report_generation_enabled=True,
            report_schedule_hours=24,
        )

        assert config.monitoring_interval_seconds == 60
        assert config.real_time_monitoring is True
        assert config.escalation_enabled is True
        assert config.audit_log_enabled is True

    def test_invalid_config(self):
        """Test configuration validation errors."""
        with pytest.raises(ValidationError):
            RiskAlertConfig(monitoring_interval_seconds=-1)

        with pytest.raises(ValidationError):
            RiskAlertConfig(escalation_timeout_minutes=-1)

    def test_default_config(self):
        """Test default configuration values."""
        config = RiskAlertConfig()

        assert config.monitoring_interval_seconds == 60
        assert config.real_time_monitoring is True
        assert config.escalation_enabled is True
        assert config.audit_log_enabled is True


class TestAlertThreshold:
    """Test AlertThreshold functionality."""

    def test_alert_threshold_creation(self):
        """Test creating alert thresholds."""
        threshold = AlertThreshold(
            metric_name="portfolio_var",
            threshold_type="max",
            threshold_value=0.05,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2,
            cooldown_minutes=30,
            enabled=True,
            description="Test threshold",
        )

        assert threshold.metric_name == "portfolio_var"
        assert threshold.threshold_type == "max"
        assert threshold.threshold_value == 0.05
        assert threshold.severity == AlertSeverity.WARNING
        assert threshold.escalation_level == EscalationLevel.LEVEL_2
        assert threshold.cooldown_minutes == 30
        assert threshold.enabled is True
        assert threshold.description == "Test threshold"

    def test_alert_threshold_defaults(self):
        """Test alert threshold default values."""
        threshold = AlertThreshold(
            metric_name="test_metric",
            threshold_type="min",
            threshold_value=0.1,
            severity=AlertSeverity.ERROR,
            escalation_level=EscalationLevel.LEVEL_3,
        )

        assert threshold.cooldown_minutes == 30
        assert threshold.enabled is True
        assert threshold.description == ""


class TestCircuitBreakerRule:
    """Test CircuitBreakerRule functionality."""

    def test_circuit_breaker_rule_creation(self):
        """Test creating circuit breaker rules."""
        rule = CircuitBreakerRule(
            name="test_rule",
            trigger_condition="var_exceeded",
            threshold_value=0.1,
            action="stop_trading",
            cooldown_minutes=60,
            enabled=True,
            description="Test rule",
        )

        assert rule.name == "test_rule"
        assert rule.trigger_condition == "var_exceeded"
        assert rule.threshold_value == 0.1
        assert rule.action == "stop_trading"
        assert rule.cooldown_minutes == 60
        assert rule.enabled is True
        assert rule.description == "Test rule"

    def test_circuit_breaker_rule_defaults(self):
        """Test circuit breaker rule default values."""
        rule = CircuitBreakerRule(
            name="test_rule",
            trigger_condition="drawdown_exceeded",
            threshold_value=0.2,
            action="liquidate",
        )

        assert rule.cooldown_minutes == 60
        assert rule.enabled is True
        assert rule.description == ""


class TestNotificationConfig:
    """Test NotificationConfig functionality."""

    def test_notification_config_creation(self):
        """Test creating notification configuration."""
        config = NotificationConfig(
            email_enabled=True,
            email_recipients=["test@example.com"],
            email_smtp_server="smtp.gmail.com",
            email_smtp_port=587,
            email_username="test@example.com",
            email_password="password",
            slack_enabled=True,
            slack_webhook_url="https://hooks.slack.com/test",
            slack_channel="#test",
            sms_enabled=False,
            webhook_enabled=False,
        )

        assert config.email_enabled is True
        assert config.email_recipients == ["test@example.com"]
        assert config.slack_enabled is True
        assert config.sms_enabled is False
        assert config.webhook_enabled is False

    def test_notification_config_defaults(self):
        """Test notification configuration default values."""
        config = NotificationConfig()

        assert config.email_enabled is True
        assert config.slack_enabled is False
        assert config.sms_enabled is False
        assert config.webhook_enabled is False
        assert config.email_recipients == []
        assert config.sms_recipients == []


class TestRiskAlertSystem:
    """Test RiskAlertSystem functionality."""

    @pytest.fixture
    def risk_manager(self):
        """Create a test risk manager."""
        risk_limits = RiskLimits(max_portfolio_var=0.02, max_drawdown=0.10, max_leverage=1.0)
        return RiskManager(risk_limits)

    @pytest.fixture
    def alert_manager(self):
        """Create a test alert manager."""
        return AlertManager()

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return RiskAlertConfig(
            monitoring_interval_seconds=30,
            real_time_monitoring=True,
            alert_thresholds=[],
            circuit_breaker_rules=[],
            notifications=NotificationConfig(),
            escalation_enabled=True,
            escalation_timeout_minutes=30,
            audit_log_enabled=True,
            audit_log_path="logs/test_audit.log",
            report_generation_enabled=True,
            report_schedule_hours=24,
        )

    @pytest.fixture
    def risk_alert_system(self, risk_manager, alert_manager, config):
        """Create a test risk alert system."""
        return RiskAlertSystem(
            risk_manager=risk_manager,
            alert_manager=alert_manager,
            config=config,
            portfolio_id="test_portfolio",
        )

    @pytest.fixture
    def sample_metrics(self):
        """Create sample risk metrics."""
        return RiskMetrics(
            portfolio_var=0.03,
            portfolio_cvar=0.05,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.1,
            sortino_ratio=1.3,
            beta=1.0,
            correlation_risk=0.4,
            concentration_risk=0.2,
            timestamp=datetime.now(),
        )

    def test_initialization(self, risk_alert_system):
        """Test RiskAlertSystem initialization."""
        assert risk_alert_system.portfolio_id == "test_portfolio"
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL
        assert risk_alert_system.is_monitoring is False
        assert len(risk_alert_system.alert_thresholds) == 0
        assert len(risk_alert_system.circuit_breaker_rules) == 0

    def test_initialize_thresholds(self, risk_alert_system):
        """Test threshold initialization."""
        # Add some thresholds to config
        risk_alert_system.config.alert_thresholds = [
            {
                "metric_name": "portfolio_var",
                "threshold_type": "max",
                "threshold_value": 0.05,
                "severity": "warning",
                "escalation_level": "level_2",
                "cooldown_minutes": 30,
                "enabled": True,
                "description": "Test threshold",
            }
        ]

        risk_alert_system._initialize_thresholds()

        assert len(risk_alert_system.alert_thresholds) == 1
        assert "portfolio_var" in risk_alert_system.alert_thresholds
        threshold = risk_alert_system.alert_thresholds["portfolio_var"]
        assert threshold.threshold_value == 0.05
        assert threshold.severity == AlertSeverity.WARNING

    def test_initialize_circuit_breakers(self, risk_alert_system):
        """Test circuit breaker initialization."""
        # Add some rules to config
        risk_alert_system.config.circuit_breaker_rules = [
            {
                "name": "test_rule",
                "trigger_condition": "var_exceeded",
                "threshold_value": 0.1,
                "action": "stop_trading",
                "cooldown_minutes": 60,
                "enabled": True,
                "description": "Test rule",
            }
        ]

        risk_alert_system._initialize_circuit_breakers()

        assert len(risk_alert_system.circuit_breaker_rules) == 1
        assert "test_rule" in risk_alert_system.circuit_breaker_rules
        rule = risk_alert_system.circuit_breaker_rules["test_rule"]
        assert rule.threshold_value == 0.1
        assert rule.action == "stop_trading"

    def test_get_metric_value(self, risk_alert_system, sample_metrics):
        """Test getting metric values."""
        # Test valid metrics
        assert risk_alert_system._get_metric_value(sample_metrics, "portfolio_var") == 0.03
        assert risk_alert_system._get_metric_value(sample_metrics, "current_drawdown") == 0.05
        assert risk_alert_system._get_metric_value(sample_metrics, "leverage") == 1.2

        # Test invalid metric
        assert risk_alert_system._get_metric_value(sample_metrics, "invalid_metric") is None

    def test_is_threshold_violated(self, risk_alert_system):
        """Test threshold violation checking."""
        # Test max threshold
        threshold = AlertThreshold(
            metric_name="test_metric",
            threshold_type="max",
            threshold_value=0.05,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2,
        )

        assert risk_alert_system._is_threshold_violated(0.06, threshold) is True
        assert risk_alert_system._is_threshold_violated(0.04, threshold) is False

        # Test min threshold
        threshold.threshold_type = "min"
        assert risk_alert_system._is_threshold_violated(0.04, threshold) is True
        assert risk_alert_system._is_threshold_violated(0.06, threshold) is False

    @pytest.mark.asyncio
    async def test_trigger_alert(self, risk_alert_system, sample_metrics):
        """Test alert triggering."""
        threshold = AlertThreshold(
            metric_name="portfolio_var",
            threshold_type="max",
            threshold_value=0.05,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2,
        )

        # Mock notification methods
        risk_alert_system._send_notifications = AsyncMock()
        risk_alert_system._trigger_escalation = AsyncMock()

        await risk_alert_system._trigger_alert(threshold, 0.06, sample_metrics)

        # Check that alert was created
        assert len(risk_alert_system.alert_manager.alerts) == 1
        alert = risk_alert_system.alert_manager.alerts[0]
        assert alert.title == "Risk Threshold Violation: portfolio_var"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.source == "risk_alert_system"

        # Check that notifications and escalation were called
        risk_alert_system._send_notifications.assert_called_once()
        risk_alert_system._trigger_escalation.assert_called_once()

    def test_should_trigger_circuit_breaker(self, risk_alert_system, sample_metrics):
        """Test circuit breaker trigger conditions."""
        # Test VaR circuit breaker
        rule = CircuitBreakerRule(
            name="var_breaker",
            trigger_condition="var_exceeded",
            threshold_value=0.05,
            action="stop_trading",
        )

        # Should trigger (VaR = 0.03 < 0.05, so no trigger)
        assert risk_alert_system._should_trigger_circuit_breaker(rule, sample_metrics) is False

        # Modify metrics to trigger
        sample_metrics.portfolio_var = 0.06
        assert risk_alert_system._should_trigger_circuit_breaker(rule, sample_metrics) is True

        # Test drawdown circuit breaker
        rule.trigger_condition = "drawdown_exceeded"
        rule.threshold_value = 0.10

        # Should trigger (drawdown = 0.05 < 0.10, so no trigger)
        sample_metrics.portfolio_var = 0.03  # Reset
        assert risk_alert_system._should_trigger_circuit_breaker(rule, sample_metrics) is False

        # Modify metrics to trigger
        sample_metrics.current_drawdown = 0.15
        assert risk_alert_system._should_trigger_circuit_breaker(rule, sample_metrics) is True

    @pytest.mark.asyncio
    async def test_trigger_circuit_breaker(self, risk_alert_system, sample_metrics):
        """Test circuit breaker triggering."""
        rule = CircuitBreakerRule(
            name="test_breaker",
            trigger_condition="var_exceeded",
            threshold_value=0.05,
            action="stop_trading",
        )

        # Mock methods
        risk_alert_system._execute_circuit_breaker_action = AsyncMock()
        risk_alert_system._send_notifications = AsyncMock()

        # Modify metrics to trigger
        sample_metrics.portfolio_var = 0.06

        await risk_alert_system._trigger_circuit_breaker(rule, sample_metrics)

        # Check that alert was created
        assert len(risk_alert_system.alert_manager.alerts) == 1
        alert = risk_alert_system.alert_manager.alerts[0]
        assert alert.title == "Circuit Breaker Triggered: test_breaker"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.alert_type == "circuit_breaker"

        # Check that action and notifications were called
        risk_alert_system._execute_circuit_breaker_action.assert_called_once_with(rule)
        risk_alert_system._send_notifications.assert_called_once()

    def test_update_circuit_breaker_status(self, risk_alert_system, sample_metrics):
        """Test circuit breaker status updates."""
        # Normal status
        risk_alert_system._update_circuit_breaker_status(sample_metrics)
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL

        # Warning status (drawdown > 80% of limit)
        sample_metrics.current_drawdown = 0.09  # 90% of 0.10 limit
        risk_alert_system._update_circuit_breaker_status(sample_metrics)
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.WARNING

        # Alert status (VaR > limit)
        sample_metrics.portfolio_var = 0.025  # > 0.02 limit
        risk_alert_system._update_circuit_breaker_status(sample_metrics)
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.ALERT

        # Critical status (VaR > 1.5x limit)
        sample_metrics.portfolio_var = 0.04  # > 1.5 * 0.02
        risk_alert_system._update_circuit_breaker_status(sample_metrics)
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.CRITICAL

    @pytest.mark.asyncio
    async def test_trigger_escalation(self, risk_alert_system):
        """Test escalation procedures."""
        alert = risk_alert_system.alert_manager.create_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.WARNING,
            source="test",
            alert_type="test",
        )

        # Mock notification methods
        risk_alert_system._send_email_notification = AsyncMock()
        risk_alert_system._send_slack_notification = AsyncMock()
        risk_alert_system._send_sms_notification = AsyncMock()
        risk_alert_system._emergency_shutdown = AsyncMock()

        # Test Level 2 escalation (email)
        await risk_alert_system._trigger_escalation(EscalationLevel.LEVEL_2, alert)
        risk_alert_system._send_email_notification.assert_called_once()

        # Test Level 3 escalation (Slack/SMS)
        await risk_alert_system._trigger_escalation(EscalationLevel.LEVEL_3, alert)
        risk_alert_system._send_slack_notification.assert_called_once()
        risk_alert_system._send_sms_notification.assert_called_once()

        # Test Level 5 escalation (emergency shutdown)
        await risk_alert_system._trigger_escalation(EscalationLevel.LEVEL_5, alert)
        risk_alert_system._emergency_shutdown.assert_called_once()

        # Check escalation history
        assert len(risk_alert_system.escalation_history) == 3

    @pytest.mark.asyncio
    async def test_emergency_shutdown(self, risk_alert_system):
        """Test emergency shutdown procedures."""
        alert = risk_alert_system.alert_manager.create_alert(
            title="Test Alert",
            message="Test message",
            severity=AlertSeverity.CRITICAL,
            source="test",
            alert_type="test",
        )

        # Mock methods
        risk_alert_system._send_notifications = AsyncMock()

        await risk_alert_system._emergency_shutdown(alert)

        # Check that monitoring was stopped
        assert risk_alert_system.is_monitoring is False
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.TRADING_SUSPENDED

        # Check that emergency alert was created
        assert len(risk_alert_system.alert_manager.alerts) == 2
        emergency_alert = risk_alert_system.alert_manager.alerts[1]
        assert emergency_alert.title == "EMERGENCY SHUTDOWN"
        assert emergency_alert.severity == AlertSeverity.CRITICAL

        # Check that notifications were sent
        risk_alert_system._send_notifications.assert_called_once()

    def test_log_audit_entry(self, risk_alert_system, tmp_path):
        """Test audit logging."""
        # Set audit log path to temp directory
        audit_log_path = tmp_path / "test_audit.log"
        risk_alert_system.config.audit_log_path = str(audit_log_path)

        # Log an entry
        test_data = {"test": "data", "value": 123}
        risk_alert_system._log_audit_entry("test_event", test_data)

        # Check that file was created and contains the entry
        assert audit_log_path.exists()

        with open(audit_log_path) as f:
            lines = f.readlines()
            assert len(lines) == 1

            entry = json.loads(lines[0])
            assert entry["event_type"] == "test_event"
            assert entry["portfolio_id"] == "test_portfolio"
            assert entry["data"] == test_data

    def test_metrics_to_dict(self, risk_alert_system, sample_metrics):
        """Test metrics to dictionary conversion."""
        metrics_dict = risk_alert_system._metrics_to_dict(sample_metrics)

        assert metrics_dict["portfolio_var"] == 0.03
        assert metrics_dict["current_drawdown"] == 0.05
        assert metrics_dict["leverage"] == 1.2
        assert "timestamp" in metrics_dict

    def test_get_risk_summary(self, risk_alert_system, sample_metrics):
        """Test risk summary generation."""
        # Set current metrics
        risk_alert_system.risk_manager.current_metrics = sample_metrics

        summary = risk_alert_system.get_risk_summary()

        assert summary["portfolio_id"] == "test_portfolio"
        assert summary["circuit_breaker_status"] == CircuitBreakerStatus.NORMAL.value
        assert summary["monitoring_active"] is False
        assert summary["current_metrics"] is not None
        assert summary["active_alerts"] == 0
        assert summary["total_alerts"] == 0

    def test_get_current_escalation_level(self, risk_alert_system):
        """Test current escalation level determination."""
        # No alerts - should be Level 1
        assert risk_alert_system._get_current_escalation_level() == EscalationLevel.LEVEL_1.value

        # Add error alert
        alert = risk_alert_system.alert_manager.create_alert(
            title="Error Alert",
            message="Test error",
            severity=AlertSeverity.ERROR,
            source="test",
            alert_type="test",
        )
        risk_alert_system.alert_history.append(alert)

        assert risk_alert_system._get_current_escalation_level() == EscalationLevel.LEVEL_4.value

        # Add critical alert
        critical_alert = risk_alert_system.alert_manager.create_alert(
            title="Critical Alert",
            message="Test critical",
            severity=AlertSeverity.CRITICAL,
            source="test",
            alert_type="test",
        )
        risk_alert_system.alert_history.append(critical_alert)

        assert risk_alert_system._get_current_escalation_level() == EscalationLevel.LEVEL_5.value

    def test_generate_risk_report(self, risk_alert_system, _sample_metrics):
        """Test risk report generation."""
        # Add some sample data
        for i in range(5):
            metrics = RiskMetrics(
                portfolio_var=0.02 + i * 0.01,
                portfolio_cvar=0.03 + i * 0.01,
                max_drawdown=0.05 + i * 0.02,
                current_drawdown=0.03 + i * 0.01,
                leverage=1.0 + i * 0.1,
                sharpe_ratio=1.0 + i * 0.1,
                sortino_ratio=1.2 + i * 0.1,
                beta=1.0 + i * 0.1,
                correlation_risk=0.3 + i * 0.1,
                concentration_risk=0.2 + i * 0.05,
                timestamp=datetime.now() - timedelta(hours=i),
            )
            risk_alert_system.risk_history.append(metrics)

        # Add some alerts
        for i in range(3):
            alert = risk_alert_system.alert_manager.create_alert(
                title=f"Alert {i}",
                message=f"Test alert {i}",
                severity=AlertSeverity.WARNING if i % 2 == 0 else AlertSeverity.ERROR,
                source="test",
                alert_type="test",
            )
            risk_alert_system.alert_history.append(alert)

        # Generate report
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        report = risk_alert_system.generate_risk_report(start_time, end_time)

        # Check report structure
        assert report["portfolio_id"] == "test_portfolio"
        assert "report_period" in report
        assert "alert_statistics" in report
        assert "risk_statistics" in report
        assert "circuit_breaker_status" in report
        assert "escalation_summary" in report
        assert "recommendations" in report

        # Check alert statistics
        alert_stats = report["alert_statistics"]
        assert alert_stats["total_alerts"] == 3
        assert alert_stats["warning_alerts"] == 2
        assert alert_stats["error_alerts"] == 1

        # Check risk statistics
        risk_stats = report["risk_statistics"]
        assert "avg_var" in risk_stats
        assert "max_var" in risk_stats
        assert "avg_drawdown" in risk_stats
        assert "max_drawdown" in risk_stats

    def test_add_remove_thresholds(self, risk_alert_system):
        """Test adding and removing alert thresholds."""
        threshold = AlertThreshold(
            metric_name="test_metric",
            threshold_type="max",
            threshold_value=0.1,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2,
        )

        # Add threshold
        risk_alert_system.add_alert_threshold(threshold)
        assert "test_metric" in risk_alert_system.alert_thresholds
        assert len(risk_alert_system.alert_thresholds) == 1

        # Remove threshold
        risk_alert_system.remove_alert_threshold("test_metric")
        assert "test_metric" not in risk_alert_system.alert_thresholds
        assert len(risk_alert_system.alert_thresholds) == 0

    def test_add_remove_circuit_breakers(self, risk_alert_system):
        """Test adding and removing circuit breaker rules."""
        rule = CircuitBreakerRule(
            name="test_rule",
            trigger_condition="var_exceeded",
            threshold_value=0.1,
            action="stop_trading",
        )

        # Add rule
        risk_alert_system.add_circuit_breaker_rule(rule)
        assert "test_rule" in risk_alert_system.circuit_breaker_rules
        assert len(risk_alert_system.circuit_breaker_rules) == 1

        # Remove rule
        risk_alert_system.remove_circuit_breaker_rule("test_rule")
        assert "test_rule" not in risk_alert_system.circuit_breaker_rules
        assert len(risk_alert_system.circuit_breaker_rules) == 0

    def test_reset_circuit_breaker(self, risk_alert_system):
        """Test circuit breaker reset."""
        # Set to critical status
        risk_alert_system.circuit_breaker_status = CircuitBreakerStatus.CRITICAL

        # Reset
        risk_alert_system.reset_circuit_breaker()

        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL

    def test_get_alert_history(self, risk_alert_system):
        """Test getting alert history."""
        # Add some alerts
        for i in range(5):
            alert = risk_alert_system.alert_manager.create_alert(
                title=f"Alert {i}",
                message=f"Test alert {i}",
                severity=AlertSeverity.INFO,
                source="test",
                alert_type="test",
            )
            risk_alert_system.alert_history.append(alert)

        # Get all history
        all_history = risk_alert_system.get_alert_history()
        assert len(all_history) == 5

        # Get limited history
        limited_history = risk_alert_system.get_alert_history(limit=3)
        assert len(limited_history) == 3
        assert limited_history[-1].title == "Alert 4"

    def test_get_risk_history(self, risk_alert_system, _sample_metrics):
        """Test getting risk history."""
        # Add some metrics
        for i in range(5):
            metrics = RiskMetrics(
                portfolio_var=0.02 + i * 0.01,
                portfolio_cvar=0.03 + i * 0.01,
                max_drawdown=0.05 + i * 0.02,
                current_drawdown=0.03 + i * 0.01,
                leverage=1.0 + i * 0.1,
                sharpe_ratio=1.0 + i * 0.1,
                sortino_ratio=1.2 + i * 0.1,
                beta=1.0 + i * 0.1,
                correlation_risk=0.3 + i * 0.1,
                concentration_risk=0.2 + i * 0.05,
                timestamp=datetime.now() - timedelta(hours=i),
            )
            risk_alert_system.risk_history.append(metrics)

        # Get all history
        all_history = risk_alert_system.get_risk_history()
        assert len(all_history) == 5

        # Get limited history
        limited_history = risk_alert_system.get_risk_history(limit=3)
        assert len(limited_history) == 3
        assert limited_history[-1].portfolio_var == 0.06  # 0.02 + 4 * 0.01


class TestIntegration:
    """Integration tests for the complete risk alert system."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self):
        """Test complete risk alert workflow."""
        # Create components
        risk_manager = RiskManager()
        alert_manager = AlertManager()

        config = RiskAlertConfig(
            monitoring_interval_seconds=30,
            real_time_monitoring=True,
            alert_thresholds=[
                {
                    "metric_name": "portfolio_var",
                    "threshold_type": "max",
                    "threshold_value": 0.05,
                    "severity": "warning",
                    "escalation_level": "level_2",
                    "cooldown_minutes": 30,
                    "enabled": True,
                    "description": "Test threshold",
                }
            ],
            circuit_breaker_rules=[
                {
                    "name": "test_breaker",
                    "trigger_condition": "var_exceeded",
                    "threshold_value": 0.1,
                    "action": "stop_trading",
                    "cooldown_minutes": 60,
                    "enabled": True,
                    "description": "Test circuit breaker",
                }
            ],
            notifications=NotificationConfig(),
            escalation_enabled=True,
            escalation_timeout_minutes=30,
            audit_log_enabled=True,
            audit_log_path="logs/test_integration.log",
            report_generation_enabled=True,
            report_schedule_hours=24,
        )

        risk_alert_system = RiskAlertSystem(
            risk_manager=risk_manager,
            alert_manager=alert_manager,
            config=config,
            portfolio_id="integration_test",
        )

        # Mock notification methods
        risk_alert_system._send_notifications = AsyncMock()
        risk_alert_system._trigger_escalation = AsyncMock()
        risk_alert_system._execute_circuit_breaker_action = AsyncMock()

        # Test normal metrics (no alerts)
        normal_metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.05,
            current_drawdown=0.03,
            leverage=1.1,
            sharpe_ratio=1.2,
            sortino_ratio=1.4,
            beta=1.0,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now(),
        )

        risk_manager.current_metrics = normal_metrics
        await risk_alert_system._check_risk_metrics()

        # Should not trigger any alerts
        assert len(alert_manager.alerts) == 0
        assert risk_alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL

        # Test threshold violation
        high_var_metrics = RiskMetrics(
            portfolio_var=0.06,  # Triggers warning
            portfolio_cvar=0.08,
            max_drawdown=0.05,
            current_drawdown=0.03,
            leverage=1.1,
            sharpe_ratio=1.2,
            sortino_ratio=1.4,
            beta=1.0,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now(),
        )

        risk_manager.current_metrics = high_var_metrics
        await risk_alert_system._check_risk_metrics()

        # Should trigger alert
        assert len(alert_manager.alerts) == 1
        alert = alert_manager.alerts[0]
        assert alert.title == "Risk Threshold Violation: portfolio_var"
        assert alert.severity == AlertSeverity.WARNING

        # Test circuit breaker violation
        critical_metrics = RiskMetrics(
            portfolio_var=0.12,  # Triggers circuit breaker
            portfolio_cvar=0.15,
            max_drawdown=0.05,
            current_drawdown=0.03,
            leverage=1.1,
            sharpe_ratio=1.2,
            sortino_ratio=1.4,
            beta=1.0,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now(),
        )

        risk_manager.current_metrics = critical_metrics
        await risk_alert_system._check_risk_metrics()

        # Should trigger circuit breaker
        assert len(alert_manager.alerts) == 2
        circuit_breaker_alert = alert_manager.alerts[1]
        assert circuit_breaker_alert.title == "Circuit Breaker Triggered: test_breaker"
        assert circuit_breaker_alert.severity == AlertSeverity.CRITICAL

        # Check that notifications and actions were called
        assert risk_alert_system._send_notifications.call_count == 2
        risk_alert_system._execute_circuit_breaker_action.assert_called_once()

        # Generate summary
        summary = risk_alert_system.get_risk_summary()
        assert summary["total_alerts"] == 2
        assert summary["active_alerts"] == 2
        assert summary["circuit_breaker_status"] == CircuitBreakerStatus.CRITICAL.value


if __name__ == "__main__":
    pytest.main([__file__])
