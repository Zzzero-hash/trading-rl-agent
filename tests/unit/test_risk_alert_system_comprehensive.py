"""
Comprehensive tests for risk alert system and monitoring.

Tests cover:
- Real-time risk monitoring
- Circuit breaker functionality
- Alert threshold management
- Escalation procedures
- Notification systems
- Regulatory compliance checks
- Performance benchmarks
"""

import asyncio
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.trading_rl_agent.risk.alert_system import (
    RiskAlertSystem, RiskAlertConfig, AlertThreshold, CircuitBreakerRule,
    CircuitBreakerStatus, EscalationLevel, NotificationConfig
)
from src.trading_rl_agent.risk.manager import RiskManager, RiskLimits, RiskMetrics
from src.trading_rl_agent.monitoring.alert_manager import AlertManager, AlertSeverity


class TestRiskAlertSystem:
    """Test risk alert system functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        # Create test configuration
        self.config = RiskAlertConfig(
            monitoring_interval_seconds=1,
            real_time_monitoring=True,
            alert_thresholds=[
                {
                    "metric_name": "portfolio_var",
                    "threshold_type": "max",
                    "threshold_value": 0.03,
                    "severity": "warning",
                    "escalation_level": "level_2",
                    "cooldown_minutes": 30,
                    "enabled": True,
                    "description": "Portfolio VaR exceeds 3%"
                },
                {
                    "metric_name": "max_drawdown",
                    "threshold_type": "max",
                    "threshold_value": 0.15,
                    "severity": "critical",
                    "escalation_level": "level_4",
                    "cooldown_minutes": 15,
                    "enabled": True,
                    "description": "Maximum drawdown exceeds 15%"
                }
            ],
            circuit_breaker_rules=[
                {
                    "name": "var_circuit_breaker",
                    "trigger_condition": "var_exceeded",
                    "threshold_value": 0.05,
                    "action": "stop_trading",
                    "cooldown_minutes": 60,
                    "enabled": True,
                    "description": "Stop trading if VaR exceeds 5%"
                }
            ],
            notifications=NotificationConfig(
                email_enabled=False,
                slack_enabled=False,
                sms_enabled=False,
                webhook_enabled=False
            ),
            escalation_enabled=True,
            escalation_timeout_minutes=30,
            audit_log_enabled=True,
            audit_log_path="test_audit.log",
            report_generation_enabled=True,
            report_schedule_hours=24
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config,
            portfolio_id="test_portfolio"
        )

    def test_alert_system_initialization(self):
        """Test risk alert system initialization."""
        assert self.alert_system.risk_manager == self.risk_manager
        assert self.alert_system.alert_manager == self.alert_manager
        assert self.alert_system.config == self.config
        assert self.alert_system.portfolio_id == "test_portfolio"
        assert self.alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL
        assert not self.alert_system.is_monitoring

    def test_alert_threshold_initialization(self):
        """Test alert threshold initialization."""
        assert len(self.alert_system.alert_thresholds) == 2
        assert "portfolio_var" in self.alert_system.alert_thresholds
        assert "max_drawdown" in self.alert_system.alert_thresholds
        
        var_threshold = self.alert_system.alert_thresholds["portfolio_var"]
        assert var_threshold.metric_name == "portfolio_var"
        assert var_threshold.threshold_value == 0.03
        assert var_threshold.severity == AlertSeverity.WARNING

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        assert len(self.alert_system.circuit_breaker_rules) == 1
        assert "var_circuit_breaker" in self.alert_system.circuit_breaker_rules
        
        var_rule = self.alert_system.circuit_breaker_rules["var_circuit_breaker"]
        assert var_rule.name == "var_circuit_breaker"
        assert var_rule.threshold_value == 0.05
        assert var_rule.action == "stop_trading"

    def test_metric_value_extraction(self):
        """Test metric value extraction from risk metrics."""
        # Create test risk metrics
        metrics = RiskMetrics(
            portfolio_var=0.025,
            portfolio_cvar=0.035,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Test metric extraction
        var_value = self.alert_system._get_metric_value(metrics, "portfolio_var")
        drawdown_value = self.alert_system._get_metric_value(metrics, "max_drawdown")
        invalid_value = self.alert_system._get_metric_value(metrics, "invalid_metric")
        
        assert var_value == 0.025
        assert drawdown_value == 0.08
        assert invalid_value is None

    def test_threshold_violation_detection(self):
        """Test threshold violation detection."""
        # Test max threshold
        threshold = AlertThreshold(
            metric_name="portfolio_var",
            threshold_type="max",
            threshold_value=0.03,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2
        )
        
        # Value below threshold
        assert not self.alert_system._is_threshold_violated(0.02, threshold)
        
        # Value at threshold
        assert not self.alert_system._is_threshold_violated(0.03, threshold)
        
        # Value above threshold
        assert self.alert_system._is_threshold_violated(0.04, threshold)
        
        # Test min threshold
        min_threshold = AlertThreshold(
            metric_name="sharpe_ratio",
            threshold_type="min",
            threshold_value=1.0,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2
        )
        
        # Value above threshold
        assert not self.alert_system._is_threshold_violated(1.5, min_threshold)
        
        # Value below threshold
        assert self.alert_system._is_threshold_violated(0.5, min_threshold)

    @pytest.mark.asyncio
    async def test_alert_triggering(self):
        """Test alert triggering functionality."""
        # Create test metrics that violate threshold
        metrics = RiskMetrics(
            portfolio_var=0.04,  # Above 0.03 threshold
            portfolio_cvar=0.05,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        threshold = self.alert_system.alert_thresholds["portfolio_var"]
        
        # Mock notification methods
        with patch.object(self.alert_system, '_send_notifications', new_callable=AsyncMock):
            await self.alert_system._trigger_alert(threshold, 0.04, metrics)
        
        # Check that alert was created
        assert len(self.alert_system.alert_history) > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggering(self):
        """Test circuit breaker triggering."""
        # Create test metrics that trigger circuit breaker
        metrics = RiskMetrics(
            portfolio_var=0.06,  # Above 0.05 circuit breaker threshold
            portfolio_cvar=0.08,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Mock circuit breaker action
        with patch.object(self.alert_system, '_execute_circuit_breaker_action', new_callable=AsyncMock):
            await self.alert_system._check_circuit_breakers(metrics)
        
        # Check circuit breaker status
        assert self.alert_system.circuit_breaker_status != CircuitBreakerStatus.NORMAL

    def test_circuit_breaker_status_update(self):
        """Test circuit breaker status update."""
        # Test normal status
        metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.05,
            current_drawdown=0.02,
            leverage=1.0,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        self.alert_system._update_circuit_breaker_status(metrics)
        assert self.alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL

    @pytest.mark.asyncio
    async def test_escalation_procedure(self):
        """Test escalation procedure."""
        # Create test alert
        alert = Mock()
        alert.severity = AlertSeverity.CRITICAL
        alert.message = "Test critical alert"
        
        # Mock escalation methods
        with patch.object(self.alert_system, '_send_notifications', new_callable=AsyncMock):
            await self.alert_system._trigger_escalation(EscalationLevel.LEVEL_3, alert)
        
        # Check escalation history
        assert len(self.alert_system.escalation_history) > 0

    def test_audit_logging(self):
        """Test audit logging functionality."""
        # Test audit entry logging
        test_data = {"test_key": "test_value"}
        self.alert_system._log_audit_entry("test_event", test_data)
        
        # Check that audit entry was created
        # Note: In a real implementation, this would write to a file
        # For testing, we just verify the method doesn't raise an exception

    def test_risk_summary_generation(self):
        """Test risk summary generation."""
        summary = self.alert_system.get_risk_summary()
        
        assert isinstance(summary, dict)
        assert "circuit_breaker_status" in summary
        assert "alert_count" in summary
        assert "escalation_level" in summary
        assert "last_risk_check" in summary

    def test_risk_report_generation(self):
        """Test risk report generation."""
        # Add some test data
        self.alert_system.risk_history.append(RiskMetrics(
            portfolio_var=0.025,
            portfolio_cvar=0.035,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        ))
        
        report = self.alert_system.generate_risk_report()
        
        assert isinstance(report, dict)
        assert "risk_metrics_summary" in report
        assert "alert_summary" in report
        assert "escalation_summary" in report
        assert "recommendations" in report


class TestRealTimeMonitoring:
    """Test real-time monitoring functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        self.config = RiskAlertConfig(
            monitoring_interval_seconds=1,
            real_time_monitoring=True
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config
        )

    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self):
        """Test monitoring start and stop."""
        # Start monitoring
        await self.alert_system.start_monitoring()
        assert self.alert_system.is_monitoring
        
        # Stop monitoring
        await self.alert_system.stop_monitoring()
        assert not self.alert_system.is_monitoring

    @pytest.mark.asyncio
    async def test_monitoring_loop(self):
        """Test monitoring loop functionality."""
        # Mock the risk check method
        with patch.object(self.alert_system, '_check_risk_metrics', new_callable=AsyncMock):
            # Start monitoring for a short duration
            monitoring_task = asyncio.create_task(self.alert_system.start_monitoring())
            
            # Wait a bit for monitoring to run
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            await self.alert_system.stop_monitoring()
            
            # Wait for task to complete
            try:
                await asyncio.wait_for(monitoring_task, timeout=1.0)
            except asyncio.TimeoutError:
                monitoring_task.cancel()
                await monitoring_task

    @pytest.mark.asyncio
    async def test_risk_metrics_checking(self):
        """Test risk metrics checking in monitoring loop."""
        # Mock risk manager to return test metrics
        test_metrics = RiskMetrics(
            portfolio_var=0.04,  # Above threshold
            portfolio_cvar=0.05,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        with patch.object(self.risk_manager, 'generate_risk_report', return_value={
            'portfolio_var': 0.04,
            'max_drawdown': 0.08
        }):
            with patch.object(self.alert_system, '_check_alert_thresholds', new_callable=AsyncMock) as mock_check:
                await self.alert_system._check_risk_metrics()
                
                # Verify that alert thresholds were checked
                mock_check.assert_called_once()


class TestCircuitBreakers:
    """Test circuit breaker functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        self.config = RiskAlertConfig(
            circuit_breaker_rules=[
                {
                    "name": "var_breaker",
                    "trigger_condition": "var_exceeded",
                    "threshold_value": 0.05,
                    "action": "stop_trading",
                    "enabled": True
                },
                {
                    "name": "drawdown_breaker",
                    "trigger_condition": "drawdown_exceeded",
                    "threshold_value": 0.20,
                    "action": "reduce_position",
                    "enabled": True
                }
            ]
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config
        )

    def test_circuit_breaker_trigger_conditions(self):
        """Test circuit breaker trigger conditions."""
        # Test VaR exceeded condition
        var_rule = self.alert_system.circuit_breaker_rules["var_breaker"]
        
        metrics_normal = RiskMetrics(
            portfolio_var=0.03,  # Below threshold
            portfolio_cvar=0.04,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        metrics_breach = RiskMetrics(
            portfolio_var=0.06,  # Above threshold
            portfolio_cvar=0.08,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Should not trigger under normal conditions
        assert not self.alert_system._should_trigger_circuit_breaker(var_rule, metrics_normal)
        
        # Should trigger under breach conditions
        assert self.alert_system._should_trigger_circuit_breaker(var_rule, metrics_breach)

    @pytest.mark.asyncio
    async def test_circuit_breaker_actions(self):
        """Test circuit breaker actions."""
        rule = CircuitBreakerRule(
            name="test_breaker",
            trigger_condition="var_exceeded",
            threshold_value=0.05,
            action="stop_trading",
            enabled=True
        )
        
        # Mock action execution
        with patch.object(self.alert_system, '_emergency_shutdown', new_callable=AsyncMock) as mock_shutdown:
            await self.alert_system._execute_circuit_breaker_action(rule)
            
            # Verify appropriate action was taken
            if rule.action == "stop_trading":
                mock_shutdown.assert_called_once()

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset functionality."""
        # Set circuit breaker to non-normal status
        self.alert_system.circuit_breaker_status = CircuitBreakerStatus.ALERT
        
        # Reset circuit breaker
        self.alert_system.reset_circuit_breaker()
        
        # Should be back to normal
        assert self.alert_system.circuit_breaker_status == CircuitBreakerStatus.NORMAL


class TestNotificationSystems:
    """Test notification system functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        self.config = RiskAlertConfig(
            notifications=NotificationConfig(
                email_enabled=True,
                email_recipients=["test@example.com"],
                slack_enabled=True,
                slack_webhook_url="https://hooks.slack.com/test",
                sms_enabled=True,
                sms_recipients=["+1234567890"],
                webhook_enabled=True,
                webhook_url="https://api.example.com/webhook"
            )
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config
        )

    @pytest.mark.asyncio
    async def test_email_notification(self):
        """Test email notification functionality."""
        alert = Mock()
        alert.severity = AlertSeverity.WARNING
        alert.message = "Test warning alert"
        alert.timestamp = datetime.now()
        
        # Mock email sending
        with patch('smtplib.SMTP') as mock_smtp:
            mock_smtp.return_value.__enter__.return_value = mock_smtp.return_value
            await self.alert_system._send_email_notification(alert, priority=False)
            
            # Verify SMTP was called
            mock_smtp.assert_called_once()

    @pytest.mark.asyncio
    async def test_slack_notification(self):
        """Test Slack notification functionality."""
        alert = Mock()
        alert.severity = AlertSeverity.CRITICAL
        alert.message = "Test critical alert"
        alert.timestamp = datetime.now()
        
        # Mock webhook call
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = AsyncMock()
            await self.alert_system._send_slack_notification(alert, priority=True)
            
            # Verify webhook was called
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_sms_notification(self):
        """Test SMS notification functionality."""
        alert = Mock()
        alert.severity = AlertSeverity.CRITICAL
        alert.message = "Test critical alert"
        alert.timestamp = datetime.now()
        
        # Mock SMS sending
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = AsyncMock()
            await self.alert_system._send_sms_notification(alert, priority=True)
            
            # Verify SMS was sent
            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_notification(self):
        """Test webhook notification functionality."""
        alert = Mock()
        alert.severity = AlertSeverity.INFO
        alert.message = "Test info alert"
        alert.timestamp = datetime.now()
        
        # Mock webhook call
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = AsyncMock()
            await self.alert_system._send_webhook_notification(alert, priority=False)
            
            # Verify webhook was called
            mock_post.assert_called_once()


class TestRegulatoryCompliance:
    """Test regulatory compliance checks."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        self.config = RiskAlertConfig(
            alert_thresholds=[
                {
                    "metric_name": "leverage",
                    "threshold_type": "max",
                    "threshold_value": 2.0,  # Regulatory leverage limit
                    "severity": "critical",
                    "escalation_level": "level_5",
                    "enabled": True,
                    "description": "Regulatory leverage limit exceeded"
                },
                {
                    "metric_name": "concentration_risk",
                    "threshold_type": "max",
                    "threshold_value": 0.25,  # Regulatory concentration limit
                    "severity": "warning",
                    "escalation_level": "level_3",
                    "enabled": True,
                    "description": "Regulatory concentration limit exceeded"
                }
            ]
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config
        )

    def test_leverage_compliance_check(self):
        """Test leverage compliance checking."""
        # Test compliant leverage
        compliant_metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.5,  # Below 2.0 limit
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Test non-compliant leverage
        non_compliant_metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=2.5,  # Above 2.0 limit
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Check compliance
        compliant_threshold = self.alert_system.alert_thresholds["leverage"]
        assert not self.alert_system._is_threshold_violated(
            compliant_metrics.leverage, compliant_threshold
        )
        assert self.alert_system._is_threshold_violated(
            non_compliant_metrics.leverage, compliant_threshold
        )

    def test_concentration_compliance_check(self):
        """Test concentration compliance checking."""
        # Test compliant concentration
        compliant_metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.15,  # Below 0.25 limit
            timestamp=datetime.now()
        )
        
        # Test non-compliant concentration
        non_compliant_metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.30,  # Above 0.25 limit
            timestamp=datetime.now()
        )
        
        # Check compliance
        concentration_threshold = self.alert_system.alert_thresholds["concentration_risk"]
        assert not self.alert_system._is_threshold_violated(
            compliant_metrics.concentration_risk, concentration_threshold
        )
        assert self.alert_system._is_threshold_violated(
            non_compliant_metrics.concentration_risk, concentration_threshold
        )

    @pytest.mark.asyncio
    async def test_regulatory_escalation(self):
        """Test regulatory compliance escalation."""
        # Create metrics that violate regulatory limits
        non_compliant_metrics = RiskMetrics(
            portfolio_var=0.02,
            portfolio_cvar=0.03,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=2.5,  # Above regulatory limit
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.30,  # Above regulatory limit
            timestamp=datetime.now()
        )
        
        # Mock escalation
        with patch.object(self.alert_system, '_trigger_escalation', new_callable=AsyncMock) as mock_escalation:
            await self.alert_system._check_alert_thresholds(non_compliant_metrics)
            
            # Verify escalation was triggered for regulatory violations
            assert mock_escalation.call_count >= 1


class TestRiskAlertPerformance:
    """Test risk alert system performance."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        self.config = RiskAlertConfig(
            monitoring_interval_seconds=1,
            real_time_monitoring=True
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config
        )

    def test_alert_processing_performance(self):
        """Test alert processing performance."""
        # Create multiple test alerts
        start_time = time.time()
        
        for i in range(100):
            alert = Mock()
            alert.severity = AlertSeverity.WARNING
            alert.message = f"Test alert {i}"
            alert.timestamp = datetime.now()
            
            # Process alert (without sending notifications)
            self.alert_system.alert_history.append(alert)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 1.0  # Should process 100 alerts within 1 second
        assert len(self.alert_system.alert_history) == 100

    def test_metrics_checking_performance(self):
        """Test metrics checking performance."""
        # Create test metrics
        metrics = RiskMetrics(
            portfolio_var=0.025,
            portfolio_cvar=0.035,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        start_time = time.time()
        
        # Check multiple thresholds
        for i in range(1000):
            for threshold in self.alert_system.alert_thresholds.values():
                self.alert_system._is_threshold_violated(
                    self.alert_system._get_metric_value(metrics, threshold.metric_name),
                    threshold
                )
        
        end_time = time.time()
        checking_time = end_time - start_time
        
        assert checking_time < 1.0  # Should check 1000 thresholds within 1 second

    @pytest.mark.asyncio
    async def test_monitoring_performance(self):
        """Test monitoring loop performance."""
        # Mock risk manager to return quickly
        with patch.object(self.risk_manager, 'generate_risk_report', return_value={
            'portfolio_var': 0.025,
            'max_drawdown': 0.08
        }):
            with patch.object(self.alert_system, '_check_alert_thresholds', new_callable=AsyncMock):
                start_time = time.time()
                
                # Run monitoring for a short duration
                monitoring_task = asyncio.create_task(self.alert_system.start_monitoring())
                await asyncio.sleep(0.1)
                await self.alert_system.stop_monitoring()
                
                end_time = time.time()
                monitoring_time = end_time - start_time
                
                # Clean up
                try:
                    await asyncio.wait_for(monitoring_task, timeout=1.0)
                except asyncio.TimeoutError:
                    monitoring_task.cancel()
                    await monitoring_task
                
                assert monitoring_time < 2.0  # Should complete monitoring cycle within 2 seconds


class TestRiskAlertErrorHandling:
    """Test risk alert system error handling."""

    def setup_method(self):
        """Setup test fixtures."""
        self.risk_manager = RiskManager()
        self.alert_manager = AlertManager()
        
        self.config = RiskAlertConfig(
            monitoring_interval_seconds=1,
            real_time_monitoring=True
        )
        
        self.alert_system = RiskAlertSystem(
            risk_manager=self.risk_manager,
            alert_manager=self.alert_manager,
            config=self.config
        )

    def test_invalid_threshold_configuration(self):
        """Test handling of invalid threshold configuration."""
        # Test with invalid threshold type
        invalid_threshold = AlertThreshold(
            metric_name="portfolio_var",
            threshold_type="invalid_type",
            threshold_value=0.03,
            severity=AlertSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_2
        )
        
        # Should handle gracefully
        result = self.alert_system._is_threshold_violated(0.04, invalid_threshold)
        assert result is False  # Should default to False for invalid types

    def test_missing_metric_handling(self):
        """Test handling of missing metrics."""
        # Create metrics without all fields
        partial_metrics = RiskMetrics(
            portfolio_var=0.025,
            portfolio_cvar=0.035,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Try to get non-existent metric
        value = self.alert_system._get_metric_value(partial_metrics, "non_existent_metric")
        assert value is None

    @pytest.mark.asyncio
    async def test_notification_failure_handling(self):
        """Test handling of notification failures."""
        alert = Mock()
        alert.severity = AlertSeverity.CRITICAL
        alert.message = "Test critical alert"
        alert.timestamp = datetime.now()
        
        # Mock notification failure
        with patch.object(self.alert_system, '_send_email_notification', side_effect=Exception("Email failed")):
            # Should handle notification failure gracefully
            await self.alert_system._send_notifications(alert, priority=True)
            
            # System should continue to function
            assert True  # No exception should be raised

    def test_circuit_breaker_failure_handling(self):
        """Test handling of circuit breaker failures."""
        # Test with invalid circuit breaker rule
        invalid_rule = CircuitBreakerRule(
            name="invalid_breaker",
            trigger_condition="invalid_condition",
            threshold_value=0.05,
            action="invalid_action",
            enabled=True
        )
        
        metrics = RiskMetrics(
            portfolio_var=0.025,
            portfolio_cvar=0.035,
            max_drawdown=0.08,
            current_drawdown=0.05,
            leverage=1.2,
            sharpe_ratio=1.5,
            sortino_ratio=1.8,
            beta=0.95,
            correlation_risk=0.3,
            concentration_risk=0.2,
            timestamp=datetime.now()
        )
        
        # Should handle invalid rule gracefully
        result = self.alert_system._should_trigger_circuit_breaker(invalid_rule, metrics)
        assert result is False  # Should default to False for invalid conditions


if __name__ == "__main__":
    pytest.main([__file__])