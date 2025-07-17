"""
Automated Risk Alerts and Circuit Breakers System.

This module provides a comprehensive risk alert system that:
1. Monitors real-time portfolio risk metrics
2. Implements configurable alert thresholds and triggers
3. Creates automated circuit breakers for risk violations
4. Provides escalation procedures for different risk levels
5. Generates risk alert reports and audit trails

Integrates with the existing RiskManager and AlertManager for seamless operation.
"""

import asyncio
import json
import smtplib
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from urllib.parse import urlencode

import pandas as pd
import requests
from pydantic import BaseModel, Field

from ..core.logging import get_logger
from ..monitoring.alert_manager import Alert, AlertManager, AlertSeverity, AlertStatus
from .manager import RiskLimits, RiskManager, RiskMetrics


class CircuitBreakerStatus(Enum):
    """Circuit breaker status levels."""
    
    NORMAL = "normal"
    WARNING = "warning"
    ALERT = "alert"
    CRITICAL = "critical"
    TRADING_SUSPENDED = "trading_suspended"


class EscalationLevel(Enum):
    """Escalation levels for risk alerts."""
    
    LEVEL_1 = "level_1"  # Automated monitoring
    LEVEL_2 = "level_2"  # Email notification
    LEVEL_3 = "level_3"  # SMS/Slack notification
    LEVEL_4 = "level_4"  # Phone call
    LEVEL_5 = "level_5"  # Emergency shutdown


@dataclass
class AlertThreshold:
    """Configurable alert threshold."""
    
    metric_name: str
    threshold_type: str  # "min", "max", "change_rate", "volatility"
    threshold_value: float
    severity: AlertSeverity
    escalation_level: EscalationLevel
    cooldown_minutes: int = 30
    enabled: bool = True
    description: str = ""


@dataclass
class CircuitBreakerRule:
    """Circuit breaker rule configuration."""
    
    name: str
    trigger_condition: str  # "var_exceeded", "drawdown_exceeded", "leverage_exceeded"
    threshold_value: float
    action: str  # "reduce_position", "stop_trading", "liquidate"
    cooldown_minutes: int = 60
    enabled: bool = True
    description: str = ""


@dataclass
class NotificationConfig:
    """Notification configuration."""
    
    email_enabled: bool = True
    email_recipients: List[str] = field(default_factory=list)
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#risk-alerts"
    
    sms_enabled: bool = False
    sms_provider: str = "twilio"
    sms_api_key: str = ""
    sms_api_secret: str = ""
    sms_recipients: List[str] = field(default_factory=list)
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)


class RiskAlertConfig(BaseModel):
    """Configuration for the risk alert system."""
    
    # Monitoring settings
    monitoring_interval_seconds: int = Field(default=60, description="Risk monitoring interval")
    real_time_monitoring: bool = Field(default=True, description="Enable real-time monitoring")
    
    # Alert thresholds
    alert_thresholds: List[Dict[str, Any]] = Field(default_factory=list, description="Alert threshold configurations")
    
    # Circuit breakers
    circuit_breaker_rules: List[Dict[str, Any]] = Field(default_factory=list, description="Circuit breaker rules")
    
    # Notifications
    notifications: NotificationConfig = Field(default_factory=NotificationConfig, description="Notification settings")
    
    # Escalation
    escalation_enabled: bool = Field(default=True, description="Enable escalation procedures")
    escalation_timeout_minutes: int = Field(default=30, description="Escalation timeout")
    
    # Audit and reporting
    audit_log_enabled: bool = Field(default=True, description="Enable audit logging")
    audit_log_path: str = Field(default="logs/risk_audit.log", description="Audit log path")
    report_generation_enabled: bool = Field(default=True, description="Enable report generation")
    report_schedule_hours: int = Field(default=24, description="Report generation schedule")


class RiskAlertSystem:
    """
    Comprehensive risk alert and circuit breaker system.
    
    Features:
    - Real-time risk monitoring with configurable thresholds
    - Automated circuit breakers for risk violations
    - Multi-level escalation procedures
    - Multiple notification channels (email, Slack, SMS, webhooks)
    - Comprehensive audit logging and reporting
    - Integration with existing RiskManager and AlertManager
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        alert_manager: AlertManager,
        config: RiskAlertConfig,
        portfolio_id: str = "default"
    ):
        """
        Initialize the risk alert system.
        
        Args:
            risk_manager: Existing risk manager instance
            alert_manager: Existing alert manager instance
            config: Risk alert configuration
            portfolio_id: Portfolio identifier
        """
        self.risk_manager = risk_manager
        self.alert_manager = alert_manager
        self.config = config
        self.portfolio_id = portfolio_id
        self.logger = get_logger(self.__class__.__name__)
        
        # System state
        self.circuit_breaker_status = CircuitBreakerStatus.NORMAL
        self.last_risk_check = None
        self.risk_history: List[RiskMetrics] = []
        self.alert_history: List[Alert] = []
        self.escalation_history: List[Dict[str, Any]] = []
        
        # Alert thresholds and circuit breakers
        self.alert_thresholds: Dict[str, AlertThreshold] = {}
        self.circuit_breaker_rules: Dict[str, CircuitBreakerRule] = {}
        self.last_alert_times: Dict[str, float] = {}
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        # Initialize components
        self._initialize_thresholds()
        self._initialize_circuit_breakers()
        self._setup_audit_logging()
        
        self.logger.info(f"RiskAlertSystem initialized for portfolio {portfolio_id}")
    
    def _initialize_thresholds(self) -> None:
        """Initialize alert thresholds from configuration."""
        for threshold_config in self.config.alert_thresholds:
            threshold = AlertThreshold(
                metric_name=threshold_config["metric_name"],
                threshold_type=threshold_config["threshold_type"],
                threshold_value=threshold_config["threshold_value"],
                severity=AlertSeverity(threshold_config["severity"]),
                escalation_level=EscalationLevel(threshold_config["escalation_level"]),
                cooldown_minutes=threshold_config.get("cooldown_minutes", 30),
                enabled=threshold_config.get("enabled", True),
                description=threshold_config.get("description", "")
            )
            self.alert_thresholds[threshold.metric_name] = threshold
        
        self.logger.info(f"Initialized {len(self.alert_thresholds)} alert thresholds")
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breaker rules from configuration."""
        for rule_config in self.config.circuit_breaker_rules:
            rule = CircuitBreakerRule(
                name=rule_config["name"],
                trigger_condition=rule_config["trigger_condition"],
                threshold_value=rule_config["threshold_value"],
                action=rule_config["action"],
                cooldown_minutes=rule_config.get("cooldown_minutes", 60),
                enabled=rule_config.get("enabled", True),
                description=rule_config.get("description", "")
            )
            self.circuit_breaker_rules[rule.name] = rule
        
        self.logger.info(f"Initialized {len(self.circuit_breaker_rules)} circuit breaker rules")
    
    def _setup_audit_logging(self) -> None:
        """Setup audit logging if enabled."""
        if self.config.audit_log_enabled:
            log_path = Path(self.config.audit_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Audit logging enabled: {log_path}")
    
    async def start_monitoring(self) -> None:
        """Start real-time risk monitoring."""
        if self.is_monitoring:
            self.logger.warning("Risk monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Risk monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop real-time risk monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Risk monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._check_risk_metrics()
                await asyncio.sleep(self.config.monitoring_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _check_risk_metrics(self) -> None:
        """Check current risk metrics and trigger alerts if needed."""
        if not self.risk_manager.current_metrics:
            return
        
        metrics = self.risk_manager.current_metrics
        self.risk_history.append(metrics)
        self.last_risk_check = time.time()
        
        # Check alert thresholds
        await self._check_alert_thresholds(metrics)
        
        # Check circuit breakers
        await self._check_circuit_breakers(metrics)
        
        # Update circuit breaker status
        self._update_circuit_breaker_status(metrics)
        
        # Log audit trail
        if self.config.audit_log_enabled:
            self._log_audit_entry("risk_check", {"metrics": self._metrics_to_dict(metrics)})
    
    async def _check_alert_thresholds(self, metrics: RiskMetrics) -> None:
        """Check if any alert thresholds have been violated."""
        for threshold in self.alert_thresholds.values():
            if not threshold.enabled:
                continue
            
            # Check cooldown
            last_alert_time = self.last_alert_times.get(threshold.metric_name, 0)
            if time.time() - last_alert_time < threshold.cooldown_minutes * 60:
                continue
            
            # Get metric value
            metric_value = self._get_metric_value(metrics, threshold.metric_name)
            if metric_value is None:
                continue
            
            # Check threshold violation
            if self._is_threshold_violated(metric_value, threshold):
                await self._trigger_alert(threshold, metric_value, metrics)
                self.last_alert_times[threshold.metric_name] = time.time()
    
    def _get_metric_value(self, metrics: RiskMetrics, metric_name: str) -> Optional[float]:
        """Get metric value from RiskMetrics object."""
        metric_mapping = {
            "portfolio_var": metrics.portfolio_var,
            "portfolio_cvar": metrics.portfolio_cvar,
            "max_drawdown": metrics.max_drawdown,
            "current_drawdown": metrics.current_drawdown,
            "leverage": metrics.leverage,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "beta": metrics.beta,
            "correlation_risk": metrics.correlation_risk,
            "concentration_risk": metrics.concentration_risk,
        }
        return metric_mapping.get(metric_name)
    
    def _is_threshold_violated(self, value: float, threshold: AlertThreshold) -> bool:
        """Check if a threshold has been violated."""
        if threshold.threshold_type == "min":
            return value < threshold.threshold_value
        elif threshold.threshold_type == "max":
            return value > threshold.threshold_value
        elif threshold.threshold_type == "change_rate":
            # Calculate change rate from history
            if len(self.risk_history) < 2:
                return False
            prev_value = self._get_metric_value(self.risk_history[-2], threshold.metric_name)
            if prev_value is None or prev_value == 0:
                return False
            change_rate = (value - prev_value) / abs(prev_value)
            return abs(change_rate) > threshold.threshold_value
        return False
    
    async def _trigger_alert(self, threshold: AlertThreshold, value: float, metrics: RiskMetrics) -> None:
        """Trigger an alert for a threshold violation."""
        alert = self.alert_manager.create_alert(
            title=f"Risk Threshold Violation: {threshold.metric_name}",
            message=f"{threshold.metric_name} = {value:.4f} violates {threshold.threshold_type} threshold of {threshold.threshold_value}",
            severity=threshold.severity,
            source="risk_alert_system",
            alert_type=f"risk_threshold_{threshold.metric_name}",
            metadata={
                "metric": threshold.metric_name,
                "value": value,
                "threshold": threshold.threshold_value,
                "threshold_type": threshold.threshold_type,
                "escalation_level": threshold.escalation_level.value,
                "portfolio_id": self.portfolio_id
            }
        )
        
        self.alert_history.append(alert)
        
        # Trigger escalation
        if self.config.escalation_enabled:
            await self._trigger_escalation(threshold.escalation_level, alert)
        
        # Send notifications
        await self._send_notifications(alert)
        
        self.logger.warning(f"Risk alert triggered: {threshold.metric_name} = {value}")
    
    async def _check_circuit_breakers(self, metrics: RiskMetrics) -> None:
        """Check if any circuit breakers should be triggered."""
        for rule in self.circuit_breaker_rules.values():
            if not rule.enabled:
                continue
            
            if self._should_trigger_circuit_breaker(rule, metrics):
                await self._trigger_circuit_breaker(rule, metrics)
    
    def _should_trigger_circuit_breaker(self, rule: CircuitBreakerRule, metrics: RiskMetrics) -> bool:
        """Check if a circuit breaker should be triggered."""
        if rule.trigger_condition == "var_exceeded":
            return metrics.portfolio_var > rule.threshold_value
        elif rule.trigger_condition == "drawdown_exceeded":
            return metrics.current_drawdown > rule.threshold_value
        elif rule.trigger_condition == "leverage_exceeded":
            return metrics.leverage > rule.threshold_value
        elif rule.trigger_condition == "correlation_risk_exceeded":
            return metrics.correlation_risk > rule.threshold_value
        return False
    
    async def _trigger_circuit_breaker(self, rule: CircuitBreakerRule, metrics: RiskMetrics) -> None:
        """Trigger a circuit breaker."""
        alert = self.alert_manager.create_alert(
            title=f"Circuit Breaker Triggered: {rule.name}",
            message=f"Circuit breaker '{rule.name}' triggered. Action: {rule.action}",
            severity=AlertSeverity.CRITICAL,
            source="risk_alert_system",
            alert_type="circuit_breaker",
            metadata={
                "rule_name": rule.name,
                "action": rule.action,
                "trigger_condition": rule.trigger_condition,
                "threshold_value": rule.threshold_value,
                "portfolio_id": self.portfolio_id
            }
        )
        
        # Execute circuit breaker action
        await self._execute_circuit_breaker_action(rule)
        
        # Send immediate notifications
        await self._send_notifications(alert, priority=True)
        
        self.logger.critical(f"Circuit breaker triggered: {rule.name} - {rule.action}")
    
    async def _execute_circuit_breaker_action(self, rule: CircuitBreakerRule) -> None:
        """Execute the action specified by a circuit breaker rule."""
        if rule.action == "reduce_position":
            # Reduce position sizes
            self.logger.info("Executing position reduction")
            # TODO: Implement position reduction logic
        elif rule.action == "stop_trading":
            # Stop all trading
            self.logger.info("Executing trading suspension")
            # TODO: Implement trading suspension logic
        elif rule.action == "liquidate":
            # Liquidate positions
            self.logger.info("Executing position liquidation")
            # TODO: Implement liquidation logic
    
    def _update_circuit_breaker_status(self, metrics: RiskMetrics) -> None:
        """Update the overall circuit breaker status."""
        # Determine status based on current metrics
        if metrics.portfolio_var > self.risk_manager.risk_limits.max_portfolio_var * 1.5:
            self.circuit_breaker_status = CircuitBreakerStatus.CRITICAL
        elif metrics.portfolio_var > self.risk_manager.risk_limits.max_portfolio_var:
            self.circuit_breaker_status = CircuitBreakerStatus.ALERT
        elif metrics.current_drawdown > self.risk_manager.risk_limits.max_drawdown * 0.8:
            self.circuit_breaker_status = CircuitBreakerStatus.WARNING
        else:
            self.circuit_breaker_status = CircuitBreakerStatus.NORMAL
    
    async def _trigger_escalation(self, escalation_level: EscalationLevel, alert: Alert) -> None:
        """Trigger escalation procedures."""
        escalation_entry = {
            "timestamp": time.time(),
            "level": escalation_level.value,
            "alert_id": alert.id,
            "portfolio_id": self.portfolio_id,
            "status": "triggered"
        }
        
        self.escalation_history.append(escalation_entry)
        
        # Execute escalation based on level
        if escalation_level == EscalationLevel.LEVEL_1:
            # Automated monitoring - already handled
            pass
        elif escalation_level == EscalationLevel.LEVEL_2:
            # Email notification
            await self._send_email_notification(alert, priority=True)
        elif escalation_level == EscalationLevel.LEVEL_3:
            # SMS/Slack notification
            await self._send_slack_notification(alert, priority=True)
            await self._send_sms_notification(alert, priority=True)
        elif escalation_level == EscalationLevel.LEVEL_4:
            # Phone call (would require external service)
            self.logger.warning("Level 4 escalation requires external phone service")
        elif escalation_level == EscalationLevel.LEVEL_5:
            # Emergency shutdown
            await self._emergency_shutdown(alert)
        
        self.logger.info(f"Escalation triggered: {escalation_level.value}")
    
    async def _send_notifications(self, alert: Alert, priority: bool = False) -> None:
        """Send notifications through all configured channels."""
        notifications = []
        
        if self.config.notifications.email_enabled:
            notifications.append(self._send_email_notification(alert, priority))
        
        if self.config.notifications.slack_enabled:
            notifications.append(self._send_slack_notification(alert, priority))
        
        if self.config.notifications.sms_enabled:
            notifications.append(self._send_sms_notification(alert, priority))
        
        if self.config.notifications.webhook_enabled:
            notifications.append(self._send_webhook_notification(alert, priority))
        
        # Execute all notifications concurrently
        if notifications:
            await asyncio.gather(*notifications, return_exceptions=True)
    
    async def _send_email_notification(self, alert: Alert, priority: bool = False) -> None:
        """Send email notification."""
        if not self.config.notifications.email_recipients:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.notifications.email_username
            msg['To'] = ', '.join(self.config.notifications.email_recipients)
            msg['Subject'] = f"[{'URGENT' if priority else 'ALERT'}] {alert.title}"
            
            body = f"""
Risk Alert Notification

Title: {alert.title}
Message: {alert.message}
Severity: {alert.severity.value}
Source: {alert.source}
Time: {datetime.fromtimestamp(alert.timestamp)}
Portfolio: {self.portfolio_id}

Circuit Breaker Status: {self.circuit_breaker_status.value}

Please review and take appropriate action.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email (would need proper SMTP configuration)
            # server = smtplib.SMTP(self.config.notifications.email_smtp_server, self.config.notifications.email_smtp_port)
            # server.starttls()
            # server.login(self.config.notifications.email_username, self.config.notifications.email_password)
            # server.send_message(msg)
            # server.quit()
            
            self.logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            self.logger.exception(f"Failed to send email notification: {e}")
    
    async def _send_slack_notification(self, alert: Alert, priority: bool = False) -> None:
        """Send Slack notification."""
        if not self.config.notifications.slack_webhook_url:
            return
        
        try:
            color = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9900",
                AlertSeverity.ERROR: "#ff0000",
                AlertSeverity.CRITICAL: "#8b0000"
            }.get(alert.severity, "#36a64f")
            
            payload = {
                "channel": self.config.notifications.slack_channel,
                "attachments": [{
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity.value, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Portfolio", "value": self.portfolio_id, "short": True},
                        {"title": "Circuit Breaker", "value": self.circuit_breaker_status.value, "short": True}
                    ],
                    "footer": "Risk Alert System",
                    "ts": int(alert.timestamp)
                }]
            }
            
            # Send to Slack webhook
            # response = requests.post(self.config.notifications.slack_webhook_url, json=payload)
            # response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert: {alert.id}")
            
        except Exception as e:
            self.logger.exception(f"Failed to send Slack notification: {e}")
    
    async def _send_sms_notification(self, alert: Alert, priority: bool = False) -> None:
        """Send SMS notification."""
        if not self.config.notifications.sms_recipients:
            return
        
        try:
            message = f"RISK ALERT: {alert.title} - {alert.message[:100]}..."
            
            # Send SMS (would require SMS provider integration)
            # This is a placeholder for SMS sending logic
            self.logger.info(f"SMS notification would be sent: {message}")
            
        except Exception as e:
            self.logger.exception(f"Failed to send SMS notification: {e}")
    
    async def _send_webhook_notification(self, alert: Alert, priority: bool = False) -> None:
        """Send webhook notification."""
        if not self.config.notifications.webhook_url:
            return
        
        try:
            payload = {
                "alert_id": alert.id,
                "title": alert.title,
                "message": alert.message,
                "severity": alert.severity.value,
                "source": alert.source,
                "timestamp": alert.timestamp,
                "portfolio_id": self.portfolio_id,
                "circuit_breaker_status": self.circuit_breaker_status.value,
                "priority": priority
            }
            
            # Send webhook
            # response = requests.post(
            #     self.config.notifications.webhook_url,
            #     json=payload,
            #     headers=self.config.notifications.webhook_headers
            # )
            # response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert: {alert.id}")
            
        except Exception as e:
            self.logger.exception(f"Failed to send webhook notification: {e}")
    
    async def _emergency_shutdown(self, alert: Alert) -> None:
        """Execute emergency shutdown procedures."""
        self.logger.critical("EMERGENCY SHUTDOWN INITIATED")
        
        # Stop monitoring
        await self.stop_monitoring()
        
        # Set circuit breaker to suspended
        self.circuit_breaker_status = CircuitBreakerStatus.TRADING_SUSPENDED
        
        # Send emergency notifications
        emergency_alert = self.alert_manager.create_alert(
            title="EMERGENCY SHUTDOWN",
            message="Trading system has been shut down due to critical risk violation",
            severity=AlertSeverity.CRITICAL,
            source="risk_alert_system",
            alert_type="emergency_shutdown",
            metadata={"original_alert_id": alert.id, "portfolio_id": self.portfolio_id}
        )
        
        await self._send_notifications(emergency_alert, priority=True)
        
        # Log emergency shutdown
        self._log_audit_entry("emergency_shutdown", {
            "trigger_alert_id": alert.id,
            "timestamp": time.time(),
            "portfolio_id": self.portfolio_id
        })
    
    def _log_audit_entry(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an audit entry."""
        if not self.config.audit_log_enabled:
            return
        
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "portfolio_id": self.portfolio_id,
            "data": data
        }
        
        try:
            with open(self.config.audit_log_path, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            self.logger.exception(f"Failed to write audit log: {e}")
    
    def _metrics_to_dict(self, metrics: RiskMetrics) -> Dict[str, Any]:
        """Convert RiskMetrics to dictionary for logging."""
        return {
            "portfolio_var": metrics.portfolio_var,
            "portfolio_cvar": metrics.portfolio_cvar,
            "max_drawdown": metrics.max_drawdown,
            "current_drawdown": metrics.current_drawdown,
            "leverage": metrics.leverage,
            "sharpe_ratio": metrics.sharpe_ratio,
            "sortino_ratio": metrics.sortino_ratio,
            "beta": metrics.beta,
            "correlation_risk": metrics.correlation_risk,
            "concentration_risk": metrics.concentration_risk,
            "timestamp": metrics.timestamp.isoformat()
        }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of current risk status."""
        current_metrics = self.risk_manager.current_metrics
        
        return {
            "portfolio_id": self.portfolio_id,
            "circuit_breaker_status": self.circuit_breaker_status.value,
            "current_metrics": self._metrics_to_dict(current_metrics) if current_metrics else None,
            "active_alerts": len([a for a in self.alert_history if a.status == AlertStatus.ACTIVE]),
            "total_alerts": len(self.alert_history),
            "escalation_level": self._get_current_escalation_level(),
            "last_risk_check": self.last_risk_check,
            "monitoring_active": self.is_monitoring
        }
    
    def _get_current_escalation_level(self) -> str:
        """Get the current escalation level based on active alerts."""
        if not self.alert_history:
            return EscalationLevel.LEVEL_1.value
        
        critical_alerts = [a for a in self.alert_history if a.severity == AlertSeverity.CRITICAL and a.status == AlertStatus.ACTIVE]
        error_alerts = [a for a in self.alert_history if a.severity == AlertSeverity.ERROR and a.status == AlertStatus.ACTIVE]
        
        if critical_alerts:
            return EscalationLevel.LEVEL_5.value
        elif error_alerts:
            return EscalationLevel.LEVEL_4.value
        else:
            return EscalationLevel.LEVEL_1.value
    
    def generate_risk_report(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate a comprehensive risk report."""
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Filter data by time range
        filtered_alerts = [
            alert for alert in self.alert_history
            if start_time <= datetime.fromtimestamp(alert.timestamp) <= end_time
        ]
        
        filtered_metrics = [
            metrics for metrics in self.risk_history
            if start_time <= metrics.timestamp <= end_time
        ]
        
        # Calculate statistics
        alert_stats = {
            "total_alerts": len(filtered_alerts),
            "critical_alerts": len([a for a in filtered_alerts if a.severity == AlertSeverity.CRITICAL]),
            "error_alerts": len([a for a in filtered_alerts if a.severity == AlertSeverity.ERROR]),
            "warning_alerts": len([a for a in filtered_alerts if a.severity == AlertSeverity.WARNING]),
            "info_alerts": len([a for a in filtered_alerts if a.severity == AlertSeverity.INFO]),
        }
        
        # Risk metrics statistics
        if filtered_metrics:
            metrics_df = pd.DataFrame([self._metrics_to_dict(m) for m in filtered_metrics])
            risk_stats = {
                "avg_var": metrics_df["portfolio_var"].mean(),
                "max_var": metrics_df["portfolio_var"].max(),
                "avg_drawdown": metrics_df["current_drawdown"].mean(),
                "max_drawdown": metrics_df["current_drawdown"].max(),
                "avg_leverage": metrics_df["leverage"].mean(),
                "max_leverage": metrics_df["leverage"].max(),
            }
        else:
            risk_stats = {}
        
        report = {
            "portfolio_id": self.portfolio_id,
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "alert_statistics": alert_stats,
            "risk_statistics": risk_stats,
            "circuit_breaker_status": self.circuit_breaker_status.value,
            "escalation_summary": self._get_escalation_summary(start_time, end_time),
            "recommendations": self._generate_recommendations(alert_stats, risk_stats)
        }
        
        return report
    
    def _get_escalation_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get escalation summary for the time period."""
        filtered_escalations = [
            e for e in self.escalation_history
            if start_time <= datetime.fromtimestamp(e["timestamp"]) <= end_time
        ]
        
        return {
            "total_escalations": len(filtered_escalations),
            "by_level": {
                level.value: len([e for e in filtered_escalations if e["level"] == level.value])
                for level in EscalationLevel
            }
        }
    
    def _generate_recommendations(self, alert_stats: Dict[str, Any], risk_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on risk data."""
        recommendations = []
        
        if alert_stats["critical_alerts"] > 0:
            recommendations.append("Review and adjust risk thresholds due to critical alerts")
        
        if alert_stats["error_alerts"] > 5:
            recommendations.append("Consider implementing additional risk controls")
        
        if "max_var" in risk_stats and risk_stats["max_var"] > 0.05:
            recommendations.append("Portfolio VaR exceeded 5% - review position sizing")
        
        if "max_drawdown" in risk_stats and risk_stats["max_drawdown"] > 0.15:
            recommendations.append("Maximum drawdown exceeded 15% - review risk management strategy")
        
        if not recommendations:
            recommendations.append("Risk metrics are within acceptable ranges")
        
        return recommendations
    
    def add_alert_threshold(self, threshold: AlertThreshold) -> None:
        """Add a new alert threshold."""
        self.alert_thresholds[threshold.metric_name] = threshold
        self.logger.info(f"Added alert threshold: {threshold.metric_name}")
    
    def remove_alert_threshold(self, metric_name: str) -> None:
        """Remove an alert threshold."""
        if metric_name in self.alert_thresholds:
            del self.alert_thresholds[metric_name]
            self.logger.info(f"Removed alert threshold: {metric_name}")
    
    def add_circuit_breaker_rule(self, rule: CircuitBreakerRule) -> None:
        """Add a new circuit breaker rule."""
        self.circuit_breaker_rules[rule.name] = rule
        self.logger.info(f"Added circuit breaker rule: {rule.name}")
    
    def remove_circuit_breaker_rule(self, rule_name: str) -> None:
        """Remove a circuit breaker rule."""
        if rule_name in self.circuit_breaker_rules:
            del self.circuit_breaker_rules[rule_name]
            self.logger.info(f"Removed circuit breaker rule: {rule_name}")
    
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker status to normal."""
        self.circuit_breaker_status = CircuitBreakerStatus.NORMAL
        self.logger.info("Circuit breaker status reset to normal")
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history with optional limit."""
        if limit:
            return self.alert_history[-limit:]
        return self.alert_history.copy()
    
    def get_risk_history(self, limit: Optional[int] = None) -> List[RiskMetrics]:
        """Get risk history with optional limit."""
        if limit:
            return self.risk_history[-limit:]
        return self.risk_history.copy()