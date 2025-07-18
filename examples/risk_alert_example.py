#!/usr/bin/env python3
"""
Risk Alert System Example

This example demonstrates how to use the automated risk alerts and circuit breakers system
with various configurations and real-world scenarios.
"""

import asyncio
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.trading_rl_agent.monitoring.alert_manager import AlertManager, AlertSeverity
from src.trading_rl_agent.risk.alert_system import (
    AlertThreshold,
    CircuitBreakerRule,
    EscalationLevel,
    NotificationConfig,
    RiskAlertConfig,
    RiskAlertSystem,
)
from src.trading_rl_agent.risk.manager import RiskLimits, RiskManager, RiskMetrics


def create_sample_portfolio_data(days: int = 252) -> dict[str, pd.Series]:
    """Create sample portfolio data for demonstration."""
    np.random.seed(42)

    # Generate sample returns for different assets
    assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX"]
    returns_data = {}

    for asset in assets:
        # Generate realistic returns with some correlation
        base_return = np.random.normal(0.0005, 0.02, days)  # Daily returns
        returns_data[asset] = pd.Series(base_return, index=pd.date_range("2023-01-01", periods=days, freq="D"))

    return returns_data


def create_sample_risk_metrics() -> RiskMetrics:
    """Create sample risk metrics for demonstration."""
    return RiskMetrics(
        portfolio_var=random.uniform(0.01, 0.08),
        portfolio_cvar=random.uniform(0.015, 0.12),
        max_drawdown=random.uniform(0.05, 0.25),
        current_drawdown=random.uniform(0.02, 0.20),
        leverage=random.uniform(0.8, 2.5),
        sharpe_ratio=random.uniform(-0.5, 2.0),
        sortino_ratio=random.uniform(-0.3, 2.5),
        beta=random.uniform(0.5, 2.0),
        correlation_risk=random.uniform(0.3, 0.9),
        concentration_risk=random.uniform(0.1, 0.6),
        timestamp=datetime.now(),
    )


def create_demo_config() -> RiskAlertConfig:
    """Create a demonstration configuration for the risk alert system."""

    # Alert thresholds for different risk metrics
    alert_thresholds = [
        # VaR thresholds
        {
            "metric_name": "portfolio_var",
            "threshold_type": "max",
            "threshold_value": 0.03,
            "severity": "warning",
            "escalation_level": "level_2",
            "cooldown_minutes": 30,
            "enabled": True,
            "description": "Portfolio VaR exceeds 3%",
        },
        {
            "metric_name": "portfolio_var",
            "threshold_type": "max",
            "threshold_value": 0.05,
            "severity": "error",
            "escalation_level": "level_3",
            "cooldown_minutes": 15,
            "enabled": True,
            "description": "Portfolio VaR exceeds 5%",
        },
        {
            "metric_name": "portfolio_var",
            "threshold_type": "max",
            "threshold_value": 0.08,
            "severity": "critical",
            "escalation_level": "level_5",
            "cooldown_minutes": 5,
            "enabled": True,
            "description": "Portfolio VaR exceeds 8% - CRITICAL",
        },
        # Drawdown thresholds
        {
            "metric_name": "current_drawdown",
            "threshold_type": "max",
            "threshold_value": 0.10,
            "severity": "warning",
            "escalation_level": "level_2",
            "cooldown_minutes": 60,
            "enabled": True,
            "description": "Current drawdown exceeds 10%",
        },
        {
            "metric_name": "current_drawdown",
            "threshold_type": "max",
            "threshold_value": 0.15,
            "severity": "error",
            "escalation_level": "level_3",
            "cooldown_minutes": 30,
            "enabled": True,
            "description": "Current drawdown exceeds 15%",
        },
        # Leverage thresholds
        {
            "metric_name": "leverage",
            "threshold_type": "max",
            "threshold_value": 1.5,
            "severity": "warning",
            "escalation_level": "level_2",
            "cooldown_minutes": 30,
            "enabled": True,
            "description": "Portfolio leverage exceeds 150%",
        },
        {
            "metric_name": "leverage",
            "threshold_type": "max",
            "threshold_value": 2.0,
            "severity": "error",
            "escalation_level": "level_3",
            "cooldown_minutes": 15,
            "enabled": True,
            "description": "Portfolio leverage exceeds 200%",
        },
        # Performance thresholds
        {
            "metric_name": "sharpe_ratio",
            "threshold_type": "min",
            "threshold_value": 0.0,
            "severity": "warning",
            "escalation_level": "level_1",
            "cooldown_minutes": 120,
            "enabled": True,
            "description": "Portfolio Sharpe ratio is negative",
        },
    ]

    # Circuit breaker rules
    circuit_breaker_rules = [
        {
            "name": "var_circuit_breaker",
            "trigger_condition": "var_exceeded",
            "threshold_value": 0.10,
            "action": "stop_trading",
            "cooldown_minutes": 120,
            "enabled": True,
            "description": "Stop all trading when VaR exceeds 10%",
        },
        {
            "name": "drawdown_circuit_breaker",
            "trigger_condition": "drawdown_exceeded",
            "threshold_value": 0.25,
            "action": "liquidate",
            "cooldown_minutes": 60,
            "enabled": True,
            "description": "Liquidate positions when drawdown exceeds 25%",
        },
        {
            "name": "leverage_circuit_breaker",
            "trigger_condition": "leverage_exceeded",
            "threshold_value": 2.5,
            "action": "reduce_position",
            "cooldown_minutes": 30,
            "enabled": True,
            "description": "Reduce positions when leverage exceeds 250%",
        },
    ]

    # Notification configuration
    notifications = NotificationConfig(
        email_enabled=True,
        email_recipients=["demo@example.com", "risk-manager@example.com"],
        email_smtp_server="smtp.gmail.com",
        email_smtp_port=587,
        email_username="risk-alerts@example.com",
        email_password="",  # Would be set via environment variable
        slack_enabled=True,
        slack_webhook_url="",  # Would be set via environment variable
        slack_channel="#risk-alerts",
        sms_enabled=False,
        sms_provider="twilio",
        sms_api_key="",
        sms_api_secret="",
        sms_recipients=["+1234567890"],
        webhook_enabled=False,
        webhook_url="",
        webhook_headers={"Content-Type": "application/json"},
    )

    return RiskAlertConfig(
        monitoring_interval_seconds=30,  # Faster for demo
        real_time_monitoring=True,
        alert_thresholds=alert_thresholds,
        circuit_breaker_rules=circuit_breaker_rules,
        notifications=notifications,
        escalation_enabled=True,
        escalation_timeout_minutes=30,
        audit_log_enabled=True,
        audit_log_path="logs/demo_risk_audit.log",
        report_generation_enabled=True,
        report_schedule_hours=24,
    )


async def demo_basic_monitoring() -> None:
    """Demonstrate basic risk monitoring functionality."""
    print("=== Basic Risk Monitoring Demo ===")

    # Create components
    risk_limits = RiskLimits(max_portfolio_var=0.02, max_drawdown=0.10, max_leverage=1.0)
    risk_manager = RiskManager(risk_limits)
    alert_manager = AlertManager()
    config = create_demo_config()

    # Create risk alert system
    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager, alert_manager=alert_manager, config=config, portfolio_id="demo_portfolio"
    )

    print("Risk Alert System initialized for portfolio: demo_portfolio")
    print(f"Alert thresholds configured: {len(risk_alert_system.alert_thresholds)}")
    print(f"Circuit breaker rules configured: {len(risk_alert_system.circuit_breaker_rules)}")

    # Simulate risk monitoring
    print("\n--- Simulating Risk Monitoring ---")

    for i in range(10):
        # Create sample risk metrics
        metrics = create_sample_risk_metrics()
        risk_manager.current_metrics = metrics

        # Check risk metrics (this would normally be done by the monitoring loop)
        await risk_alert_system._check_risk_metrics()

        print(
            f"Check {i + 1}: VaR={metrics.portfolio_var:.3f}, "
            f"Drawdown={metrics.current_drawdown:.3f}, "
            f"Leverage={metrics.leverage:.2f}, "
            f"Circuit Breaker: {risk_alert_system.circuit_breaker_status.value}"
        )

        # Check if any alerts were triggered
        recent_alerts = alert_manager.alerts[-3:]  # Last 3 alerts
        if recent_alerts:
            for alert in recent_alerts:
                print(f"  ALERT: {alert.severity.value.upper()} - {alert.title}")

        await asyncio.sleep(1)  # Simulate time passing

    # Show final status
    summary = risk_alert_system.get_risk_summary()
    print("\n--- Final Status ---")
    print(f"Total alerts generated: {summary['total_alerts']}")
    print(f"Active alerts: {summary['active_alerts']}")
    print(f"Circuit breaker status: {summary['circuit_breaker_status']}")
    print(f"Escalation level: {summary['escalation_level']}")


async def demo_alert_thresholds() -> None:
    """Demonstrate alert threshold functionality."""
    print("\n=== Alert Thresholds Demo ===")

    # Create components
    risk_manager = RiskManager()
    alert_manager = AlertManager()
    config = create_demo_config()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager, alert_manager=alert_manager, config=config, portfolio_id="threshold_demo"
    )

    # Test different threshold scenarios
    test_scenarios: list[dict[str, Any]] = [
        {
            "name": "Normal Risk Levels",
            "metrics": RiskMetrics(
                portfolio_var=0.02,
                current_drawdown=0.05,
                leverage=1.1,
                sharpe_ratio=1.2,
                portfolio_cvar=0.03,
                max_drawdown=0.08,
                sortino_ratio=1.5,
                beta=1.0,
                correlation_risk=0.4,
                concentration_risk=0.2,
                timestamp=datetime.now(),
            ),
        },
        {
            "name": "High VaR Warning",
            "metrics": RiskMetrics(
                portfolio_var=0.04,  # Triggers warning
                current_drawdown=0.05,
                leverage=1.1,
                sharpe_ratio=1.2,
                portfolio_cvar=0.06,
                max_drawdown=0.08,
                sortino_ratio=1.5,
                beta=1.0,
                correlation_risk=0.4,
                concentration_risk=0.2,
                timestamp=datetime.now(),
            ),
        },
        {
            "name": "Critical Risk Levels",
            "metrics": RiskMetrics(
                portfolio_var=0.09,  # Triggers critical
                current_drawdown=0.18,  # Triggers error
                leverage=2.2,  # Triggers error
                sharpe_ratio=-0.3,  # Triggers warning
                portfolio_cvar=0.12,
                max_drawdown=0.20,
                sortino_ratio=-0.2,
                beta=1.8,
                correlation_risk=0.6,
                concentration_risk=0.3,
                timestamp=datetime.now(),
            ),
        },
    ]

    for scenario in test_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")

        # Set metrics
        metrics = scenario["metrics"]
        assert isinstance(metrics, RiskMetrics)
        risk_manager.current_metrics = metrics

        # Check thresholds
        await risk_alert_system._check_alert_thresholds(metrics)

        # Show triggered alerts
        recent_alerts = alert_manager.alerts[-5:]  # Last 5 alerts
        if recent_alerts:
            for alert in recent_alerts:
                print(f"  {alert.severity.value.upper()}: {alert.title}")
        else:
            print("  No alerts triggered")


async def demo_circuit_breakers() -> None:
    """Demonstrate circuit breaker functionality."""
    print("\n=== Circuit Breakers Demo ===")

    # Create components
    risk_manager = RiskManager()
    alert_manager = AlertManager()
    config = create_demo_config()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager, alert_manager=alert_manager, config=config, portfolio_id="circuit_breaker_demo"
    )

    # Test circuit breaker scenarios
    circuit_breaker_scenarios: list[dict[str, Any]] = [
        {
            "name": "VaR Circuit Breaker",
            "metrics": RiskMetrics(
                portfolio_var=0.12,  # Triggers VaR circuit breaker
                current_drawdown=0.05,
                leverage=1.1,
                sharpe_ratio=1.2,
                portfolio_cvar=0.15,
                max_drawdown=0.08,
                sortino_ratio=1.5,
                beta=1.0,
                correlation_risk=0.4,
                concentration_risk=0.2,
                timestamp=datetime.now(),
            ),
        },
        {
            "name": "Drawdown Circuit Breaker",
            "metrics": RiskMetrics(
                portfolio_var=0.03,
                current_drawdown=0.28,  # Triggers drawdown circuit breaker
                leverage=1.1,
                sharpe_ratio=1.2,
                portfolio_cvar=0.05,
                max_drawdown=0.30,
                sortino_ratio=1.5,
                beta=1.0,
                correlation_risk=0.4,
                concentration_risk=0.2,
                timestamp=datetime.now(),
            ),
        },
        {
            "name": "Leverage Circuit Breaker",
            "metrics": RiskMetrics(
                portfolio_var=0.03,
                current_drawdown=0.05,
                leverage=2.8,  # Triggers leverage circuit breaker
                sharpe_ratio=1.2,
                portfolio_cvar=0.05,
                max_drawdown=0.08,
                sortino_ratio=1.5,
                beta=1.0,
                correlation_risk=0.4,
                concentration_risk=0.2,
                timestamp=datetime.now(),
            ),
        },
    ]

    for scenario in circuit_breaker_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")

        # Set metrics
        metrics = scenario["metrics"]
        assert isinstance(metrics, RiskMetrics)
        risk_manager.current_metrics = metrics

        # Check circuit breakers
        await risk_alert_system._check_circuit_breakers(metrics)

        # Show triggered circuit breakers
        recent_alerts = alert_manager.alerts[-3:]  # Last 3 alerts
        if recent_alerts:
            for alert in recent_alerts:
                if hasattr(alert, "alert_type") and alert.alert_type == "circuit_breaker":
                    print(f"  CIRCUIT BREAKER: {alert.title}")
                    print(f"    Action: {alert.metadata.get('action', 'Unknown')}")
        else:
            print("  No circuit breakers triggered")


async def demo_escalation() -> None:
    """Demonstrate escalation procedures."""
    print("\n=== Escalation Procedures Demo ===")

    # Create components
    risk_manager = RiskManager()
    alert_manager = AlertManager()
    config = create_demo_config()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager, alert_manager=alert_manager, config=config, portfolio_id="escalation_demo"
    )

    # Test different escalation levels
    escalation_scenarios: list[dict[str, Any]] = [
        {
            "name": "Level 2 Escalation (Email)",
            "threshold": AlertThreshold(
                metric_name="portfolio_var",
                threshold_type="max",
                threshold_value=0.04,
                severity=AlertSeverity.WARNING,
                escalation_level=EscalationLevel.LEVEL_2,
                cooldown_minutes=30,
                enabled=True,
                description="Test Level 2 escalation",
            ),
        },
        {
            "name": "Level 3 Escalation (Slack/SMS)",
            "threshold": AlertThreshold(
                metric_name="current_drawdown",
                threshold_type="max",
                threshold_value=0.15,
                severity=AlertSeverity.ERROR,
                escalation_level=EscalationLevel.LEVEL_3,
                cooldown_minutes=15,
                enabled=True,
                description="Test Level 3 escalation",
            ),
        },
        {
            "name": "Level 5 Escalation (Emergency Shutdown)",
            "threshold": AlertThreshold(
                metric_name="portfolio_var",
                threshold_type="max",
                threshold_value=0.10,
                severity=AlertSeverity.CRITICAL,
                escalation_level=EscalationLevel.LEVEL_5,
                cooldown_minutes=5,
                enabled=True,
                description="Test Level 5 escalation",
            ),
        },
    ]

    for scenario in escalation_scenarios:
        print(f"\n--- Testing: {scenario['name']} ---")

        # Create test alert
        threshold = scenario["threshold"]
        assert isinstance(threshold, AlertThreshold)
        test_alert = alert_manager.create_alert(
            title=f"Test Alert: {scenario['name']}",
            message="This is a test alert for escalation demonstration",
            severity=threshold.severity,
            source="escalation_demo",
            alert_type="test_escalation",
        )

        # Trigger escalation
        await risk_alert_system._trigger_escalation(threshold.escalation_level, test_alert)

        print(f"  Escalation level: {threshold.escalation_level.value}")
        print(f"  Alert severity: {test_alert.severity.value}")

        # Show escalation history
        recent_escalations = risk_alert_system.escalation_history[-1:]
        if recent_escalations:
            for escalation in recent_escalations:
                print(f"  Escalation triggered: {escalation['level']}")


async def demo_reporting() -> None:
    """Demonstrate risk reporting functionality."""
    print("\n=== Risk Reporting Demo ===")

    # Create components
    risk_manager = RiskManager()
    alert_manager = AlertManager()
    config = create_demo_config()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager, alert_manager=alert_manager, config=config, portfolio_id="reporting_demo"
    )

    # Generate some sample data
    print("Generating sample risk data...")

    for i in range(20):
        metrics = create_sample_risk_metrics()
        risk_alert_system.risk_history.append(metrics)

        # Create some sample alerts
        if random.random() < 0.3:  # 30% chance of alert
            alert = alert_manager.create_alert(
                title=f"Sample Alert {i + 1}",
                message=f"This is sample alert number {i + 1}",
                severity=random.choice(list(AlertSeverity)),
                source="reporting_demo",
                alert_type="sample_alert",
            )
            risk_alert_system.alert_history.append(alert)

    # Generate report
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)

    print("Generating risk report...")
    report = risk_alert_system.generate_risk_report(start_time, end_time)

    # Display report summary
    print("\n--- Risk Report Summary ---")
    print(f"Portfolio: {report['portfolio_id']}")
    print(f"Report Period: {report['report_period']['start']} to {report['report_period']['end']}")
    print(f"Circuit Breaker Status: {report['circuit_breaker_status']}")

    # Alert statistics
    alert_stats = report["alert_statistics"]
    print("\nAlert Statistics:")
    for key, value in alert_stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")

    # Risk statistics
    if report["risk_statistics"]:
        risk_stats = report["risk_statistics"]
        print("\nRisk Statistics:")
        for key, value in risk_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    # Recommendations
    if report["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")

    # Save report to file
    report_file = "demo_risk_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_file}")


async def demo_configuration_management() -> None:
    """Demonstrate configuration management functionality."""
    print("\n=== Configuration Management Demo ===")

    # Create components
    risk_manager = RiskManager()
    alert_manager = AlertManager()
    config = create_demo_config()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager, alert_manager=alert_manager, config=config, portfolio_id="config_demo"
    )

    # Show current configuration
    print("Current alert thresholds:")
    for threshold in risk_alert_system.alert_thresholds.values():
        print(
            f"  {threshold.metric_name}: {threshold.threshold_type} {threshold.threshold_value} "
            f"({threshold.severity.value}, {threshold.escalation_level.value})"
        )

    print("\nCurrent circuit breaker rules:")
    for rule in risk_alert_system.circuit_breaker_rules.values():
        print(f"  {rule.name}: {rule.trigger_condition} > {rule.threshold_value} -> {rule.action}")

    # Add new threshold
    print("\n--- Adding New Alert Threshold ---")
    new_threshold = AlertThreshold(
        metric_name="beta",
        threshold_type="max",
        threshold_value=1.8,
        severity=AlertSeverity.WARNING,
        escalation_level=EscalationLevel.LEVEL_2,
        cooldown_minutes=60,
        enabled=True,
        description="Portfolio beta exceeds 1.8",
    )

    risk_alert_system.add_alert_threshold(new_threshold)
    print(f"Added new threshold: {new_threshold.metric_name}")

    # Add new circuit breaker rule
    print("\n--- Adding New Circuit Breaker Rule ---")
    new_rule = CircuitBreakerRule(
        name="beta_circuit_breaker",
        trigger_condition="beta_exceeded",
        threshold_value=2.5,
        action="reduce_position",
        cooldown_minutes=45,
        enabled=True,
        description="Reduce positions when beta exceeds 2.5",
    )

    risk_alert_system.add_circuit_breaker_rule(new_rule)
    print(f"Added new circuit breaker rule: {new_rule.name}")

    # Show updated configuration
    print("\nUpdated configuration:")
    print(f"  Alert thresholds: {len(risk_alert_system.alert_thresholds)}")
    print(f"  Circuit breaker rules: {len(risk_alert_system.circuit_breaker_rules)}")


async def main() -> None:
    """Run all demonstration scenarios."""
    print("Risk Alert System Demonstration")
    print("=" * 50)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    try:
        # Run all demos
        await demo_basic_monitoring()
        await demo_alert_thresholds()
        await demo_circuit_breakers()
        await demo_escalation()
        await demo_reporting()
        await demo_configuration_management()

        print("\n" + "=" * 50)
        print("Demonstration completed successfully!")
        print("Check the logs/ directory for audit logs and demo_risk_report.json for the generated report.")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
