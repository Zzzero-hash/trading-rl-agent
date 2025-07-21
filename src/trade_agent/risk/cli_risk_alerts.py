"""
CLI interface for the Risk Alert System.

Provides command-line tools for:
- Starting/stopping risk monitoring
- Viewing alerts and risk metrics
- Managing alert thresholds and circuit breakers
- Generating risk reports
- Configuring notifications
"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

from src.trade_agent.core.logging import get_logger
from src.trade_agent.monitoring.alert_manager import AlertManager, AlertSeverity

from .alert_system import (
    AlertThreshold,
    CircuitBreakerRule,
    EscalationLevel,
    RiskAlertConfig,
    RiskAlertSystem,
)
from .manager import RiskLimits, RiskManager

console = Console()
logger = get_logger(__name__)


def load_risk_alert_config(config_path: str | None = None) -> RiskAlertConfig:
    """Load risk alert configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        return RiskAlertConfig(**config_data)
    # Use default configuration
    return RiskAlertConfig()


def create_sample_risk_manager() -> RiskManager:
    """Create a sample risk manager for demonstration purposes."""
    risk_limits = RiskLimits(
        max_portfolio_var=0.02,
        max_drawdown=0.10,
        max_leverage=1.0,
        max_correlation=0.8,
        max_position_size=0.1,
        max_sector_exposure=0.3,
        max_daily_trades=100,
        max_daily_volume=1000000,
        stop_loss_pct=0.05,
        take_profit_pct=0.15,
    )
    return RiskManager(risk_limits)


@click.group()
@click.option("--config", "-c", help="Path to risk alert configuration file")
@click.pass_context
def cli(ctx: click.Context, config: str | None) -> None:
    """Risk Alert System CLI - Manage automated risk alerts and circuit breakers."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.option("--interval", default=60, help="Monitoring interval in seconds")
@click.pass_context
def start_monitoring(ctx: click.Context, portfolio_id: str, interval: int) -> None:
    """Start real-time risk monitoring."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    config.monitoring_interval_seconds = interval

    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    # Create risk alert system
    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    console.print(Panel(f"Starting risk monitoring for portfolio: {portfolio_id}", style="green"))

    async def run_monitoring() -> None:
        await risk_alert_system.start_monitoring()

        # Keep monitoring running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping monitoring...[/yellow]")
            await risk_alert_system.stop_monitoring()

    asyncio.run(run_monitoring())


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.pass_context
def status(ctx: click.Context, portfolio_id: str) -> None:
    """Show current risk alert system status."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    summary = risk_alert_system.get_risk_summary()

    # Create status table
    table = Table(title=f"Risk Alert System Status - Portfolio: {portfolio_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Circuit Breaker Status", summary["circuit_breaker_status"])
    table.add_row("Monitoring Active", str(summary["monitoring_active"]))
    table.add_row("Active Alerts", str(summary["active_alerts"]))
    table.add_row("Total Alerts", str(summary["total_alerts"]))
    table.add_row("Escalation Level", summary["escalation_level"])
    table.add_row("Last Risk Check", str(summary["last_risk_check"] or "Never"))

    console.print(table)

    # Show current metrics if available
    if summary["current_metrics"]:
        metrics_table = Table(title="Current Risk Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")

        for key, value in summary["current_metrics"].items():
            if key != "timestamp":
                metrics_table.add_row(key.replace("_", " ").title(), f"{value:.4f}")

        console.print(metrics_table)


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.option("--limit", default=10, help="Number of alerts to show")
@click.option(
    "--severity",
    type=click.Choice(["info", "warning", "error", "critical"]),
    help="Filter by severity",
)
@click.option(
    "--status",
    type=click.Choice(["active", "acknowledged", "resolved", "dismissed"]),
    help="Filter by status",
)
@click.pass_context
def alerts(
    ctx: click.Context,
    portfolio_id: str,
    limit: int,
    severity: str | None,
    status: str | None,
) -> None:
    """Show recent alerts."""
    load_risk_alert_config(ctx.obj.get("config_path"))
    create_sample_risk_manager()
    alert_manager = AlertManager()

    # Get alerts from alert manager
    all_alerts = alert_manager.alerts

    # Apply filters
    if severity:
        all_alerts = [a for a in all_alerts if a.severity.value == severity]
    if status:
        all_alerts = [a for a in all_alerts if a.status.value == status]

    # Limit results
    recent_alerts = all_alerts[-limit:] if all_alerts else []

    if not recent_alerts:
        console.print("[yellow]No alerts found.[/yellow]")
        return

    # Create alerts table
    table = Table(title=f"Recent Alerts - Portfolio: {portfolio_id}")
    table.add_column("Time", style="cyan")
    table.add_column("Severity", style="red")
    table.add_column("Title", style="white")
    table.add_column("Status", style="yellow")
    table.add_column("Source", style="blue")

    for alert in recent_alerts:
        severity_color = {
            AlertSeverity.INFO: "green",
            AlertSeverity.WARNING: "yellow",
            AlertSeverity.ERROR: "red",
            AlertSeverity.CRITICAL: "bold red",
        }.get(alert.severity, "white")

        table.add_row(
            datetime.fromtimestamp(alert.timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S"),
            f"[{severity_color}]{alert.severity.value}[/{severity_color}]",
            alert.title[:50] + "..." if len(alert.title) > 50 else alert.title,
            alert.status.value,
            alert.source,
        )

    console.print(table)


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.option("--days", default=7, help="Number of days to include in report")
@click.option("--output", help="Output file path for JSON report")
@click.pass_context
def report(ctx: click.Context, portfolio_id: str, days: int, output: str | None) -> None:
    """Generate risk report."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Generating risk report...", total=None)
        report_data = risk_alert_system.generate_risk_report(start_time, end_time)
        progress.update(task, completed=True)

    # Display report summary
    console.print(Panel(f"Risk Report - Portfolio: {portfolio_id}", style="green"))

    # Alert statistics
    alert_stats = report_data["alert_statistics"]
    alert_table = Table(title="Alert Statistics")
    alert_table.add_column("Metric", style="cyan")
    alert_table.add_column("Count", style="green")

    for key, value in alert_stats.items():
        alert_table.add_row(key.replace("_", " ").title(), str(value))

    console.print(alert_table)

    # Risk statistics
    if report_data["risk_statistics"]:
        risk_stats = report_data["risk_statistics"]
        risk_table = Table(title="Risk Statistics")
        risk_table.add_column("Metric", style="cyan")
        risk_table.add_column("Value", style="green")

        for key, value in risk_stats.items():
            risk_table.add_row(key.replace("_", " ").title(), f"{value:.4f}")

        console.print(risk_table)

    # Recommendations
    if report_data["recommendations"]:
        console.print(Panel("Recommendations", style="yellow"))
        for i, rec in enumerate(report_data["recommendations"], 1):
            console.print(f"{i}. {rec}")

    # Save to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        console.print(f"[green]Report saved to: {output}[/green]")


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.pass_context
def thresholds(ctx: click.Context, portfolio_id: str) -> None:
    """Show configured alert thresholds."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    if not risk_alert_system.alert_thresholds:
        console.print("[yellow]No alert thresholds configured.[/yellow]")
        return

    table = Table(title=f"Alert Thresholds - Portfolio: {portfolio_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Type", style="blue")
    table.add_column("Threshold", style="green")
    table.add_column("Severity", style="red")
    table.add_column("Escalation", style="yellow")
    table.add_column("Enabled", style="white")

    for threshold in risk_alert_system.alert_thresholds.values():
        table.add_row(
            threshold.metric_name,
            threshold.threshold_type,
            str(threshold.threshold_value),
            threshold.severity.value,
            threshold.escalation_level.value,
            "✓" if threshold.enabled else "✗",
        )

    console.print(table)


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.pass_context
def circuit_breakers(ctx: click.Context, portfolio_id: str) -> None:
    """Show configured circuit breaker rules."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    if not risk_alert_system.circuit_breaker_rules:
        console.print("[yellow]No circuit breaker rules configured.[/yellow]")
        return

    table = Table(title=f"Circuit Breaker Rules - Portfolio: {portfolio_id}")
    table.add_column("Name", style="cyan")
    table.add_column("Condition", style="blue")
    table.add_column("Threshold", style="green")
    table.add_column("Action", style="red")
    table.add_column("Enabled", style="white")
    table.add_column("Description", style="yellow")

    for rule in risk_alert_system.circuit_breaker_rules.values():
        table.add_row(
            rule.name,
            rule.trigger_condition,
            str(rule.threshold_value),
            rule.action,
            "✓" if rule.enabled else "✗",
            (rule.description[:50] + "..." if len(rule.description) > 50 else rule.description),
        )

    console.print(table)


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.pass_context
def reset(ctx: click.Context, portfolio_id: str) -> None:
    """Reset circuit breaker status to normal."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    if Confirm.ask(f"Reset circuit breaker status for portfolio {portfolio_id}?"):
        risk_alert_system.reset_circuit_breaker()
        console.print(f"[green]Circuit breaker status reset for portfolio: {portfolio_id}[/green]")


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.pass_context
def config_show(ctx: click.Context, portfolio_id: str) -> None:
    """Show current configuration."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))

    console.print(Panel(f"Risk Alert Configuration - Portfolio: {portfolio_id}", style="green"))

    # Monitoring settings
    monitoring_table = Table(title="Monitoring Settings")
    monitoring_table.add_column("Setting", style="cyan")
    monitoring_table.add_column("Value", style="green")

    monitoring_table.add_row("Monitoring Interval", f"{config.monitoring_interval_seconds} seconds")
    monitoring_table.add_row("Real-time Monitoring", str(config.real_time_monitoring))
    monitoring_table.add_row("Escalation Enabled", str(config.escalation_enabled))
    monitoring_table.add_row("Escalation Timeout", f"{config.escalation_timeout_minutes} minutes")
    monitoring_table.add_row("Audit Log Enabled", str(config.audit_log_enabled))
    monitoring_table.add_row("Report Generation", str(config.report_generation_enabled))

    console.print(monitoring_table)

    # Notification settings
    notifications = config.notifications
    notification_table = Table(title="Notification Settings")
    notification_table.add_column("Channel", style="cyan")
    notification_table.add_column("Enabled", style="green")
    notification_table.add_column("Details", style="yellow")

    notification_table.add_row(
        "Email",
        "✓" if notifications.email_enabled else "✗",
        f"{len(notifications.email_recipients)} recipients",
    )
    notification_table.add_row(
        "Slack",
        "✓" if notifications.slack_enabled else "✗",
        notifications.slack_channel,
    )
    notification_table.add_row(
        "SMS",
        "✓" if notifications.sms_enabled else "✗",
        f"{len(notifications.sms_recipients)} recipients",
    )
    notification_table.add_row(
        "Webhook",
        "✓" if notifications.webhook_enabled else "✗",
        "Configured" if notifications.webhook_url else "Not configured",
    )

    console.print(notification_table)


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.option("--metric", required=True, help="Metric name")
@click.option(
    "--threshold-type",
    required=True,
    type=click.Choice(["min", "max", "change_rate"]),
    help="Threshold type",
)
@click.option("--value", required=True, type=float, help="Threshold value")
@click.option(
    "--severity",
    required=True,
    type=click.Choice(["info", "warning", "error", "critical"]),
    help="Alert severity",
)
@click.option(
    "--escalation",
    required=True,
    type=click.Choice(["level_1", "level_2", "level_3", "level_4", "level_5"]),
    help="Escalation level",
)
@click.option("--cooldown", default=30, help="Cooldown period in minutes")
@click.option("--description", help="Threshold description")
@click.pass_context
def add_threshold(
    ctx: click.Context,
    portfolio_id: str,
    metric: str,
    threshold_type: str,
    value: float,
    severity: str,
    escalation: str,
    cooldown: int,
    description: str,
) -> None:
    """Add a new alert threshold."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    threshold = AlertThreshold(
        metric_name=metric,
        threshold_type=threshold_type,
        threshold_value=value,
        severity=AlertSeverity(severity),
        escalation_level=EscalationLevel(escalation),
        cooldown_minutes=cooldown,
        enabled=True,
        description=description or "",
    )

    risk_alert_system.add_alert_threshold(threshold)
    console.print(f"[green]Added alert threshold for {metric}[/green]")


@cli.command()
@click.option("--portfolio-id", default="default", help="Portfolio identifier")
@click.option("--name", required=True, help="Circuit breaker rule name")
@click.option(
    "--condition",
    required=True,
    type=click.Choice(
        [
            "var_exceeded",
            "drawdown_exceeded",
            "leverage_exceeded",
            "correlation_risk_exceeded",
        ]
    ),
    help="Trigger condition",
)
@click.option("--value", required=True, type=float, help="Threshold value")
@click.option(
    "--action",
    required=True,
    type=click.Choice(["reduce_position", "stop_trading", "liquidate"]),
    help="Action to take",
)
@click.option("--cooldown", default=60, help="Cooldown period in minutes")
@click.option("--description", help="Rule description")
@click.pass_context
def add_circuit_breaker(
    ctx: click.Context,
    portfolio_id: str,
    name: str,
    condition: str,
    value: float,
    action: str,
    cooldown: int,
    description: str,
) -> None:
    """Add a new circuit breaker rule."""
    config = load_risk_alert_config(ctx.obj.get("config_path"))
    risk_manager = create_sample_risk_manager()
    alert_manager = AlertManager()

    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id=portfolio_id,
    )

    rule = CircuitBreakerRule(
        name=name,
        trigger_condition=condition,
        threshold_value=value,
        action=action,
        cooldown_minutes=cooldown,
        enabled=True,
        description=description or "",
    )

    risk_alert_system.add_circuit_breaker_rule(rule)
    console.print(f"[green]Added circuit breaker rule: {name}[/green]")


if __name__ == "__main__":
    cli()
