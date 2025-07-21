"""
CLI commands for system health monitoring.

This module provides command-line interface for the system health monitoring
capabilities, integrating with the existing CLI framework.
"""

import time
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .monitoring import (
    AlertManager,
    AlertSeverity,
    HealthDashboard,
    HealthStatus,
    MetricsCollector,
    SystemHealthMonitor,
)

app = typer.Typer(name="health", help="System health monitoring commands")
console = Console()


@app.command()
def monitor(
    duration: int = typer.Option(300, "--duration", "-d", help="Monitoring duration in seconds"),
    interval: float = typer.Option(30.0, "--interval", "-i", help="Health check interval in seconds"),
    output_dir: str = typer.Option("health_reports", "--output", "-o", help="Output directory for reports"),
    live: bool = typer.Option(False, "--live", "-l", help="Show live dashboard"),
    html: bool = typer.Option(True, "--html", help="Generate HTML dashboard"),
    json_output: bool = typer.Option(True, "--json", help="Generate JSON reports"),
) -> None:
    """Start comprehensive system health monitoring."""
    console.print("[bold blue]🚀 Starting System Health Monitoring[/bold blue]")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize monitoring components
    with console.status("[bold green]Initializing monitoring components..."):
        metrics_collector = MetricsCollector(max_history=10000)
        alert_manager = AlertManager(max_alerts=1000)

        health_monitor = SystemHealthMonitor(
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            check_interval=interval,
            max_history=1000,
        )

        health_dashboard = HealthDashboard(
            system_health_monitor=health_monitor,
            metrics_collector=metrics_collector,
            alert_manager=alert_manager,
            output_dir=str(output_path),
        )

    console.print("✅ Monitoring components initialized")
    console.print(f"📁 Output directory: {output_path.absolute()}")
    console.print(f"⏱️  Duration: {duration} seconds")
    console.print(f"🔄 Check interval: {interval} seconds")

    # Start monitoring
    health_monitor.start_monitoring()

    try:
        if live:
            _run_live_monitoring(
                health_monitor,
                health_dashboard,
                duration,
                output_path,
                html,
                json_output,
            )
        else:
            _run_basic_monitoring(
                health_monitor,
                health_dashboard,
                duration,
                output_path,
                html,
                json_output,
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Monitoring interrupted by user[/yellow]")
    finally:
        health_monitor.stop_monitoring()
        console.print("[green]✅ Monitoring stopped[/green]")


def _run_live_monitoring(
    health_monitor: Any,
    health_dashboard: Any,
    duration: int,
    output_path: Path,
    html: bool,
    json_output: bool,
) -> None:
    """Run monitoring with live dashboard display."""
    start_time = time.time()

    def generate_layout() -> Layout:
        """Generate the live dashboard layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        layout["left"].split_column(Layout(name="system"), Layout(name="trading"))

        layout["right"].split_column(Layout(name="health"), Layout(name="alerts"))

        return layout

    def update_dashboard() -> Layout:
        """Update the dashboard content."""
        layout = generate_layout()

        # Get current data
        dashboard_data = health_monitor.get_system_health_dashboard()
        health_summary = health_monitor.get_health_summary()

        # Header
        elapsed = time.time() - start_time
        remaining = max(0, duration - elapsed)

        status_color = {
            "healthy": "green",
            "degraded": "yellow",
            "critical": "red",
            "unknown": "gray",
        }.get(health_summary["status"], "gray")

        layout["header"].update(
            Panel(
                f"[bold]System Health Dashboard[/bold]\n"
                f"Status: [{status_color}]{health_summary['status'].upper()}[/{status_color}] | "
                f"Elapsed: {elapsed:.0f}s | Remaining: {remaining:.0f}s",
                style="blue",
            )
        )

        # System metrics
        system_table = Table(title="System Metrics", show_header=True, header_style="bold magenta")
        system_table.add_column("Metric", style="cyan")
        system_table.add_column("Value", style="green")

        metrics = dashboard_data["current_metrics"]
        system_table.add_row("CPU Usage", f"{metrics['cpu_percent']:.1f}%")
        system_table.add_row("Memory Usage", f"{metrics['memory_percent']:.1f}%")
        system_table.add_row("Disk Usage", f"{metrics['disk_percent']:.1f}%")
        system_table.add_row("Load Average", f"{metrics['load_average'][0]:.2f}")
        system_table.add_row("Process Count", str(metrics["process_count"]))

        layout["system"].update(Panel(system_table))

        # Trading metrics
        trading_table = Table(title="Trading Metrics", show_header=True, header_style="bold magenta")
        trading_table.add_column("Metric", style="cyan")
        trading_table.add_column("Value", style="green")

        trading_metrics = dashboard_data["trading_metrics"]
        trading_table.add_row("Total P&L", f"${trading_metrics['total_pnl']:.2f}")
        trading_table.add_row("Daily P&L", f"${trading_metrics['daily_pnl']:.2f}")
        trading_table.add_row("Sharpe Ratio", f"{trading_metrics['sharpe_ratio']:.3f}")
        trading_table.add_row("Max Drawdown", f"{trading_metrics['max_drawdown']:.2%}")
        trading_table.add_row("Win Rate", f"{trading_metrics['win_rate']:.2%}")
        trading_table.add_row("Total Trades", str(trading_metrics["total_trades"]))
        trading_table.add_row("Execution Latency", f"{trading_metrics['execution_latency']:.1f}ms")

        layout["trading"].update(Panel(trading_table))

        # Health status
        health_table = Table(title="Health Status", show_header=True, header_style="bold magenta")
        health_table.add_column("Check Type", style="cyan")
        health_table.add_column("Status", style="green")
        health_table.add_column("Message", style="white")

        recent_checks = dashboard_data.get("recent_health_checks", [])
        for check in recent_checks[-5:]:  # Last 5 checks
            status_color = {
                "healthy": "green",
                "degraded": "yellow",
                "critical": "red",
                "unknown": "gray",
            }.get(check["status"], "gray")

            health_table.add_row(
                check["check_type"],
                f"[{status_color}]{check['status'].upper()}[/{status_color}]",
                (check["message"][:50] + "..." if len(check["message"]) > 50 else check["message"]),
            )

        layout["health"].update(Panel(health_table))

        # Alerts
        alerts_text = Text()
        active_alerts = health_monitor.alert_manager.get_active_alerts() if health_monitor.alert_manager else []

        if active_alerts:
            for alert in active_alerts[-3:]:  # Last 3 alerts
                severity_color = {
                    "info": "blue",
                    "warning": "yellow",
                    "error": "red",
                    "critical": "red",
                }.get(alert.severity.value, "white")

                alerts_text.append(
                    f"[{severity_color}]{alert.severity.value.upper()}[/{severity_color}] ",
                    style="bold",
                )
                alerts_text.append(f"{alert.title}\n")
        else:
            alerts_text.append("No active alerts", style="green")

        layout["alerts"].update(Panel(alerts_text, title="Active Alerts"))

        # Footer
        layout["footer"].update(
            Panel(
                f"Health checks: {health_summary.get('total_checks', 0)} | "
                f"Requests: {health_monitor.total_requests} | "
                f"Errors: {health_monitor.error_count} | "
                f"Error rate: {health_monitor.get_error_rate():.2%} | "
                f"Avg latency: {health_monitor.get_average_latency():.1f}ms",
                style="dim",
            )
        )

        return layout

    # Run live monitoring
    with Live(update_dashboard(), refresh_per_second=2, screen=True) as live:
        while time.time() - start_time < duration:
            live.update(update_dashboard())

            # Generate reports periodically
            elapsed = time.time() - start_time
            if int(elapsed) % 60 == 0 and elapsed > 0:  # Every minute
                _generate_reports(
                    health_monitor,
                    health_dashboard,
                    output_path,
                    html,
                    json_output,
                    elapsed,
                )

            time.sleep(0.5)

    # Generate final reports
    _generate_reports(
        health_monitor,
        health_dashboard,
        output_path,
        html,
        json_output,
        duration,
        final=True,
    )


def _run_basic_monitoring(
    health_monitor: Any,
    health_dashboard: Any,
    duration: int,
    output_path: Path,
    html: bool,
    json_output: bool,
) -> None:
    """Run basic monitoring with progress bar."""
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Monitoring system health...", total=duration)

        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            progress.update(task, completed=elapsed)

            # Generate reports periodically
            if int(elapsed) % 60 == 0 and elapsed > 0:  # Every minute
                _generate_reports(
                    health_monitor,
                    health_dashboard,
                    output_path,
                    html,
                    json_output,
                    elapsed,
                )

            time.sleep(1)

    # Generate final reports
    _generate_reports(
        health_monitor,
        health_dashboard,
        output_path,
        html,
        json_output,
        duration,
        final=True,
    )


def _generate_reports(
    health_monitor: Any,
    health_dashboard: Any,
    output_path: Path,
    html: bool,
    json_output: bool,
    elapsed: float,
    final: bool = False,
) -> None:
    """Generate monitoring reports."""
    prefix = "final" if final else f"{int(elapsed)}s"

    if html:
        html_path = output_path / f"health_dashboard_{prefix}.html"
        health_dashboard.generate_html_dashboard(str(html_path))
        console.print(f"   🌐 HTML dashboard: {html_path}")

    if json_output:
        json_path = health_dashboard.save_dashboard_data(f"dashboard_data_{prefix}.json")
        console.print(f"   📊 Dashboard data: {json_path}")

    # Generate text report
    report_path = output_path / f"health_report_{prefix}.txt"
    health_monitor.generate_health_report(str(report_path))
    console.print(f"   📄 Health report: {report_path}")


@app.command()
def status() -> None:
    """Show current system health status."""
    console.print("[bold blue]📊 System Health Status[/bold blue]")

    # Initialize monitoring components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    health_monitor = SystemHealthMonitor(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        check_interval=1.0,  # Quick check
    )

    # Run a single health check
    health_results = health_monitor.run_health_checks()
    health_summary = health_monitor.get_health_summary()

    # Display status
    status_color = {
        "healthy": "green",
        "degraded": "yellow",
        "critical": "red",
        "unknown": "gray",
    }.get(health_summary["status"], "gray")

    console.print(f"Overall Status: [{status_color}]{health_summary['status'].upper()}[/{status_color}]")

    # Show health check results
    table = Table(title="Health Check Results", show_header=True, header_style="bold magenta")
    table.add_column("Check Type", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Message", style="white")

    for result in health_results:
        status_color = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.DEGRADED: "yellow",
            HealthStatus.CRITICAL: "red",
            HealthStatus.UNKNOWN: "gray",
        }.get(result.status, "gray")

        table.add_row(
            result.check_type.value,
            f"[{status_color}]{result.status.value.upper()}[/{status_color}]",
            result.message,
        )

    console.print(table)


@app.command()
def report(
    output_file: str | None = typer.Option(None, "--output", "-o", help="Output file path"),
    report_format: str = typer.Option("text", "--format", "-f", help="Report format (text, html, json)"),
) -> None:
    """Generate a comprehensive system health report."""
    console.print("[bold blue]📋 Generating System Health Report[/bold blue]")

    # Initialize monitoring components
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    health_monitor = SystemHealthMonitor(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
    )

    health_dashboard = HealthDashboard(
        system_health_monitor=health_monitor,
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
    )

    # Generate report based on format
    if report_format == "text":
        if output_file:
            report_content = health_monitor.generate_health_report(output_file)
        else:
            report_content = health_monitor.generate_health_report()
        console.print(report_content)

    elif report_format == "html":
        if output_file:
            html_content = health_dashboard.generate_html_dashboard(output_file)
        else:
            html_content = health_dashboard.generate_html_dashboard()
        console.print(f"HTML dashboard generated: {len(html_content)} characters")

    elif report_format == "json":
        if output_file:
            json_path = health_dashboard.save_dashboard_data(output_file)
        else:
            json_path = health_dashboard.save_dashboard_data()
        console.print(f"JSON report saved to: {json_path}")

    else:
        console.print(f"[red]Unknown format: {report_format}[/red]")
        raise typer.Exit(1)


@app.command()
def alerts(
    severity: str | None = typer.Option(None, "--severity", "-s", help="Filter by severity"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of alerts to show"),
) -> None:
    """Show system health alerts."""
    console.print("[bold blue]🚨 System Health Alerts[/bold blue]")

    # Initialize alert manager
    alert_manager = AlertManager()

    # Get alerts
    if severity:
        try:
            severity_enum = AlertSeverity(severity.lower())
            alerts = alert_manager.get_active_alerts(severity=severity_enum)
        except ValueError:
            console.print(f"[red]Invalid severity: {severity}[/red]")
            raise typer.Exit(1) from None
    else:
        alerts = alert_manager.get_active_alerts()

    if not alerts:
        console.print("[green]No active alerts[/green]")
        return

    # Display alerts
    table = Table(title="Active Alerts", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan")
    table.add_column("Severity", style="green")
    table.add_column("Title", style="white")
    table.add_column("Source", style="yellow")
    table.add_column("Time", style="dim")

    for alert in alerts[:limit]:
        severity_color = {
            "info": "blue",
            "warning": "yellow",
            "error": "red",
            "critical": "red",
        }.get(alert.severity.value, "white")

        table.add_row(
            alert.id[:8] + "...",
            f"[{severity_color}]{alert.severity.value.upper()}[/{severity_color}]",
            alert.title[:50] + "..." if len(alert.title) > 50 else alert.title,
            alert.source,
            time.strftime("%H:%M:%S", time.localtime(alert.timestamp)),
        )

    console.print(table)


if __name__ == "__main__":
    app()
