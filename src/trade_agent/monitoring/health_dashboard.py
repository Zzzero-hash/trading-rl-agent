"""
Comprehensive health dashboard for the trading system.

This module provides a dedicated health dashboard that integrates with the existing
monitoring framework to provide real-time system health visualization.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class HealthDashboard:
    """Comprehensive health dashboard for system monitoring."""

    def __init__(
        self,
        system_health_monitor: Any,
        metrics_collector: Any = None,
        alert_manager: Any = None,
        output_dir: str = "health_reports",
    ) -> None:
        """Initialize the health dashboard.

        Args:
            system_health_monitor: SystemHealthMonitor instance
            metrics_collector: Optional MetricsCollector instance
            alert_manager: Optional AlertManager instance
            output_dir: Directory to save health reports and dashboards
        """
        self.system_health_monitor = system_health_monitor
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Dashboard state
        self.last_update = time.time()
        self.update_interval = 5.0  # seconds

    def generate_system_health_overview(self) -> dict[str, Any]:
        """Generate comprehensive system health overview."""
        dashboard_data = self.system_health_monitor.get_system_health_dashboard()

        # Add additional metrics if available
        if self.metrics_collector:
            # Get recent metrics trends
            cpu_trend = self._get_metric_trend("cpu_usage", hours=24)
            memory_trend = self._get_metric_trend("memory_usage", hours=24)
            disk_trend = self._get_metric_trend("disk_usage", hours=24)

            dashboard_data["metric_trends"] = {
                "cpu_usage": cpu_trend,
                "memory_usage": memory_trend,
                "disk_usage": disk_trend,
            }

        # Add alert information
        if self.alert_manager:
            active_alerts = self.alert_manager.get_active_alerts()
            dashboard_data["active_alerts"] = [
                {
                    "id": alert.id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "timestamp": alert.timestamp,
                }
                for alert in active_alerts[:10]  # Last 10 alerts
            ]

        return dict(dashboard_data)

    def create_health_visualization(self, save_path: str | None = None) -> Figure:
        """Create a comprehensive health visualization."""
        dashboard_data = self.system_health_monitor.get_system_health_dashboard()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("System Health Dashboard", fontsize=16, fontweight="bold")

        # System metrics
        self._plot_system_metrics(axes[0, 0], dashboard_data)
        self._plot_memory_usage(axes[0, 1], dashboard_data)
        self._plot_disk_usage(axes[0, 2], dashboard_data)

        # Trading metrics
        self._plot_trading_metrics(axes[1, 0], dashboard_data)
        self._plot_health_status(axes[1, 1], dashboard_data)
        self._plot_recent_checks(axes[1, 2], dashboard_data)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def _plot_system_metrics(self, ax: Any, dashboard_data: dict[str, Any]) -> None:
        """Plot system metrics."""
        metrics = dashboard_data["current_metrics"]

        # Create gauge-style plot for CPU usage
        cpu_percent = metrics["cpu_percent"]
        ax.clear()

        # Create gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Background circle
        ax.plot(x, y, "k-", linewidth=2)

        # CPU usage arc
        cpu_angle = (cpu_percent / 100) * np.pi
        theta_cpu = np.linspace(0, cpu_angle, 50)
        x_cpu = r * np.cos(theta_cpu)
        y_cpu = r * np.sin(theta_cpu)

        # Color based on usage
        if cpu_percent > 90:
            color = "red"
        elif cpu_percent > 70:
            color = "orange"
        else:
            color = "green"

        ax.plot(x_cpu, y_cpu, color=color, linewidth=8)

        # Add text
        ax.text(
            0,
            0,
            f"{cpu_percent:.1f}%",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )
        ax.text(0, -1.5, "CPU Usage", ha="center", va="center", fontsize=12)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.5, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")

    def _plot_memory_usage(self, ax: Any, dashboard_data: dict[str, Any]) -> None:
        """Plot memory usage."""
        metrics = dashboard_data["current_metrics"]
        memory_percent = metrics["memory_percent"]

        ax.clear()

        # Create bar chart
        bars = ax.bar(["Memory"], [memory_percent], color="skyblue", alpha=0.7)

        # Color based on usage
        if memory_percent > 90:
            bars[0].set_color("red")
        elif memory_percent > 70:
            bars[0].set_color("orange")

        ax.set_ylim(0, 100)
        ax.set_ylabel("Usage (%)")
        ax.set_title("Memory Usage")
        ax.grid(True, alpha=0.3)

        # Add value label
        ax.text(
            0,
            memory_percent + 2,
            f"{memory_percent:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    def _plot_disk_usage(self, ax: Any, dashboard_data: dict[str, Any]) -> None:
        """Plot disk usage."""
        metrics = dashboard_data["current_metrics"]
        disk_percent = metrics["disk_percent"]

        ax.clear()

        # Create pie chart
        sizes = [disk_percent, 100 - disk_percent]
        colors = ["lightcoral" if disk_percent > 90 else "lightgreen", "lightgray"]
        labels = ["Used", "Free"]

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Disk Usage")

    def _plot_trading_metrics(self, ax: Any, dashboard_data: dict[str, Any]) -> None:
        """Plot trading metrics."""
        trading_metrics = dashboard_data["trading_metrics"]

        ax.clear()

        # Create multiple metrics display
        metrics_data = [
            ("Total P&L", f"${trading_metrics['total_pnl']:.2f}"),
            ("Daily P&L", f"${trading_metrics['daily_pnl']:.2f}"),
            ("Sharpe Ratio", f"{trading_metrics['sharpe_ratio']:.3f}"),
            ("Win Rate", f"{trading_metrics['win_rate']:.1%}"),
            ("Total Trades", str(trading_metrics["total_trades"])),
            ("Latency", f"{trading_metrics['execution_latency']:.1f}ms"),
        ]

        y_pos = np.arange(len(metrics_data))
        values = [
            (
                float(metric[1].replace("$", "").replace("%", "").replace("ms", ""))
                if metric[1] not in ["N/A", "0"]
                else 0
            )
            for metric in metrics_data
        ]

        bars = ax.barh(y_pos, values, color="lightblue", alpha=0.7)

        # Color bars based on values
        for i, (metric_name, metric_value) in enumerate(metrics_data):
            if "P&L" in metric_name and float(metric_value.replace("$", "")) < 0:
                bars[i].set_color("lightcoral")
            elif "Latency" in metric_name and float(metric_value.replace("ms", "")) > 100:
                bars[i].set_color("orange")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([metric[0] for metric in metrics_data])
        ax.set_xlabel("Value")
        ax.set_title("Trading Metrics")
        ax.grid(True, alpha=0.3)

        # Add value labels
        for i, (metric_name, metric_value) in enumerate(metrics_data):
            ax.text(
                values[i] + max(values) * 0.01,
                i,
                metric_value,
                va="center",
                fontweight="bold",
            )

    def _plot_health_status(self, ax: Any, dashboard_data: dict[str, Any]) -> None:
        """Plot health status summary."""
        health_summary = dashboard_data["health_summary"]
        status_counts = health_summary.get("status_counts", {})

        ax.clear()

        # Create pie chart of health statuses
        labels = list(status_counts.keys())
        sizes = list(status_counts.values())
        colors = ["green", "orange", "red", "gray"]

        if sizes:
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors[: len(sizes)],
                autopct="%1.0f",
                startangle=90,
            )
            ax.set_title("Health Status Distribution")
        else:
            ax.text(
                0.5,
                0.5,
                "No health checks\nperformed",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Health Status")

    def _plot_recent_checks(self, ax: Any, dashboard_data: dict[str, Any]) -> None:
        """Plot recent health checks timeline."""
        recent_checks = dashboard_data.get("recent_health_checks", [])

        ax.clear()

        if not recent_checks:
            ax.text(
                0.5,
                0.5,
                "No recent health checks",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Recent Health Checks")
            return

        # Convert timestamps to datetime
        timestamps = [datetime.fromtimestamp(check["timestamp"], tz=UTC) for check in recent_checks]
        statuses = [check["status"] for check in recent_checks]

        # Create status mapping
        status_map = {"healthy": 0, "degraded": 1, "critical": 2, "unknown": 3}
        status_values = [status_map.get(status, 3) for status in statuses]

        # Create scatter plot
        colors = ["green", "orange", "red", "gray"]
        for i, (timestamp, status_value) in enumerate(zip(timestamps, status_values, strict=False)):
            ax.scatter(timestamp, status_value, c=colors[status_value], s=100, alpha=0.7)

        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(["Healthy", "Degraded", "Critical", "Unknown"])
        ax.set_xlabel("Time")
        ax.set_title("Recent Health Checks")
        ax.grid(True, alpha=0.3)

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    def _get_metric_trend(self, metric_name: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get metric trend data."""
        if not self.metrics_collector:
            return []

        end_time = time.time()
        start_time = end_time - (hours * 3600)

        history = self.metrics_collector.get_metric_history(metric_name, start_time, end_time)

        return [
            {
                "timestamp": point.timestamp,
                "value": point.value,
            }
            for point in history
        ]

    def generate_html_dashboard(self, output_path: str | None = None) -> str:
        """Generate an HTML dashboard."""
        dashboard_data = self.generate_system_health_overview()

        # Create HTML content
        html_content = self._create_html_content(dashboard_data)

        if output_path:
            with open(output_path, "w") as f:
                f.write(html_content)

        return html_content

    def _create_html_content(self, dashboard_data: dict[str, Any]) -> str:
        """Create HTML dashboard content."""
        current_time = datetime.fromtimestamp(dashboard_data["timestamp"], tz=UTC)

        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>System Health Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .card {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 10px;
                    padding: 8px;
                    background-color: #f8f9fa;
                    border-radius: 4px;
                }}
                .status-healthy {{ color: #28a745; font-weight: bold; }}
                .status-degraded {{ color: #ffc107; font-weight: bold; }}
                .status-critical {{ color: #dc3545; font-weight: bold; }}
                .status-unknown {{ color: #6c757d; font-weight: bold; }}
                .alert {{
                    padding: 10px;
                    margin-bottom: 10px;
                    border-radius: 4px;
                    border-left: 4px solid;
                }}
                .alert-info {{ background-color: #d1ecf1; border-color: #17a2b8; }}
                .alert-warning {{ background-color: #fff3cd; border-color: #ffc107; }}
                .alert-error {{ background-color: #f8d7da; border-color: #dc3545; }}
                .alert-critical {{ background-color: #f5c6cb; border-color: #dc3545; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>System Health Dashboard</h1>
                <p>Last updated: {current_time.strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Overall Status: <span class="status-{dashboard_data["health_summary"]["status"]}">
                    {dashboard_data["health_summary"]["status"].upper()}</span></p>
            </div>

            <div class="dashboard-grid">
                <div class="card">
                    <h2>System Metrics</h2>
                    <div class="metric">
                        <span>CPU Usage:</span>
                        <span>{dashboard_data["current_metrics"]["cpu_percent"]:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Memory Usage:</span>
                        <span>{dashboard_data["current_metrics"]["memory_percent"]:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Disk Usage:</span>
                        <span>{dashboard_data["current_metrics"]["disk_percent"]:.1f}%</span>
                    </div>
                    <div class="metric">
                        <span>Load Average:</span>
                        <span>{dashboard_data["current_metrics"]["load_average"][0]:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Process Count:</span>
                        <span>{dashboard_data["current_metrics"]["process_count"]}</span>
                    </div>
                </div>

                <div class="card">
                    <h2>Trading Metrics</h2>
                    <div class="metric">
                        <span>Total P&L:</span>
                        <span>${dashboard_data["trading_metrics"]["total_pnl"]:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Daily P&L:</span>
                        <span>${dashboard_data["trading_metrics"]["daily_pnl"]:.2f}</span>
                    </div>
                    <div class="metric">
                        <span>Sharpe Ratio:</span>
                        <span>{dashboard_data["trading_metrics"]["sharpe_ratio"]:.3f}</span>
                    </div>
                    <div class="metric">
                        <span>Max Drawdown:</span>
                        <span>{dashboard_data["trading_metrics"]["max_drawdown"]:.2%}</span>
                    </div>
                    <div class="metric">
                        <span>Win Rate:</span>
                        <span>{dashboard_data["trading_metrics"]["win_rate"]:.2%}</span>
                    </div>
                    <div class="metric">
                        <span>Total Trades:</span>
                        <span>{dashboard_data["trading_metrics"]["total_trades"]}</span>
                    </div>
                    <div class="metric">
                        <span>Execution Latency:</span>
                        <span>{dashboard_data["trading_metrics"]["execution_latency"]:.1f}ms</span>
                    </div>
                </div>

                <div class="card">
                    <h2>System Information</h2>
                    <div class="metric">
                        <span>Platform:</span>
                        <span>{dashboard_data["system_info"]["platform"]}</span>
                    </div>
                    <div class="metric">
                        <span>Hostname:</span>
                        <span>{dashboard_data["system_info"]["hostname"]}</span>
                    </div>
                    <div class="metric">
                        <span>CPU Count:</span>
                        <span>{dashboard_data["system_info"]["cpu_count"]}</span>
                    </div>
                    <div class="metric">
                        <span>Memory Total:</span>
                        <span>{dashboard_data["system_info"]["memory_total"] / (1024**3):.1f} GB</span>
                    </div>
                    <div class="metric">
                        <span>Disk Total:</span>
                        <span>{dashboard_data["system_info"]["disk_total"] / (1024**3):.1f} GB</span>
                    </div>
                </div>
            </div>

            <div class="card">
                <h2>Recent Health Checks</h2>
                {self._create_health_checks_html(dashboard_data.get("recent_health_checks", []))}
            </div>

            {self._create_alerts_html(dashboard_data.get("active_alerts", []))}
        </body>
        </html>
        """

    def _create_health_checks_html(self, health_checks: list[dict[str, Any]]) -> str:
        """Create HTML for health checks."""
        if not health_checks:
            return "<p>No recent health checks available.</p>"

        html = ""
        for check in health_checks[-10:]:  # Last 10 checks
            timestamp = datetime.fromtimestamp(check["timestamp"], tz=UTC)
            html += f"""
            <div class="metric">
                <span>[{check["status"].upper()}] {check["check_type"]}</span>
                <span>{timestamp.strftime("%H:%M:%S")}</span>
            </div>
            <div style="margin-left: 20px; margin-bottom: 10px; color: #666;">
                {check["message"]}
            </div>
            """

        return html

    def _create_alerts_html(self, alerts: list[dict[str, Any]]) -> str:
        """Create HTML for alerts."""
        if not alerts:
            return ""

        html = '<div class="card"><h2>Active Alerts</h2>'

        for alert in alerts:
            severity_class = f"alert-{alert['severity']}"
            timestamp = datetime.fromtimestamp(alert["timestamp"], tz=UTC)
            html += f"""
            <div class="alert {severity_class}">
                <strong>{alert["title"]}</strong><br>
                <small>{timestamp.strftime("%Y-%m-%d %H:%M:%S")}</small>
            </div>
            """

        html += "</div>"
        return html

    def save_dashboard_data(self, filename: str | None = None) -> str:
        """Save dashboard data to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_dashboard_{timestamp}.json"

        filepath = self.output_dir / filename
        dashboard_data = self.generate_system_health_overview()

        with open(filepath, "w") as f:
            json.dump(dashboard_data, f, indent=2, default=str)

        return str(filepath)

    def update(self) -> None:
        """Update the dashboard."""
        self.last_update = time.time()

    def should_update(self) -> bool:
        """Check if dashboard should be updated."""
        return time.time() - self.last_update >= self.update_interval

    def set_update_interval(self, interval: float) -> None:
        """Set the update interval."""
        self.update_interval = interval
