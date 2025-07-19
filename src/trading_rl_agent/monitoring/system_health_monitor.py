"""
Comprehensive system health monitoring for the trading RL agent.

This module provides comprehensive system health monitoring capabilities including:
- System resource monitoring (CPU, memory, disk, network)
- Trading system performance metrics
- Automated health checks and diagnostics
- System health alerts and notifications
- Health dashboards and reports
"""

import platform
import socket
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Callable

import numpy as np
import psutil


class HealthStatus(Enum):
    """System health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class HealthCheckType(Enum):
    """Types of health checks."""

    SYSTEM_RESOURCES = "system_resources"
    TRADING_PERFORMANCE = "trading_performance"
    NETWORK_CONNECTIVITY = "network_connectivity"
    DATABASE_CONNECTIVITY = "database_connectivity"
    MODEL_PERFORMANCE = "model_performance"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    check_type: HealthCheckType
    status: HealthStatus
    message: str
    timestamp: float
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available: int
    memory_total: int
    disk_percent: float
    disk_free: int
    disk_total: int
    network_bytes_sent: int
    network_bytes_recv: int
    network_packets_sent: int
    network_packets_recv: int
    load_average: tuple[float, float, float]
    process_count: int
    thread_count: int


@dataclass
class TradingMetrics:
    """Trading system performance metrics."""

    timestamp: float
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    open_positions: int
    avg_trade_duration: float
    avg_slippage: float
    execution_latency: float
    order_fill_rate: float


class SystemHealthMonitor:
    """Comprehensive system health monitoring for the trading system."""

    def __init__(
        self,
        metrics_collector: Any = None,
        alert_manager: Any = None,
        check_interval: float = 30.0,
        max_history: int = 1000,
        health_thresholds: dict[str, dict[str, float]] | None = None,
    ) -> None:
        """Initialize the system health monitor.

        Args:
            metrics_collector: Optional MetricsCollector instance
            alert_manager: Optional AlertManager instance
            check_interval: Interval between health checks in seconds
            max_history: Maximum number of health check results to keep
            health_thresholds: Custom health thresholds
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.check_interval = check_interval
        self.max_history = max_history

        # Health check results history
        self.health_history: deque = deque(maxlen=max_history)
        self.system_metrics_history: deque = deque(maxlen=max_history)
        self.trading_metrics_history: deque = deque(maxlen=max_history)

        # Health check functions
        self.health_checks: dict[HealthCheckType, Callable] = {
            HealthCheckType.SYSTEM_RESOURCES: self._check_system_resources,
            HealthCheckType.TRADING_PERFORMANCE: self._check_trading_performance,
            HealthCheckType.NETWORK_CONNECTIVITY: self._check_network_connectivity,
            HealthCheckType.MODEL_PERFORMANCE: self._check_model_performance,
        }

        # Custom health checks
        self.custom_health_checks: dict[str, Callable] = {}

        # Default health thresholds
        self.health_thresholds = health_thresholds or {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_percent": {"warning": 85.0, "critical": 95.0},
            "network_latency": {"warning": 100.0, "critical": 500.0},  # ms
            "error_rate": {"warning": 0.05, "critical": 0.10},  # 5% and 10%
            "execution_latency": {"warning": 100.0, "critical": 500.0},  # ms
            "drawdown": {"warning": 0.05, "critical": 0.10},  # 5% and 10%
        }

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: threading.Thread | None = None
        self.start_time = time.time()

        # Performance tracking
        self.latency_history: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0

        # Initialize system info
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "disk_total": psutil.disk_usage("/").total,
        }

    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        if self.alert_manager:
            self.alert_manager.create_alert(
                title="System Health Monitoring Started",
                message="Continuous health monitoring has been initiated",
                severity=self.alert_manager.AlertSeverity.INFO,
                source="system_health_monitor",
                alert_type="monitoring_started",
            )

    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        if self.alert_manager:
            self.alert_manager.create_alert(
                title="System Health Monitoring Stopped",
                message="Continuous health monitoring has been stopped",
                severity=self.alert_manager.AlertSeverity.INFO,
                source="system_health_monitor",
                alert_type="monitoring_stopped",
            )

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)

                # Run health checks
                health_results = self.run_health_checks()

                # Update metrics collector
                if self.metrics_collector:
                    self._update_metrics_collector(system_metrics, health_results)

                # Check for alerts
                self._check_alerts(system_metrics, health_results)

                # Sleep until next check
                time.sleep(self.check_interval)

            except Exception as e:
                if self.alert_manager:
                    self.alert_manager.create_alert(
                        title="Health Monitoring Error",
                        message=f"Error in health monitoring loop: {e!s}",
                        severity=self.alert_manager.AlertSeverity.ERROR,
                        source="system_health_monitor",
                        alert_type="monitoring_error",
                    )
                time.sleep(self.check_interval)

    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        cpu_percent = psutil.cpu_percent(interval=1.0)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        network = psutil.net_io_counters()
        load_avg = psutil.getloadavg()

        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            memory_total=memory.total,
            disk_percent=disk.percent,
            disk_free=disk.free,
            disk_total=disk.total,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            network_packets_sent=network.packets_sent,
            network_packets_recv=network.packets_recv,
            load_average=load_avg,
            process_count=len(psutil.pids()),
            thread_count=psutil.cpu_count() * 2,  # Approximate
        )

    def run_health_checks(self) -> list[HealthCheckResult]:
        """Run all health checks."""
        results = []

        for check_type, check_func in self.health_checks.items():
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                result = HealthCheckResult(
                    check_type=check_type,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {e!s}",
                    timestamp=time.time(),
                    error=str(e),
                )
                results.append(result)

        # Run custom health checks
        for check_name, check_func in self.custom_health_checks.items():
            try:
                result = check_func()
                result.check_type = HealthCheckType.CUSTOM
                results.append(result)
            except Exception as e:
                result = HealthCheckResult(
                    check_type=HealthCheckType.CUSTOM,
                    status=HealthStatus.UNKNOWN,
                    message=f"Custom health check '{check_name}' failed: {e!s}",
                    timestamp=time.time(),
                    error=str(e),
                )
                results.append(result)

        self.health_history.extend(results)
        return results

    def _check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage."""
        metrics = self._collect_system_metrics()

        # Check CPU usage
        cpu_status = self._evaluate_threshold("cpu_percent", metrics.cpu_percent)

        # Check memory usage
        memory_status = self._evaluate_threshold("memory_percent", metrics.memory_percent)

        # Check disk usage
        disk_status = self._evaluate_threshold("disk_percent", metrics.disk_percent)

        # Determine overall status
        if any(status == HealthStatus.CRITICAL for status in [cpu_status, memory_status, disk_status]):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.DEGRADED for status in [cpu_status, memory_status, disk_status]):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        message = (
            f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%, Disk: {metrics.disk_percent:.1f}%"
        )

        return HealthCheckResult(
            check_type=HealthCheckType.SYSTEM_RESOURCES,
            status=overall_status,
            message=message,
            timestamp=time.time(),
            metrics={
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
                "load_average": metrics.load_average,
                "process_count": metrics.process_count,
            },
        )

    def _check_trading_performance(self) -> HealthCheckResult:
        """Check trading system performance."""
        # Get trading metrics from metrics collector
        trading_metrics = self._get_trading_metrics()

        # Check drawdown
        drawdown_status = self._evaluate_threshold("drawdown", abs(trading_metrics.max_drawdown))

        # Check execution latency
        latency_status = self._evaluate_threshold("execution_latency", trading_metrics.execution_latency)

        # Check error rate
        error_rate = self.error_count / max(self.total_requests, 1)
        error_status = self._evaluate_threshold("error_rate", error_rate)

        # Determine overall status
        if any(status == HealthStatus.CRITICAL for status in [drawdown_status, latency_status, error_status]):
            overall_status = HealthStatus.CRITICAL
        elif any(status == HealthStatus.DEGRADED for status in [drawdown_status, latency_status, error_status]):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        message = f"Drawdown: {trading_metrics.max_drawdown:.2%}, Latency: {trading_metrics.execution_latency:.1f}ms, Error Rate: {error_rate:.2%}"

        return HealthCheckResult(
            check_type=HealthCheckType.TRADING_PERFORMANCE,
            status=overall_status,
            message=message,
            timestamp=time.time(),
            metrics={
                "total_pnl": trading_metrics.total_pnl,
                "daily_pnl": trading_metrics.daily_pnl,
                "sharpe_ratio": trading_metrics.sharpe_ratio,
                "max_drawdown": trading_metrics.max_drawdown,
                "win_rate": trading_metrics.win_rate,
                "execution_latency": trading_metrics.execution_latency,
                "error_rate": error_rate,
            },
        )

    def _check_network_connectivity(self) -> HealthCheckResult:
        """Check network connectivity and latency."""
        try:
            # Test network connectivity
            start_time = time.time()
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            latency_status = self._evaluate_threshold("network_latency", latency)

            message = f"Network latency: {latency:.1f}ms"

            return HealthCheckResult(
                check_type=HealthCheckType.NETWORK_CONNECTIVITY,
                status=latency_status,
                message=message,
                timestamp=time.time(),
                metrics={"network_latency": latency},
            )

        except Exception as e:
            return HealthCheckResult(
                check_type=HealthCheckType.NETWORK_CONNECTIVITY,
                status=HealthStatus.CRITICAL,
                message=f"Network connectivity failed: {e!s}",
                timestamp=time.time(),
                error=str(e),
            )

    def _check_model_performance(self) -> HealthCheckResult:
        """Check model performance metrics."""
        if not self.metrics_collector:
            return HealthCheckResult(
                check_type=HealthCheckType.MODEL_PERFORMANCE,
                status=HealthStatus.UNKNOWN,
                message="No metrics collector available",
                timestamp=time.time(),
            )

        # Get model metrics
        model_accuracy = self.metrics_collector.get_metric_summary("model_accuracy")
        model_loss = self.metrics_collector.get_metric_summary("model_loss")
        prediction_latency = self.metrics_collector.get_metric_summary("prediction_latency")

        # Evaluate model performance
        accuracy = model_accuracy.get("latest", 0.0)
        loss = model_loss.get("latest", float("inf"))
        latency = prediction_latency.get("latest", 0.0)

        # Simple model health evaluation
        if accuracy < 0.5 or loss > 1.0:
            status = HealthStatus.CRITICAL
        elif accuracy < 0.7 or loss > 0.5:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        message = f"Model accuracy: {accuracy:.3f}, Loss: {loss:.3f}, Latency: {latency:.1f}ms"

        return HealthCheckResult(
            check_type=HealthCheckType.MODEL_PERFORMANCE,
            status=status,
            message=message,
            timestamp=time.time(),
            metrics={
                "model_accuracy": accuracy,
                "model_loss": loss,
                "prediction_latency": latency,
            },
        )

    def _evaluate_threshold(self, metric_name: str, value: float) -> HealthStatus:
        """Evaluate a metric against its thresholds."""
        if metric_name not in self.health_thresholds:
            return HealthStatus.UNKNOWN

        thresholds = self.health_thresholds[metric_name]

        if "critical" in thresholds and value >= thresholds["critical"]:
            return HealthStatus.CRITICAL
        if "warning" in thresholds and value >= thresholds["warning"]:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _get_trading_metrics(self) -> TradingMetrics:
        """Get current trading metrics."""
        if self.metrics_collector:
            # Get metrics from collector
            pnl_summary = self.metrics_collector.get_metric_summary("pnl")
            daily_pnl_summary = self.metrics_collector.get_metric_summary("daily_pnl")
            sharpe_summary = self.metrics_collector.get_metric_summary("sharpe_ratio")
            drawdown_summary = self.metrics_collector.get_metric_summary("max_drawdown")
            win_rate_summary = self.metrics_collector.get_metric_summary("win_rate")
            latency_summary = self.metrics_collector.get_metric_summary("execution_latency")

            return TradingMetrics(
                timestamp=time.time(),
                total_pnl=pnl_summary.get("latest", 0.0),
                daily_pnl=daily_pnl_summary.get("latest", 0.0),
                sharpe_ratio=sharpe_summary.get("latest", 0.0),
                max_drawdown=drawdown_summary.get("latest", 0.0),
                win_rate=win_rate_summary.get("latest", 0.0),
                total_trades=self.metrics_collector.get_counter_value("total_trades"),
                open_positions=self.metrics_collector.get_gauge_value("open_positions") or 0,
                avg_trade_duration=0.0,  # Would need to be calculated
                avg_slippage=0.0,  # Would need to be calculated
                execution_latency=latency_summary.get("latest", 0.0),
                order_fill_rate=0.0,  # Would need to be calculated
            )
        # Return default metrics
        return TradingMetrics(
            timestamp=time.time(),
            total_pnl=0.0,
            daily_pnl=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            open_positions=0,
            avg_trade_duration=0.0,
            avg_slippage=0.0,
            execution_latency=0.0,
            order_fill_rate=0.0,
        )

    def _update_metrics_collector(self, system_metrics: SystemMetrics, health_results: list[HealthCheckResult]) -> None:
        """Update metrics collector with system health data."""
        if not self.metrics_collector:
            return

        # Record system metrics
        self.metrics_collector.record_metric("cpu_usage", system_metrics.cpu_percent)
        self.metrics_collector.record_metric("memory_usage", system_metrics.memory_percent)
        self.metrics_collector.record_metric("disk_usage", system_metrics.disk_percent)
        self.metrics_collector.record_metric("network_latency", 0.0)  # Would need to be calculated
        self.metrics_collector.record_metric("load_average", system_metrics.load_average[0])

        # Record health status
        critical_count = sum(1 for result in health_results if result.status == HealthStatus.CRITICAL)
        degraded_count = sum(1 for result in health_results if result.status == HealthStatus.DEGRADED)

        self.metrics_collector.set_gauge("health_critical_checks", critical_count)
        self.metrics_collector.set_gauge("health_degraded_checks", degraded_count)
        self.metrics_collector.set_gauge("health_total_checks", len(health_results))

    def _check_alerts(self, system_metrics: SystemMetrics, health_results: list[HealthCheckResult]) -> None:
        """Check for conditions that require alerts."""
        if not self.alert_manager:
            return

        # Check for critical health issues
        critical_results = [r for r in health_results if r.status == HealthStatus.CRITICAL]
        if critical_results:
            for result in critical_results:
                self.alert_manager.create_alert(
                    title=f"Critical Health Issue: {result.check_type.value}",
                    message=result.message,
                    severity=self.alert_manager.AlertSeverity.CRITICAL,
                    source="system_health_monitor",
                    alert_type="health_critical",
                    metadata=result.metrics,
                )

        # Check for degraded health issues
        degraded_results = [r for r in health_results if r.status == HealthStatus.DEGRADED]
        if degraded_results:
            for result in degraded_results:
                self.alert_manager.create_alert(
                    title=f"Degraded Health: {result.check_type.value}",
                    message=result.message,
                    severity=self.alert_manager.AlertSeverity.WARNING,
                    source="system_health_monitor",
                    alert_type="health_degraded",
                    metadata=result.metrics,
                )

    def add_custom_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Add a custom health check function."""
        self.custom_health_checks[name] = check_func

    def set_health_threshold(self, metric_name: str, threshold_type: str, value: float) -> None:
        """Set a health threshold."""
        if metric_name not in self.health_thresholds:
            self.health_thresholds[metric_name] = {}
        self.health_thresholds[metric_name][threshold_type] = value

    def get_health_summary(self) -> dict[str, Any]:
        """Get a summary of system health."""
        if not self.health_history:
            return {"status": HealthStatus.UNKNOWN.value, "message": "No health checks performed"}

        # Get recent health results
        recent_results = list(self.health_history)[-10:]  # Last 10 results

        # Count statuses
        status_counts: defaultdict[str, int] = defaultdict(int)
        for result in recent_results:
            status_counts[result.status.value] += 1

        # Determine overall status
        if status_counts[HealthStatus.CRITICAL.value] > 0:
            overall_status = HealthStatus.CRITICAL
        elif status_counts[HealthStatus.DEGRADED.value] > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status.value,
            "status_counts": dict(status_counts),
            "total_checks": len(recent_results),
            "last_check": recent_results[-1].timestamp if recent_results else None,
            "uptime": time.time() - self.start_time,
        }

    def get_system_health_dashboard(self) -> dict[str, Any]:
        """Get comprehensive system health dashboard data."""
        # Get current system metrics
        current_metrics = self._collect_system_metrics()

        # Get health summary
        health_summary = self.get_health_summary()

        # Get recent health results
        recent_results = list(self.health_history)[-20:]  # Last 20 results

        # Get trading metrics
        trading_metrics = self._get_trading_metrics()

        return {
            "timestamp": time.time(),
            "system_info": self.system_info,
            "health_summary": health_summary,
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "disk_percent": current_metrics.disk_percent,
                "load_average": current_metrics.load_average,
                "process_count": current_metrics.process_count,
            },
            "trading_metrics": {
                "total_pnl": trading_metrics.total_pnl,
                "daily_pnl": trading_metrics.daily_pnl,
                "sharpe_ratio": trading_metrics.sharpe_ratio,
                "max_drawdown": trading_metrics.max_drawdown,
                "win_rate": trading_metrics.win_rate,
                "total_trades": trading_metrics.total_trades,
                "execution_latency": trading_metrics.execution_latency,
            },
            "recent_health_checks": [
                {
                    "check_type": result.check_type.value,
                    "status": result.status.value,
                    "message": result.message,
                    "timestamp": result.timestamp,
                }
                for result in recent_results
            ],
            "health_thresholds": self.health_thresholds,
        }

    def generate_health_report(self, output_path: str | None = None) -> str:
        """Generate a comprehensive health report."""
        dashboard_data = self.get_system_health_dashboard()

        # Create report content
        report_lines = [
            "=== SYSTEM HEALTH REPORT ===",
            f"Generated: {datetime.fromtimestamp(dashboard_data['timestamp'], tz=UTC)}",
            f"System: {dashboard_data['system_info']['platform']}",
            f"Hostname: {dashboard_data['system_info']['hostname']}",
            "",
            "=== HEALTH SUMMARY ===",
            f"Overall Status: {dashboard_data['health_summary']['status']}",
            f"Uptime: {dashboard_data['health_summary']['uptime']:.1f} seconds",
            f"Total Checks: {dashboard_data['health_summary']['total_checks']}",
            "",
            "=== CURRENT METRICS ===",
            f"CPU Usage: {dashboard_data['current_metrics']['cpu_percent']:.1f}%",
            f"Memory Usage: {dashboard_data['current_metrics']['memory_percent']:.1f}%",
            f"Disk Usage: {dashboard_data['current_metrics']['disk_percent']:.1f}%",
            f"Load Average: {dashboard_data['current_metrics']['load_average'][0]:.2f}",
            f"Process Count: {dashboard_data['current_metrics']['process_count']}",
            "",
            "=== TRADING METRICS ===",
            f"Total P&L: ${dashboard_data['trading_metrics']['total_pnl']:.2f}",
            f"Daily P&L: ${dashboard_data['trading_metrics']['daily_pnl']:.2f}",
            f"Sharpe Ratio: {dashboard_data['trading_metrics']['sharpe_ratio']:.3f}",
            f"Max Drawdown: {dashboard_data['trading_metrics']['max_drawdown']:.2%}",
            f"Win Rate: {dashboard_data['trading_metrics']['win_rate']:.2%}",
            f"Total Trades: {dashboard_data['trading_metrics']['total_trades']}",
            f"Execution Latency: {dashboard_data['trading_metrics']['execution_latency']:.1f}ms",
            "",
            "=== RECENT HEALTH CHECKS ===",
        ]

        for check in dashboard_data["recent_health_checks"][-10:]:  # Last 10 checks
            report_lines.append(f"[{check['status'].upper()}] {check['check_type']}: {check['message']}")

        report_content = "\n".join(report_lines)

        # Save to file if path provided
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_content)

        return report_content

    def record_latency(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latency_history.append(latency_ms)
        if self.metrics_collector:
            self.metrics_collector.record_metric("system_latency", latency_ms)

    def record_error(self) -> None:
        """Record an error occurrence."""
        self.error_count += 1
        if self.metrics_collector:
            self.metrics_collector.increment_counter("system_errors")

    def record_request(self) -> None:
        """Record a request."""
        self.total_requests += 1
        if self.metrics_collector:
            self.metrics_collector.increment_counter("total_requests")

    def get_average_latency(self) -> float:
        """Get average latency."""
        if not self.latency_history:
            return 0.0
        return float(np.mean(self.latency_history))

    def get_error_rate(self) -> float:
        """Get current error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests
