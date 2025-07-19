"""
Tests for the SystemHealthMonitor module.

This module tests the comprehensive system health monitoring functionality
including health checks, alerts, dashboards, and integration.
"""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.trading_rl_agent.monitoring import (
    AlertManager,
    HealthCheckResult,
    HealthCheckType,
    HealthDashboard,
    HealthStatus,
    MetricsCollector,
    SystemHealthMonitor,
    SystemMetrics,
    TradingMetrics,
)


class TestSystemHealthMonitor:
    """Test cases for SystemHealthMonitor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.metrics_collector = Mock(spec=MetricsCollector)
        self.alert_manager = Mock(spec=AlertManager)
        self.health_monitor = SystemHealthMonitor(
            metrics_collector=self.metrics_collector,
            alert_manager=self.alert_manager,
            check_interval=1.0,
            max_history=100,
        )

    def test_initialization(self):
        """Test SystemHealthMonitor initialization."""
        assert self.health_monitor.metrics_collector == self.metrics_collector
        assert self.health_monitor.alert_manager == self.alert_manager
        assert self.health_monitor.check_interval == 1.0
        assert self.health_monitor.max_history == 100
        assert not self.health_monitor.is_monitoring
        assert self.health_monitor.error_count == 0
        assert self.health_monitor.total_requests == 0

    def test_get_system_info(self):
        """Test system information collection."""
        system_info = self.health_monitor._get_system_info()

        assert "platform" in system_info
        assert "python_version" in system_info
        assert "hostname" in system_info
        assert "cpu_count" in system_info
        assert "memory_total" in system_info
        assert "disk_total" in system_info

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.net_io_counters")
    @patch("psutil.getloadavg")
    @patch("psutil.pids")
    def test_collect_system_metrics(self, mock_pids, mock_load, mock_net, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock psutil responses
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(percent=60.0, available=1024**3, total=2 * 1024**3)
        mock_disk.return_value = Mock(percent=45.0, free=500 * 1024**3, total=1000 * 1024**3)
        mock_net.return_value = Mock(bytes_sent=1024**2, bytes_recv=2 * 1024**2, packets_sent=100, packets_recv=200)
        mock_load.return_value = (1.5, 1.2, 1.0)
        mock_pids.return_value = [1, 2, 3, 4, 5]

        metrics = self.health_monitor._collect_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 60.0
        assert metrics.disk_percent == 45.0
        assert metrics.process_count == 5

    def test_evaluate_threshold(self):
        """Test threshold evaluation."""
        # Test with default thresholds
        self.health_monitor.health_thresholds = {"cpu_percent": {"warning": 80.0, "critical": 95.0}}

        # Test healthy
        status = self.health_monitor._evaluate_threshold("cpu_percent", 50.0)
        assert status == HealthStatus.HEALTHY

        # Test warning
        status = self.health_monitor._evaluate_threshold("cpu_percent", 85.0)
        assert status == HealthStatus.DEGRADED

        # Test critical
        status = self.health_monitor._evaluate_threshold("cpu_percent", 98.0)
        assert status == HealthStatus.CRITICAL

        # Test unknown metric
        status = self.health_monitor._evaluate_threshold("unknown_metric", 50.0)
        assert status == HealthStatus.UNKNOWN

    def test_check_system_resources(self):
        """Test system resources health check."""
        with patch.object(self.health_monitor, "_collect_system_metrics") as mock_collect:
            mock_collect.return_value = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=25.0,
                memory_percent=60.0,
                memory_available=1024**3,
                memory_total=2 * 1024**3,
                disk_percent=45.0,
                disk_free=500 * 1024**3,
                disk_total=1000 * 1024**3,
                network_bytes_sent=1024**2,
                network_bytes_recv=2 * 1024**2,
                network_packets_sent=100,
                network_packets_recv=200,
                load_average=(1.5, 1.2, 1.0),
                process_count=5,
                thread_count=8,
            )

            result = self.health_monitor._check_system_resources()

            assert isinstance(result, HealthCheckResult)
            assert result.check_type == HealthCheckType.SYSTEM_RESOURCES
            assert result.status == HealthStatus.HEALTHY
            assert "CPU: 25.0%" in result.message
            assert "Memory: 60.0%" in result.message
            assert "Disk: 45.0%" in result.message

    def test_check_trading_performance(self):
        """Test trading performance health check."""
        with patch.object(self.health_monitor, "_get_trading_metrics") as mock_get_trading:
            mock_get_trading.return_value = TradingMetrics(
                timestamp=time.time(),
                total_pnl=1000.0,
                daily_pnl=100.0,
                sharpe_ratio=1.5,
                max_drawdown=-0.05,
                win_rate=0.65,
                total_trades=50,
                open_positions=3,
                avg_trade_duration=300.0,
                avg_slippage=0.001,
                execution_latency=50.0,
                order_fill_rate=0.95,
            )

            result = self.health_monitor._check_trading_performance()

            assert isinstance(result, HealthCheckResult)
            assert result.check_type == HealthCheckType.TRADING_PERFORMANCE
            assert "Drawdown: -5.00%" in result.message
            assert "Latency: 50.0ms" in result.message

    @patch("socket.create_connection")
    def test_check_network_connectivity_success(self, mock_socket):
        """Test network connectivity check with success."""
        mock_socket.return_value = Mock()

        result = self.health_monitor._check_network_connectivity()

        assert isinstance(result, HealthCheckResult)
        assert result.check_type == HealthCheckType.NETWORK_CONNECTIVITY
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "Network latency:" in result.message

    @patch("socket.create_connection")
    def test_check_network_connectivity_failure(self, mock_socket):
        """Test network connectivity check with failure."""
        mock_socket.side_effect = Exception("Connection failed")

        result = self.health_monitor._check_network_connectivity()

        assert isinstance(result, HealthCheckResult)
        assert result.check_type == HealthCheckType.NETWORK_CONNECTIVITY
        assert result.status == HealthStatus.CRITICAL
        assert "Network connectivity failed" in result.message

    def test_check_model_performance(self):
        """Test model performance health check."""
        # Mock metrics collector responses
        self.metrics_collector.get_metric_summary.side_effect = lambda name: {
            "model_accuracy": {"latest": 0.85},
            "model_loss": {"latest": 0.3},
            "prediction_latency": {"latest": 25.0},
        }.get(name, {"latest": None})

        result = self.health_monitor._check_model_performance()

        assert isinstance(result, HealthCheckResult)
        assert result.check_type == HealthCheckType.MODEL_PERFORMANCE
        assert result.status == HealthStatus.HEALTHY
        assert "Model accuracy: 0.850" in result.message

    def test_get_trading_metrics_with_collector(self):
        """Test getting trading metrics with metrics collector."""
        # Mock metrics collector responses
        self.metrics_collector.get_metric_summary.side_effect = lambda name: {
            "pnl": {"latest": 1000.0},
            "daily_pnl": {"latest": 100.0},
            "sharpe_ratio": {"latest": 1.5},
            "max_drawdown": {"latest": -0.05},
            "win_rate": {"latest": 0.65},
            "execution_latency": {"latest": 50.0},
        }.get(name, {"latest": 0.0})

        self.metrics_collector.get_counter_value.return_value = 50
        self.metrics_collector.get_gauge_value.return_value = 3

        metrics = self.health_monitor._get_trading_metrics()

        assert isinstance(metrics, TradingMetrics)
        assert metrics.total_pnl == 1000.0
        assert metrics.daily_pnl == 100.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_trades == 50
        assert metrics.open_positions == 3

    def test_get_trading_metrics_without_collector(self):
        """Test getting trading metrics without metrics collector."""
        self.health_monitor.metrics_collector = None

        metrics = self.health_monitor._get_trading_metrics()

        assert isinstance(metrics, TradingMetrics)
        assert metrics.total_pnl == 0.0
        assert metrics.daily_pnl == 0.0
        assert metrics.total_trades == 0

    def test_run_health_checks(self):
        """Test running all health checks."""
        with patch.object(self.health_monitor, "_check_system_resources") as mock_system:
            with patch.object(self.health_monitor, "_check_trading_performance") as mock_trading:
                with patch.object(self.health_monitor, "_check_network_connectivity") as mock_network:
                    with patch.object(self.health_monitor, "_check_model_performance") as mock_model:
                        # Mock health check results
                        mock_system.return_value = HealthCheckResult(
                            check_type=HealthCheckType.SYSTEM_RESOURCES,
                            status=HealthStatus.HEALTHY,
                            message="System healthy",
                            timestamp=time.time(),
                        )
                        mock_trading.return_value = HealthCheckResult(
                            check_type=HealthCheckType.TRADING_PERFORMANCE,
                            status=HealthStatus.HEALTHY,
                            message="Trading healthy",
                            timestamp=time.time(),
                        )
                        mock_network.return_value = HealthCheckResult(
                            check_type=HealthCheckType.NETWORK_CONNECTIVITY,
                            status=HealthStatus.HEALTHY,
                            message="Network healthy",
                            timestamp=time.time(),
                        )
                        mock_model.return_value = HealthCheckResult(
                            check_type=HealthCheckType.MODEL_PERFORMANCE,
                            status=HealthStatus.HEALTHY,
                            message="Model healthy",
                            timestamp=time.time(),
                        )

                        results = self.health_monitor.run_health_checks()

                        assert len(results) == 4
                        assert all(isinstance(r, HealthCheckResult) for r in results)
                        assert len(self.health_monitor.health_history) == 4

    def test_add_custom_health_check(self):
        """Test adding custom health checks."""

        def custom_check():
            return HealthCheckResult(
                check_type=HealthCheckType.CUSTOM,
                status=HealthStatus.HEALTHY,
                message="Custom check passed",
                timestamp=time.time(),
            )

        self.health_monitor.add_custom_health_check("custom_test", custom_check)

        assert "custom_test" in self.health_monitor.custom_health_checks
        assert self.health_monitor.custom_health_checks["custom_test"] == custom_check

    def test_set_health_threshold(self):
        """Test setting health thresholds."""
        self.health_monitor.set_health_threshold("test_metric", "warning", 75.0)
        self.health_monitor.set_health_threshold("test_metric", "critical", 90.0)

        assert "test_metric" in self.health_monitor.health_thresholds
        assert self.health_monitor.health_thresholds["test_metric"]["warning"] == 75.0
        assert self.health_monitor.health_thresholds["test_metric"]["critical"] == 90.0

    def test_get_health_summary(self):
        """Test getting health summary."""
        # Add some health check results
        self.health_monitor.health_history.extend(
            [
                HealthCheckResult(
                    check_type=HealthCheckType.SYSTEM_RESOURCES,
                    status=HealthStatus.HEALTHY,
                    message="Healthy",
                    timestamp=time.time(),
                ),
                HealthCheckResult(
                    check_type=HealthCheckType.TRADING_PERFORMANCE,
                    status=HealthStatus.DEGRADED,
                    message="Degraded",
                    timestamp=time.time(),
                ),
            ]
        )

        summary = self.health_monitor.get_health_summary()

        assert "status" in summary
        assert "status_counts" in summary
        assert "total_checks" in summary
        assert summary["total_checks"] == 2
        assert summary["status_counts"]["healthy"] == 1
        assert summary["status_counts"]["degraded"] == 1

    def test_record_latency_and_errors(self):
        """Test recording latency and errors."""
        self.health_monitor.record_latency(50.0)
        self.health_monitor.record_latency(75.0)
        self.health_monitor.record_request()
        self.health_monitor.record_request()
        self.health_monitor.record_error()

        assert len(self.health_monitor.latency_history) == 2
        assert self.health_monitor.total_requests == 2
        assert self.health_monitor.error_count == 1
        assert self.health_monitor.get_average_latency() == 62.5
        assert self.health_monitor.get_error_rate() == 0.5

    def test_get_system_health_dashboard(self):
        """Test getting system health dashboard data."""
        with patch.object(self.health_monitor, "_collect_system_metrics") as mock_collect:
            mock_collect.return_value = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=25.0,
                memory_percent=60.0,
                memory_available=1024**3,
                memory_total=2 * 1024**3,
                disk_percent=45.0,
                disk_free=500 * 1024**3,
                disk_total=1000 * 1024**3,
                network_bytes_sent=1024**2,
                network_bytes_recv=2 * 1024**2,
                network_packets_sent=100,
                network_packets_recv=200,
                load_average=(1.5, 1.2, 1.0),
                process_count=5,
                thread_count=8,
            )

            dashboard_data = self.health_monitor.get_system_health_dashboard()

            assert "timestamp" in dashboard_data
            assert "system_info" in dashboard_data
            assert "health_summary" in dashboard_data
            assert "current_metrics" in dashboard_data
            assert "trading_metrics" in dashboard_data
            assert "recent_health_checks" in dashboard_data
            assert "health_thresholds" in dashboard_data

    def test_generate_health_report(self):
        """Test generating health report."""
        with patch.object(self.health_monitor, "get_system_health_dashboard") as mock_dashboard:
            mock_dashboard.return_value = {
                "timestamp": time.time(),
                "system_info": {
                    "platform": "Linux-5.4.0-x86_64",
                    "hostname": "test-host",
                },
                "health_summary": {
                    "status": "healthy",
                    "uptime": 3600.0,
                    "total_checks": 5,
                },
                "current_metrics": {
                    "cpu_percent": 25.0,
                    "memory_percent": 60.0,
                    "disk_percent": 45.0,
                    "load_average": (1.5, 1.2, 1.0),
                    "process_count": 5,
                },
                "trading_metrics": {
                    "total_pnl": 1000.0,
                    "daily_pnl": 100.0,
                    "sharpe_ratio": 1.5,
                    "max_drawdown": -0.05,
                    "win_rate": 0.65,
                    "total_trades": 50,
                    "execution_latency": 50.0,
                },
                "recent_health_checks": [
                    {
                        "check_type": "system_resources",
                        "status": "healthy",
                        "message": "System healthy",
                        "timestamp": time.time(),
                    }
                ],
            }

            report = self.health_monitor.generate_health_report()

            assert "SYSTEM HEALTH REPORT" in report
            assert "Overall Status: healthy" in report
            assert "CPU Usage: 25.0%" in report
            assert "Total P&L: $1000.00" in report


class TestHealthDashboard:
    """Test cases for HealthDashboard."""

    def setup_method(self):
        """Set up test fixtures."""
        self.health_monitor = Mock(spec=SystemHealthMonitor)
        self.metrics_collector = Mock(spec=MetricsCollector)
        self.alert_manager = Mock(spec=AlertManager)

        self.dashboard = HealthDashboard(
            system_health_monitor=self.health_monitor,
            metrics_collector=self.metrics_collector,
            alert_manager=self.alert_manager,
            output_dir="test_output",
        )

    def test_initialization(self):
        """Test HealthDashboard initialization."""
        assert self.dashboard.system_health_monitor == self.health_monitor
        assert self.dashboard.metrics_collector == self.metrics_collector
        assert self.dashboard.alert_manager == self.alert_manager
        assert self.dashboard.output_dir == Path("test_output")
        assert self.dashboard.update_interval == 5.0

    def test_generate_system_health_overview(self):
        """Test generating system health overview."""
        # Mock health monitor dashboard data
        self.health_monitor.get_system_health_dashboard.return_value = {
            "timestamp": time.time(),
            "health_summary": {"status": "healthy"},
            "current_metrics": {"cpu_percent": 25.0},
            "trading_metrics": {"total_pnl": 1000.0},
        }

        # Mock metrics collector
        self.metrics_collector.get_metric_history.return_value = []

        # Mock alert manager
        self.alert_manager.get_active_alerts.return_value = []

        overview = self.dashboard.generate_system_health_overview()

        assert "timestamp" in overview
        assert "health_summary" in overview
        assert "current_metrics" in overview
        assert "trading_metrics" in overview
        assert "metric_trends" in overview
        assert "active_alerts" in overview

    def test_save_dashboard_data(self):
        """Test saving dashboard data."""
        with patch.object(self.dashboard, "generate_system_health_overview") as mock_overview:
            mock_overview.return_value = {"test": "data"}

            with patch("builtins.open", create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file

                filepath = self.dashboard.save_dashboard_data("test.json")

                assert "test.json" in filepath
                mock_file.write.assert_called_once()

    def test_should_update(self):
        """Test update interval checking."""
        self.dashboard.last_update = time.time() - 10.0  # 10 seconds ago
        self.dashboard.update_interval = 5.0

        assert self.dashboard.should_update()

        self.dashboard.last_update = time.time() - 2.0  # 2 seconds ago

        assert not self.dashboard.should_update()


if __name__ == "__main__":
    pytest.main([__file__])
