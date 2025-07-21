# System Health Monitoring

The System Health Monitor provides comprehensive monitoring capabilities for the trading RL agent, including system resource monitoring, trading performance tracking, automated health checks, and real-time alerts.

## Features

### 🔍 System Resource Monitoring

- **CPU Usage**: Real-time CPU utilization monitoring with configurable thresholds
- **Memory Usage**: Memory consumption tracking with available/total memory metrics
- **Disk Usage**: Disk space monitoring with free/total space tracking
- **Network Metrics**: Network I/O statistics and connectivity testing
- **Process Monitoring**: Active process and thread count tracking
- **Load Average**: System load monitoring across different time periods

### 📈 Trading Performance Metrics

- **P&L Tracking**: Total and daily profit/loss monitoring
- **Risk Metrics**: Sharpe ratio, maximum drawdown, and win rate tracking
- **Execution Metrics**: Trade execution latency and order fill rates
- **Position Monitoring**: Open positions and trade count tracking
- **Performance Alerts**: Automated alerts for performance degradation

### 🏥 Automated Health Checks

- **System Resources**: Automated checks for CPU, memory, and disk usage
- **Trading Performance**: Performance threshold monitoring and alerts
- **Network Connectivity**: Network latency and connectivity testing
- **Model Performance**: ML model accuracy, loss, and prediction latency monitoring
- **Custom Health Checks**: Extensible framework for custom health checks

### 🚨 Alert System

- **Multi-level Alerts**: Info, Warning, Error, and Critical alert levels
- **Configurable Thresholds**: Customizable thresholds for all monitored metrics
- **Alert Handlers**: Custom alert handlers for different alert types
- **Alert Management**: Alert acknowledgment, resolution, and dismissal
- **Real-time Notifications**: Immediate alert generation for critical issues

### 📊 Health Dashboards

- **Real-time Dashboard**: Live system health visualization
- **HTML Dashboards**: Web-based dashboard with interactive elements
- **JSON Reports**: Structured data export for external monitoring systems
- **Text Reports**: Human-readable health reports
- **Historical Data**: Trend analysis and historical health data

## Quick Start

### Basic Usage

```python
from src.trading_rl_agent.monitoring import (
    SystemHealthMonitor,
    MetricsCollector,
    AlertManager,
    HealthDashboard
)

# Initialize monitoring components
metrics_collector = MetricsCollector()
alert_manager = AlertManager()
health_monitor = SystemHealthMonitor(
    metrics_collector=metrics_collector,
    alert_manager=alert_manager,
    check_interval=30.0  # Check every 30 seconds
)

# Start monitoring
health_monitor.start_monitoring()

# Get health status
health_summary = health_monitor.get_health_summary()
print(f"System Status: {health_summary['status']}")

# Generate health report
report = health_monitor.generate_health_report("health_report.txt")
```

### CLI Usage

```bash
# Start monitoring for 5 minutes with live dashboard
python -m src.trading_rl_agent.cli_health monitor --duration 300 --live

# Show current system health status
python -m src.trading_rl_agent.cli_health status

# Generate health report
python -m src.trading_rl_agent.cli_health report --format html --output dashboard.html

# Show active alerts
python -m src.trading_rl_agent.cli_health alerts --severity critical
```

## Configuration

### Health Thresholds

Configure custom thresholds for different metrics:

```python
health_thresholds = {
    "cpu_percent": {"warning": 80.0, "critical": 95.0},
    "memory_percent": {"warning": 85.0, "critical": 95.0},
    "disk_percent": {"warning": 85.0, "critical": 95.0},
    "network_latency": {"warning": 100.0, "critical": 500.0},  # ms
    "error_rate": {"warning": 0.05, "critical": 0.10},  # 5% and 10%
    "execution_latency": {"warning": 100.0, "critical": 500.0},  # ms
    "drawdown": {"warning": 0.05, "critical": 0.10},  # 5% and 10%
}

health_monitor = SystemHealthMonitor(
    health_thresholds=health_thresholds
)
```

### Custom Health Checks

Add custom health checks for specific requirements:

```python
def custom_database_check():
    """Custom database connectivity check."""
    try:
        # Test database connection
        db_latency = test_database_connection()

        if db_latency > 100:
            status = HealthStatus.CRITICAL
        elif db_latency > 50:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return HealthCheckResult(
            check_type=HealthCheckType.CUSTOM,
            status=status,
            message=f"Database latency: {db_latency:.1f}ms",
            timestamp=time.time(),
            metrics={"database_latency": db_latency}
        )
    except Exception as e:
        return HealthCheckResult(
            check_type=HealthCheckType.CUSTOM,
            status=HealthStatus.CRITICAL,
            message=f"Database connection failed: {str(e)}",
            timestamp=time.time(),
            error=str(e)
        )

health_monitor.add_custom_health_check("database_connectivity", custom_database_check)
```

### Alert Handlers

Configure custom alert handlers:

```python
def critical_alert_handler(alert):
    """Handle critical alerts."""
    # Send email notification
    send_email_alert(alert)

    # Log to external monitoring system
    log_to_monitoring_system(alert)

    # Trigger automated response
    if alert.alert_type == "health_critical":
        trigger_emergency_response()

alert_manager.add_alert_handler("health_critical", critical_alert_handler)
```

## API Reference

### SystemHealthMonitor

The main class for system health monitoring.

#### Constructor

```python
SystemHealthMonitor(
    metrics_collector=None,
    alert_manager=None,
    check_interval=30.0,
    max_history=1000,
    health_thresholds=None
)
```

#### Methods

- `start_monitoring()`: Start continuous health monitoring
- `stop_monitoring()`: Stop health monitoring
- `run_health_checks()`: Run all health checks once
- `get_health_summary()`: Get overall health status summary
- `get_system_health_dashboard()`: Get comprehensive dashboard data
- `generate_health_report(output_path=None)`: Generate health report
- `add_custom_health_check(name, check_func)`: Add custom health check
- `set_health_threshold(metric_name, threshold_type, value)`: Set health threshold
- `record_latency(latency_ms)`: Record latency measurement
- `record_error()`: Record error occurrence
- `record_request()`: Record request
- `get_average_latency()`: Get average latency
- `get_error_rate()`: Get current error rate

### HealthDashboard

Comprehensive dashboard for system health visualization.

#### Constructor

```python
HealthDashboard(
    system_health_monitor,
    metrics_collector=None,
    alert_manager=None,
    output_dir="health_reports"
)
```

#### Methods

- `generate_system_health_overview()`: Generate comprehensive overview
- `create_health_visualization(save_path=None)`: Create matplotlib visualization
- `generate_html_dashboard(output_path=None)`: Generate HTML dashboard
- `save_dashboard_data(filename=None)`: Save dashboard data to JSON
- `update()`: Update dashboard
- `should_update()`: Check if dashboard should be updated
- `set_update_interval(interval)`: Set update interval

### Health Status Levels

- `HEALTHY`: System is operating normally
- `DEGRADED`: System performance is degraded but functional
- `CRITICAL`: System has critical issues requiring immediate attention
- `UNKNOWN`: Health status cannot be determined

### Health Check Types

- `SYSTEM_RESOURCES`: System resource monitoring
- `TRADING_PERFORMANCE`: Trading system performance
- `NETWORK_CONNECTIVITY`: Network connectivity testing
- `DATABASE_CONNECTIVITY`: Database connectivity (custom)
- `MODEL_PERFORMANCE`: ML model performance monitoring
- `CUSTOM`: Custom health checks

## Integration Examples

### Integration with Trading System

```python
class TradingSystem:
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.health_monitor = SystemHealthMonitor(
            metrics_collector=self.metrics_collector,
            alert_manager=self.alert_manager
        )
        self.health_monitor.start_monitoring()

    def execute_trade(self, order):
        start_time = time.time()

        try:
            # Execute trade
            result = self.broker.execute_order(order)

            # Record metrics
            execution_time = (time.time() - start_time) * 1000
            self.metrics_collector.record_metric("execution_latency", execution_time)
            self.metrics_collector.increment_counter("total_trades")
            self.health_monitor.record_latency(execution_time)
            self.health_monitor.record_request()

            return result

        except Exception as e:
            self.health_monitor.record_error()
            raise
```

### Integration with Web Dashboard

```python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/health", response_class=HTMLResponse)
async def health_dashboard():
    """Serve health dashboard."""
    health_monitor = get_health_monitor()
    health_dashboard = HealthDashboard(health_monitor)

    html_content = health_dashboard.generate_html_dashboard()
    return HTMLResponse(content=html_content)

@app.get("/health/api")
async def health_api():
    """Health API endpoint."""
    health_monitor = get_health_monitor()
    return health_monitor.get_system_health_dashboard()
```

### Integration with External Monitoring

```python
def export_to_prometheus(health_monitor):
    """Export metrics to Prometheus format."""
    dashboard_data = health_monitor.get_system_health_dashboard()

    prometheus_metrics = []

    # System metrics
    metrics = dashboard_data["current_metrics"]
    prometheus_metrics.append(f"cpu_usage_percent {metrics['cpu_percent']}")
    prometheus_metrics.append(f"memory_usage_percent {metrics['memory_percent']}")
    prometheus_metrics.append(f"disk_usage_percent {metrics['disk_percent']}")

    # Trading metrics
    trading_metrics = dashboard_data["trading_metrics"]
    prometheus_metrics.append(f"trading_pnl {trading_metrics['total_pnl']}")
    prometheus_metrics.append(f"trading_sharpe_ratio {trading_metrics['sharpe_ratio']}")
    prometheus_metrics.append(f"trading_max_drawdown {trading_metrics['max_drawdown']}")

    return "\n".join(prometheus_metrics)
```

## Best Practices

### 1. Threshold Configuration

- Set realistic thresholds based on your system's capabilities
- Use different thresholds for development and production environments
- Regularly review and adjust thresholds based on system performance

### 2. Alert Management

- Configure appropriate alert handlers for different severity levels
- Implement alert escalation procedures for critical issues
- Set up alert deduplication to avoid alert fatigue

### 3. Performance Optimization

- Use appropriate check intervals to balance monitoring overhead and responsiveness
- Implement custom health checks for critical system components
- Monitor the monitoring system itself to ensure it's functioning properly

### 4. Data Management

- Configure appropriate history limits to manage memory usage
- Implement data retention policies for historical health data
- Export important metrics to external monitoring systems

### 5. Integration

- Integrate health monitoring with your existing monitoring infrastructure
- Use health monitoring data for automated scaling decisions
- Implement health-based circuit breakers for critical operations

## Troubleshooting

### Common Issues

1. **High CPU Usage**: Check if monitoring interval is too frequent
2. **Memory Leaks**: Ensure proper cleanup of historical data
3. **False Alerts**: Review and adjust health thresholds
4. **Network Issues**: Verify network connectivity checks are appropriate

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor will now log detailed information
health_monitor = SystemHealthMonitor()
```

### Health Check Validation

Test individual health checks:

```python
# Test system resources check
result = health_monitor._check_system_resources()
print(f"System Resources: {result.status} - {result.message}")

# Test network connectivity
result = health_monitor._check_network_connectivity()
print(f"Network: {result.status} - {result.message}")
```

## Contributing

To extend the system health monitoring:

1. Add new health check types to `HealthCheckType` enum
2. Implement health check functions in `SystemHealthMonitor`
3. Add corresponding metrics collection
4. Update dashboard visualizations
5. Add CLI commands if needed
6. Update documentation

## License

This system health monitoring module is part of the trading RL agent project and follows the same license terms.
