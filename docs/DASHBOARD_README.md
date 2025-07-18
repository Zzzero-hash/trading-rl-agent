# Real-Time P&L and Performance Dashboard

This document provides comprehensive information about the real-time P&L and performance dashboard system, including setup, configuration, and usage examples.

## Overview

The Performance Dashboard is a comprehensive web-based monitoring system that provides real-time insights into trading performance, risk metrics, and system health. It features:

- **Real-time P&L tracking** with interactive charts
- **Risk metrics visualization** including VaR, CVaR, and volatility
- **Position overview** with detailed position information
- **System health monitoring** for CPU, memory, and network metrics
- **Real-time alerts** and notifications
- **WebSocket streaming** for live data updates
- **Customizable layouts** and configurations

## Features

### 📊 Real-Time P&L and Performance Metrics
- Live portfolio P&L tracking
- Daily and cumulative returns
- Sharpe ratio and risk-adjusted metrics
- Win rate and trade statistics
- Maximum drawdown analysis

### ⚠️ Risk Metrics and Position Information
- Value at Risk (VaR) and Conditional VaR (CVaR)
- Portfolio volatility and beta
- Current exposure and position concentration
- Real-time position tracking
- Risk limit monitoring

### 📈 Interactive Charts and Visualizations
- Interactive Plotly charts
- Real-time data updates
- Multiple chart types (line, bar, gauge)
- Customizable time ranges
- Export capabilities

### 🔄 Real-Time Data Updates and Streaming
- WebSocket-based real-time updates
- Configurable update intervals
- Live data streaming
- Connection management
- Data persistence

### 🎨 Customizable Dashboard Layouts
- Grid layout for comprehensive view
- Single column layout for focused monitoring
- Custom layout support
- Theme customization (light/dark)
- Chart visibility controls

## Installation

### Prerequisites
- Python 3.8+
- Required dependencies (see requirements.txt)

### Dependencies
The dashboard requires the following additional dependencies:

```bash
pip install streamlit plotly websockets
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Dashboard

Run the basic dashboard with default settings:

```bash
python -m trading_rl_agent.cli_dashboard run
```

The dashboard will be available at `http://localhost:8501`

### 2. Dashboard with Streaming

Run the dashboard with WebSocket streaming enabled:

```bash
python -m trading_rl_agent.cli_dashboard run --streaming
```

This starts both the web dashboard and a WebSocket streaming server.

### 3. Custom Configuration

Create a custom configuration file:

```bash
python -m trading_rl_agent.cli_dashboard config --create my_config.json
```

Run with custom configuration:

```bash
python -m trading_rl_agent.cli_dashboard run --config my_config.json
```

## Configuration

### Dashboard Configuration

The dashboard can be configured using JSON configuration files:

```json
{
  "layout": "grid",
  "theme": "light",
  "auto_refresh": true,
  "refresh_interval": 1.0,
  "charts": {
    "pnl_chart": true,
    "risk_metrics": true,
    "position_overview": true,
    "performance_metrics": true,
    "system_health": true,
    "alerts": true
  },
  "time_range": "24h",
  "metrics_display": {
    "show_percentages": true,
    "show_currency": true,
    "currency_symbol": "$",
    "decimal_places": 2
  }
}
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `layout` | string | "grid" | Dashboard layout ("grid", "single_column", "custom") |
| `theme` | string | "light" | Theme ("light", "dark") |
| `auto_refresh` | boolean | true | Enable automatic refresh |
| `refresh_interval` | float | 1.0 | Refresh interval in seconds |
| `charts.*` | boolean | true | Enable/disable specific charts |
| `time_range` | string | "24h" | Default time range ("1h", "6h", "24h", "7d", "30d") |
| `metrics_display.*` | various | - | Display formatting options |

## CLI Commands

### Run Dashboard

```bash
# Basic dashboard
python -m trading_rl_agent.cli_dashboard run

# With custom configuration
python -m trading_rl_agent.cli_dashboard run --config config.json

# With streaming enabled
python -m trading_rl_agent.cli_dashboard run --streaming

# Custom host and port
python -m trading_rl_agent.cli_dashboard run --host 0.0.0.0 --port 8502
```

### Streaming Server

```bash
# Run streaming server only
python -m trading_rl_agent.cli_dashboard stream --host 0.0.0.0 --port 8765

# Custom update interval
python -m trading_rl_agent.cli_dashboard stream --update-interval 0.5
```

### Configuration Management

```bash
# Create new configuration
python -m trading_rl_agent.cli_dashboard config --create config.json

# Validate configuration
python -m trading_rl_agent.cli_dashboard config --validate config.json

# Show configuration
python -m trading_rl_agent.cli_dashboard config --show config.json
```

### Data Export

```bash
# Export dashboard data
python -m trading_rl_agent.cli_dashboard export --output data.json

# Export with custom configuration
python -m trading_rl_agent.cli_dashboard export --output data.json --config config.json
```

### Status Check

```bash
# Check dashboard status
python -m trading_rl_agent.cli_dashboard status

# Check specific streaming server
python -m trading_rl_agent.cli_dashboard status --streaming-host localhost --streaming-port 8765
```

## API Usage

### Basic Dashboard Setup

```python
from trading_rl_agent.monitoring.dashboard import Dashboard
from trading_rl_agent.monitoring.metrics_collector import MetricsCollector
from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard

# Create components
metrics_collector = MetricsCollector()
dashboard = Dashboard(metrics_collector)
performance_dashboard = PerformanceDashboard(
    metrics_collector=metrics_collector,
    dashboard=dashboard,
    update_interval=1.0,
    max_data_points=1000
)

# Run dashboard
performance_dashboard.run_dashboard()
```

### Streaming Dashboard

```python
import asyncio
from trading_rl_agent.monitoring.streaming_dashboard import StreamingDashboard

# Create streaming dashboard
streaming_dashboard = StreamingDashboard(
    performance_dashboard=performance_dashboard,
    host="localhost",
    port=8765,
    update_interval=0.1
)

# Run streaming server
asyncio.run(streaming_dashboard.start_server())
```

### WebSocket Client

```python
import asyncio
from trading_rl_agent.monitoring.streaming_dashboard import WebSocketClient

async def main():
    # Create client
    client = WebSocketClient("ws://localhost:8765")
    
    # Connect to server
    await client.connect()
    
    # Subscribe to data streams
    await client.subscribe("all")
    
    # Add message handler
    def handle_data_update(data):
        print(f"Received update: {data}")
    
    client.add_message_handler("data_update", handle_data_update)
    
    # Listen for messages
    await client.listen()

# Run client
asyncio.run(main())
```

## Dashboard Components

### 1. Header Metrics
- Total P&L
- Daily P&L
- Sharpe Ratio
- Max Drawdown

### 2. P&L Chart
- Real-time P&L line chart
- Cumulative return chart
- Interactive time range selection

### 3. Risk Metrics
- VaR and CVaR gauges
- Volatility and beta indicators
- Exposure and concentration metrics

### 4. Position Overview
- Current positions table
- Position P&L and values
- Position summary statistics

### 5. Performance Metrics
- Win rate and trade counts
- Performance bar charts
- Additional metrics display

### 6. System Health
- CPU, memory, and disk usage
- Network latency and error rates
- System health gauges

### 7. Alerts
- Real-time alert display
- Alert severity indicators
- Alert history

## Data Integration

### Metrics Collection

The dashboard integrates with the existing `MetricsCollector` class:

```python
# Record trading metrics
metrics_collector.record_metric('pnl', 1250.50)
metrics_collector.record_metric('daily_pnl', 125.75)
metrics_collector.record_metric('total_return', 0.125)

# Record risk metrics
metrics_collector.record_metric('var_95', -0.025)
metrics_collector.record_metric('volatility', 0.18)

# Record system metrics
metrics_collector.record_metric('cpu_usage', 45.2)
metrics_collector.record_metric('memory_usage', 62.8)

# Update counters and gauges
metrics_collector.increment_counter('total_trades', 1)
metrics_collector.set_gauge('open_positions', 8)
```

### Real-Time Updates

The dashboard automatically updates when new metrics are recorded:

```python
import time
import threading

def update_metrics():
    while True:
        # Update metrics
        metrics_collector.record_metric('pnl', get_current_pnl())
        metrics_collector.record_metric('daily_pnl', get_daily_pnl())
        time.sleep(1.0)

# Start update thread
update_thread = threading.Thread(target=update_metrics, daemon=True)
update_thread.start()
```

## Examples

### Example 1: Basic Dashboard with Sample Data

```python
#!/usr/bin/env python3
import time
import threading
from trading_rl_agent.monitoring.dashboard import Dashboard
from trading_rl_agent.monitoring.metrics_collector import MetricsCollector
from trading_rl_agent.monitoring.performance_dashboard import PerformanceDashboard

def create_sample_data(metrics_collector):
    import random
    metrics_collector.record_metric('pnl', 10000 + random.uniform(-500, 500))
    metrics_collector.record_metric('daily_pnl', random.uniform(-200, 300))
    metrics_collector.record_metric('total_return', 0.15 + random.uniform(-0.02, 0.02))
    metrics_collector.record_metric('sharpe_ratio', random.uniform(1.0, 2.5))

def run_data_simulator(metrics_collector, stop_event):
    while not stop_event.is_set():
        create_sample_data(metrics_collector)
        time.sleep(2.0)

# Setup
metrics_collector = MetricsCollector()
dashboard = Dashboard(metrics_collector)
performance_dashboard = PerformanceDashboard(
    metrics_collector=metrics_collector,
    dashboard=dashboard
)

# Start data simulator
stop_event = threading.Event()
simulator_thread = threading.Thread(
    target=run_data_simulator,
    args=(metrics_collector, stop_event),
    daemon=True
)
simulator_thread.start()

# Run dashboard
try:
    performance_dashboard.run_dashboard()
except KeyboardInterrupt:
    stop_event.set()
```

### Example 2: Streaming Dashboard

```python
#!/usr/bin/env python3
import asyncio
import threading
import time
from trading_rl_agent.monitoring.streaming_dashboard import StreamingDashboard

async def main():
    # Create streaming dashboard
    streaming_dashboard = StreamingDashboard(
        performance_dashboard=performance_dashboard,
        host="localhost",
        port=8765
    )
    
    # Start data simulator
    stop_event = threading.Event()
    simulator_thread = threading.Thread(
        target=run_data_simulator,
        args=(metrics_collector, stop_event),
        daemon=True
    )
    simulator_thread.start()
    
    try:
        await streaming_dashboard.start_server()
    except KeyboardInterrupt:
        stop_event.set()
        await streaming_dashboard.stop_server()

# Run streaming dashboard
asyncio.run(main())
```

## Troubleshooting

### Common Issues

1. **Dashboard not starting**
   - Check if port 8501 is available
   - Verify all dependencies are installed
   - Check console for error messages

2. **No data displayed**
   - Ensure metrics are being recorded
   - Check metrics collector configuration
   - Verify data update frequency

3. **Streaming not working**
   - Check if WebSocket port is available
   - Verify network connectivity
   - Check firewall settings

4. **Performance issues**
   - Reduce update frequency
   - Limit data points in memory
   - Check system resources

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Check dashboard status:

```bash
python -m trading_rl_agent.cli_dashboard status
```

## Performance Considerations

### Data Management
- Limit data points in memory (default: 1000)
- Configure appropriate update intervals
- Use data retention policies

### Resource Usage
- Monitor CPU and memory usage
- Adjust update frequencies as needed
- Consider scaling for high-frequency updates

### Network Considerations
- Use appropriate WebSocket configurations
- Monitor connection limits
- Implement connection pooling for multiple clients

## Security

### Access Control
- Configure appropriate host bindings
- Use authentication if needed
- Implement rate limiting

### Data Protection
- Secure sensitive trading data
- Implement data encryption
- Use secure WebSocket connections (WSS)

## Contributing

To contribute to the dashboard:

1. Follow the existing code style
2. Add appropriate tests
3. Update documentation
4. Test with real trading data

## License

This dashboard is part of the trading RL agent project and follows the same license terms.

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review example code
3. Check existing documentation
4. Create an issue in the project repository