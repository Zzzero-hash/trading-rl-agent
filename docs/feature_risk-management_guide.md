# Risk Alert System

## Overview

The Risk Alert System is a comprehensive automated risk monitoring and alerting solution that integrates with the existing trading RL agent framework. It provides real-time portfolio risk monitoring, configurable alert thresholds, automated circuit breakers, multi-level escalation procedures, and comprehensive reporting capabilities.

## Features

### 🚨 Real-time Risk Monitoring

- Continuous monitoring of portfolio risk metrics
- Configurable monitoring intervals
- Integration with existing RiskManager
- Real-time risk metric calculations

### ⚠️ Configurable Alert Thresholds

- Multiple threshold types (min, max, change_rate)
- Per-metric alert configuration
- Cooldown periods to prevent alert spam
- Severity-based alert classification

### 🔌 Automated Circuit Breakers

- Configurable trigger conditions
- Multiple action types (reduce_position, stop_trading, liquidate)
- Automatic risk mitigation
- Cooldown periods for circuit breakers

### 📈 Multi-level Escalation

- 5-level escalation system
- Automated escalation procedures
- Configurable escalation timeouts
- Emergency shutdown capabilities

### 📧 Multi-channel Notifications

- Email notifications
- Slack integration
- SMS alerts (via Twilio)
- Webhook notifications
- Priority-based notification routing

### 📊 Comprehensive Reporting

- Risk alert reports
- Audit trail logging
- Historical risk analysis
- Automated report generation
- Recommendations engine

### 🔧 Configuration Management

- YAML-based configuration
- Environment variable support
- Runtime configuration updates
- Portfolio-specific settings

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   RiskManager   │    │  AlertManager    │    │ RiskAlertSystem │
│                 │    │                  │    │                 │
│ • VaR Calc      │◄──►│ • Alert Storage  │◄──►│ • Monitoring    │
│ • Risk Metrics  │    │ • Alert Status   │    │ • Thresholds    │
│ • Risk Limits   │    │ • Alert History  │    │ • Circuit Breakers│
└─────────────────┘    └──────────────────┘    │ • Escalation    │
                                               │ • Notifications │
                                               │ • Reporting     │
                                               └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Email/Slack   │    │   Audit Logs     │    │   Risk Reports  │
│   Notifications │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

The Risk Alert System is part of the trading RL agent package. No additional installation is required beyond the existing dependencies.

### Dependencies

The system uses the following key dependencies:

- `asyncio` - Asynchronous operations
- `pydantic` - Configuration validation
- `pandas` - Data manipulation
- `requests` - HTTP notifications
- `rich` - CLI interface
- `click` - Command-line interface

## Quick Start

### 1. Basic Usage

```python
from src.trading_rl_agent.risk import RiskAlertSystem, RiskAlertConfig
from src.trading_rl_agent.risk.manager import RiskManager
from src.trading_rl_agent.monitoring.alert_manager import AlertManager

# Create components
risk_manager = RiskManager()
alert_manager = AlertManager()
config = RiskAlertConfig()

# Create risk alert system
risk_alert_system = RiskAlertSystem(
    risk_manager=risk_manager,
    alert_manager=alert_manager,
    config=config,
    portfolio_id="my_portfolio"
)

# Start monitoring
await risk_alert_system.start_monitoring()
```

### 2. Using Configuration File

```python
import yaml
from src.trading_rl_agent.risk import RiskAlertConfig

# Load configuration from file
with open('configs/risk_alert_config.yaml', 'r') as f:
    config_data = yaml.safe_load(f)

config = RiskAlertConfig(**config_data)
```

### 3. CLI Usage

```bash
# Start monitoring
python -m src.trading_rl_agent.risk.cli_risk_alerts start-monitoring --portfolio-id my_portfolio

# Check status
python -m src.trading_rl_agent.risk.cli_risk_alerts status --portfolio-id my_portfolio

# View alerts
python -m src.trading_rl_agent.risk.cli_risk_alerts alerts --limit 10

# Generate report
python -m src.trading_rl_agent.risk.cli_risk_alerts report --days 7 --output report.json
```

## Configuration

### Alert Thresholds

Alert thresholds define when risk alerts should be triggered:

```yaml
alert_thresholds:
  - metric_name: "portfolio_var"
    threshold_type: "max"
    threshold_value: 0.05
    severity: "warning"
    escalation_level: "level_2"
    cooldown_minutes: 30
    enabled: true
    description: "Portfolio VaR exceeds 5%"
```

**Supported Metrics:**

- `portfolio_var` - Portfolio Value at Risk
- `portfolio_cvar` - Portfolio Conditional Value at Risk
- `current_drawdown` - Current portfolio drawdown
- `max_drawdown` - Maximum historical drawdown
- `leverage` - Portfolio leverage ratio
- `sharpe_ratio` - Sharpe ratio
- `sortino_ratio` - Sortino ratio
- `beta` - Portfolio beta
- `correlation_risk` - Portfolio correlation risk
- `concentration_risk` - Portfolio concentration risk

**Threshold Types:**

- `max` - Alert when value exceeds threshold
- `min` - Alert when value falls below threshold
- `change_rate` - Alert when change rate exceeds threshold

**Severity Levels:**

- `info` - Informational alerts
- `warning` - Warning alerts
- `error` - Error alerts
- `critical` - Critical alerts

**Escalation Levels:**

- `level_1` - Automated monitoring
- `level_2` - Email notification
- `level_3` - Slack/SMS notification
- `level_4` - Phone call (requires external service)
- `level_5` - Emergency shutdown

### Circuit Breaker Rules

Circuit breakers automatically take action when risk thresholds are exceeded:

```yaml
circuit_breaker_rules:
  - name: "var_circuit_breaker"
    trigger_condition: "var_exceeded"
    threshold_value: 0.10
    action: "stop_trading"
    cooldown_minutes: 120
    enabled: true
    description: "Stop all trading when VaR exceeds 10%"
```

**Trigger Conditions:**

- `var_exceeded` - Portfolio VaR exceeds threshold
- `drawdown_exceeded` - Portfolio drawdown exceeds threshold
- `leverage_exceeded` - Portfolio leverage exceeds threshold
- `correlation_risk_exceeded` - Portfolio correlation risk exceeds threshold

**Actions:**

- `reduce_position` - Reduce position sizes
- `stop_trading` - Stop all trading activities
- `liquidate` - Liquidate all positions

### Notification Configuration

Configure notification channels:

```yaml
notifications:
  email_enabled: true
  email_recipients:
    - "risk-manager@company.com"
    - "trading-desk@company.com"
  email_smtp_server: "smtp.gmail.com"
  email_smtp_port: 587
  email_username: "risk-alerts@company.com"
  email_password: "" # Set via environment variable

  slack_enabled: true
  slack_webhook_url: "" # Set via environment variable
  slack_channel: "#risk-alerts"

  sms_enabled: false
  sms_provider: "twilio"
  sms_api_key: "" # Set via environment variable
  sms_api_secret: "" # Set via environment variable
  sms_recipients:
    - "+1234567890"

  webhook_enabled: false
  webhook_url: "" # Set via environment variable
  webhook_headers:
    Content-Type: "application/json"
    Authorization: "Bearer your-token"
```

## API Reference

### RiskAlertSystem

#### Constructor

```python
RiskAlertSystem(
    risk_manager: RiskManager,
    alert_manager: AlertManager,
    config: RiskAlertConfig,
    portfolio_id: str = "default"
)
```

#### Methods

##### Monitoring

```python
# Start real-time monitoring
await risk_alert_system.start_monitoring()

# Stop monitoring
await risk_alert_system.stop_monitoring()

# Check if monitoring is active
is_active = risk_alert_system.is_monitoring
```

##### Alert Management

```python
# Add alert threshold
threshold = AlertThreshold(
    metric_name="portfolio_var",
    threshold_type="max",
    threshold_value=0.05,
    severity=AlertSeverity.WARNING,
    escalation_level=EscalationLevel.LEVEL_2
)
risk_alert_system.add_alert_threshold(threshold)

# Remove alert threshold
risk_alert_system.remove_alert_threshold("portfolio_var")

# Get alert history
alerts = risk_alert_system.get_alert_history(limit=10)
```

##### Circuit Breakers

```python
# Add circuit breaker rule
rule = CircuitBreakerRule(
    name="var_breaker",
    trigger_condition="var_exceeded",
    threshold_value=0.1,
    action="stop_trading"
)
risk_alert_system.add_circuit_breaker_rule(rule)

# Remove circuit breaker rule
risk_alert_system.remove_circuit_breaker_rule("var_breaker")

# Reset circuit breaker status
risk_alert_system.reset_circuit_breaker()

# Get circuit breaker status
status = risk_alert_system.circuit_breaker_status
```

##### Reporting

```python
# Get risk summary
summary = risk_alert_system.get_risk_summary()

# Generate risk report
from datetime import datetime, timedelta
end_time = datetime.now()
start_time = end_time - timedelta(days=7)
report = risk_alert_system.generate_risk_report(start_time, end_time)

# Get risk history
history = risk_alert_system.get_risk_history(limit=100)
```

## CLI Commands

### Basic Commands

```bash
# Start monitoring
python -m src.trading_rl_agent.risk.cli_risk_alerts start-monitoring [OPTIONS]

# Check system status
python -m src.trading_rl_agent.risk.cli_risk_alerts status [OPTIONS]

# View alerts
python -m src.trading_rl_agent.risk.cli_risk_alerts alerts [OPTIONS]

# Generate report
python -m src.trading_rl_agent.risk.cli_risk_alerts report [OPTIONS]
```

### Configuration Commands

```bash
# Show configuration
python -m src.trading_rl_agent.risk.cli_risk_alerts config-show [OPTIONS]

# Show alert thresholds
python -m src.trading_rl_agent.risk.cli_risk_alerts thresholds [OPTIONS]

# Show circuit breakers
python -m src.trading_rl_agent.risk.cli_risk_alerts circuit-breakers [OPTIONS]

# Add alert threshold
python -m src.trading_rl_agent.risk.cli_risk_alerts add-threshold [OPTIONS]

# Add circuit breaker rule
python -m src.trading_rl_agent.risk.cli_risk_alerts add-circuit-breaker [OPTIONS]

# Reset circuit breaker
python -m src.trading_rl_agent.risk.cli_risk_alerts reset [OPTIONS]
```

### Command Options

Common options for all commands:

- `--config, -c` - Path to configuration file
- `--portfolio-id` - Portfolio identifier (default: "default")

Alert-specific options:

- `--limit` - Number of alerts to show
- `--severity` - Filter by severity (info, warning, error, critical)
- `--status` - Filter by status (active, acknowledged, resolved, dismissed)

Report-specific options:

- `--days` - Number of days to include in report
- `--output` - Output file path for JSON report

## Examples

### Example 1: Basic Risk Monitoring

```python
import asyncio
from src.trading_rl_agent.risk import RiskAlertSystem, RiskAlertConfig
from src.trading_rl_agent.risk.manager import RiskManager
from src.trading_rl_agent.monitoring.alert_manager import AlertManager

async def main():
    # Create components
    risk_manager = RiskManager()
    alert_manager = AlertManager()

    # Create configuration with basic thresholds
    config = RiskAlertConfig(
        monitoring_interval_seconds=60,
        alert_thresholds=[
            {
                "metric_name": "portfolio_var",
                "threshold_type": "max",
                "threshold_value": 0.05,
                "severity": "warning",
                "escalation_level": "level_2",
                "cooldown_minutes": 30,
                "enabled": True,
                "description": "Portfolio VaR exceeds 5%"
            }
        ],
        circuit_breaker_rules=[
            {
                "name": "var_circuit_breaker",
                "trigger_condition": "var_exceeded",
                "threshold_value": 0.10,
                "action": "stop_trading",
                "cooldown_minutes": 120,
                "enabled": True,
                "description": "Stop trading when VaR exceeds 10%"
            }
        ]
    )

    # Create risk alert system
    risk_alert_system = RiskAlertSystem(
        risk_manager=risk_manager,
        alert_manager=alert_manager,
        config=config,
        portfolio_id="example_portfolio"
    )

    # Start monitoring
    await risk_alert_system.start_monitoring()

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await risk_alert_system.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
```

### Example 2: Custom Alert Thresholds

```python
from src.trading_rl_agent.risk import AlertThreshold, EscalationLevel
from src.trading_rl_agent.monitoring.alert_manager import AlertSeverity

# Create custom thresholds
var_threshold = AlertThreshold(
    metric_name="portfolio_var",
    threshold_type="max",
    threshold_value=0.03,
    severity=AlertSeverity.WARNING,
    escalation_level=EscalationLevel.LEVEL_2,
    cooldown_minutes=30,
    enabled=True,
    description="Portfolio VaR exceeds 3%"
)

drawdown_threshold = AlertThreshold(
    metric_name="current_drawdown",
    threshold_type="max",
    threshold_value=0.15,
    severity=AlertSeverity.ERROR,
    escalation_level=EscalationLevel.LEVEL_3,
    cooldown_minutes=60,
    enabled=True,
    description="Current drawdown exceeds 15%"
)

# Add to system
risk_alert_system.add_alert_threshold(var_threshold)
risk_alert_system.add_alert_threshold(drawdown_threshold)
```

### Example 3: Custom Circuit Breakers

```python
from src.trading_rl_agent.risk import CircuitBreakerRule

# Create custom circuit breakers
var_breaker = CircuitBreakerRule(
    name="conservative_var_breaker",
    trigger_condition="var_exceeded",
    threshold_value=0.08,
    action="reduce_position",
    cooldown_minutes=60,
    enabled=True,
    description="Reduce positions when VaR exceeds 8%"
)

leverage_breaker = CircuitBreakerRule(
    name="leverage_breaker",
    trigger_condition="leverage_exceeded",
    threshold_value=2.0,
    action="stop_trading",
    cooldown_minutes=30,
    enabled=True,
    description="Stop trading when leverage exceeds 200%"
)

# Add to system
risk_alert_system.add_circuit_breaker_rule(var_breaker)
risk_alert_system.add_circuit_breaker_rule(leverage_breaker)
```

### Example 4: Email Notifications

```python
from src.trading_rl_agent.risk import NotificationConfig

# Configure email notifications
notifications = NotificationConfig(
    email_enabled=True,
    email_recipients=[
        "risk-manager@company.com",
        "trading-desk@company.com",
        "compliance@company.com"
    ],
    email_smtp_server="smtp.gmail.com",
    email_smtp_port=587,
    email_username="risk-alerts@company.com",
    email_password="your-app-password"  # Use environment variable in production
)

# Update configuration
config.notifications = notifications
```

### Example 5: Slack Integration

```python
# Configure Slack notifications
notifications = NotificationConfig(
    slack_enabled=True,
    slack_webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
    slack_channel="#risk-alerts"
)

# Update configuration
config.notifications = notifications
```

### Example 6: Risk Reporting

```python
from datetime import datetime, timedelta

# Generate daily report
end_time = datetime.now()
start_time = end_time - timedelta(days=1)

report = risk_alert_system.generate_risk_report(start_time, end_time)

# Print report summary
print(f"Portfolio: {report['portfolio_id']}")
print(f"Period: {report['report_period']['start']} to {report['report_period']['end']}")
print(f"Circuit Breaker Status: {report['circuit_breaker_status']}")

# Alert statistics
alert_stats = report['alert_statistics']
print(f"Total Alerts: {alert_stats['total_alerts']}")
print(f"Critical Alerts: {alert_stats['critical_alerts']}")
print(f"Error Alerts: {alert_stats['error_alerts']}")

# Risk statistics
risk_stats = report['risk_statistics']
print(f"Average VaR: {risk_stats['avg_var']:.4f}")
print(f"Maximum VaR: {risk_stats['max_var']:.4f}")
print(f"Average Drawdown: {risk_stats['avg_drawdown']:.4f}")
print(f"Maximum Drawdown: {risk_stats['max_drawdown']:.4f}")

# Recommendations
for i, rec in enumerate(report['recommendations'], 1):
    print(f"{i}. {rec}")

# Save report to file
import json
with open('risk_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)
```

## Best Practices

### 1. Configuration Management

- Use environment variables for sensitive information (API keys, passwords)
- Keep configuration files in version control
- Use different configurations for different environments (dev, staging, prod)
- Document all threshold values and their rationale

### 2. Alert Thresholds

- Start with conservative thresholds and adjust based on performance
- Use cooldown periods to prevent alert spam
- Group related thresholds by severity level
- Regularly review and update thresholds based on market conditions

### 3. Circuit Breakers

- Set circuit breaker thresholds higher than alert thresholds
- Use appropriate cooldown periods to allow for recovery
- Test circuit breaker actions in a safe environment
- Have manual override procedures for emergency situations

### 4. Notifications

- Use multiple notification channels for redundancy
- Set up escalation procedures for critical alerts
- Test notification systems regularly
- Monitor notification delivery rates

### 5. Monitoring and Maintenance

- Monitor system performance and resource usage
- Regularly review audit logs
- Update risk models and thresholds as needed
- Conduct regular system tests and drills

### 6. Security

- Secure all API keys and credentials
- Use HTTPS for all external communications
- Implement proper access controls
- Regularly audit system access

## Troubleshooting

### Common Issues

1. **Alerts not triggering**
   - Check if thresholds are enabled
   - Verify metric values are being updated
   - Check cooldown periods
   - Review alert manager configuration

2. **Notifications not sending**
   - Verify notification configuration
   - Check network connectivity
   - Validate API keys and credentials
   - Review notification service status

3. **Circuit breakers not working**
   - Check if rules are enabled
   - Verify trigger conditions
   - Review cooldown periods
   - Check action implementation

4. **Performance issues**
   - Reduce monitoring frequency
   - Optimize metric calculations
   - Review alert history size
   - Check system resources

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Checks

Use the CLI status command to check system health:

```bash
python -m src.trading_rl_agent.risk.cli_risk_alerts status --portfolio-id your_portfolio
```

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `pytest tests/test_risk_alert_system.py`
4. Run example: `python examples/risk_alert_example.py`

### Adding New Features

1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Submit pull request

### Testing

Run the test suite:

```bash
# Run all tests
pytest tests/test_risk_alert_system.py

# Run specific test class
pytest tests/test_risk_alert_system.py::TestRiskAlertSystem

# Run with coverage
pytest tests/test_risk_alert_system.py --cov=src.trading_rl_agent.risk
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

1. Check the documentation
2. Review the examples
3. Run the test suite
4. Create an issue on GitHub

## Changelog

### Version 1.0.0

- Initial release
- Basic risk monitoring
- Alert thresholds
- Circuit breakers
- Multi-channel notifications
- CLI interface
- Comprehensive testing
- Documentation
