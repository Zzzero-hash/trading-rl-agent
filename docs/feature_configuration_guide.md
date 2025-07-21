# Unified Configuration Schema

This document describes the unified configuration system for the Trading RL Agent, which consolidates all settings for data, model, backtest, and live trading operations.

## Overview

The unified configuration system provides:

- **Single source of truth** for all configuration settings
- **Type-safe configuration** using Pydantic models
- **Environment variable support** for sensitive data
- **YAML-based configuration files** for easy editing
- **Validation and documentation** built into the schema

## Configuration Sections

### 1. Data Configuration (`data`)

**Purpose**: Controls data collection, processing, and feature engineering.

**Key Settings**:

- **Sources**: Primary/backup data providers (yfinance, alpaca, alphavantage)
- **Symbols**: List of trading instruments to collect
- **Date ranges**: Start/end dates for historical data
- **Feature engineering**: Technical indicators, sentiment, market regime features
- **Data quality**: Outlier detection, missing value handling
- **Performance**: Parallel processing, caching, memory mapping

**Example**:

```yaml
data:
  sources:
    primary: yfinance
    backup: alpaca
    real_time_enabled: false
  symbols: ["AAPL", "GOOGL", "MSFT"]
  start_date: "2023-01-01"
  end_date: "2024-01-01"
  technical_indicators: true
  sentiment_features: true
```

### 2. Model Configuration (`model`)

**Purpose**: Defines model architecture, training parameters, and optimization settings.

**Key Settings**:

- **Model type**: CNN+LSTM, RL agents, or hybrid approaches
- **Architecture**: Layer configurations, activation functions, dropout rates
- **Training**: Learning rates, batch sizes, epochs, early stopping
- **RL-specific**: Timesteps, evaluation frequency, save frequency
- **Persistence**: Model saving, checkpoints, versioning

**Example**:

```yaml
model:
  type: cnn_lstm
  algorithm: sac
  cnn_filters: [64, 128, 256]
  lstm_units: 128
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

### 3. Backtest Configuration (`backtest`)

**Purpose**: Controls backtesting parameters and evaluation metrics.

**Key Settings**:

- **Test period**: Start/end dates for backtesting
- **Instruments**: Symbols to test
- **Capital**: Initial capital, commission rates, slippage
- **Risk management**: Position sizing, leverage limits, stop losses
- **Metrics**: Performance metrics to calculate
- **Output**: Results storage and visualization

**Example**:

```yaml
backtest:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  symbols: ["AAPL", "GOOGL", "MSFT"]
  initial_capital: 100000.0
  commission_rate: 0.001
  max_position_size: 0.1
```

### 4. Live Trading Configuration (`live`)

**Purpose**: Manages live trading execution and risk management.

**Key Settings**:

- **Exchange**: Trading platform (alpaca, ib, paper)
- **Execution**: Order timeouts, slippage, frequency
- **Risk management**: Position limits, drawdown controls, VaR
- **Portfolio**: Capital allocation, rebalancing
- **Monitoring**: Alerts, health checks

**Example**:

```yaml
live:
  exchange: alpaca
  paper_trading: true
  symbols: ["AAPL", "GOOGL", "MSFT"]
  max_position_size: 0.1
  max_drawdown: 0.15
  alerts_enabled: true
```

### 5. Monitoring Configuration (`monitoring`)

**Purpose**: Controls logging, experiment tracking, and alerting.

**Key Settings**:

- **Logging**: Log levels, file paths, structured logging
- **Experiment tracking**: MLflow, TensorBoard integration
- **Metrics**: Collection frequency, health checks
- **Alerting**: Email, Slack notifications

**Example**:

```yaml
monitoring:
  log_level: "INFO"
  mlflow_enabled: true
  tensorboard_enabled: true
  alerts_enabled: true
```

### 6. Infrastructure Configuration (`infrastructure`)

**Purpose**: Manages system resources and distributed computing.

**Key Settings**:

- **Distributed computing**: Ray cluster, worker management
- **Storage**: Model registry, experiment tracking
- **System resources**: GPU usage, memory limits, worker counts

**Example**:

```yaml
infrastructure:
  distributed: false
  gpu_enabled: true
  max_workers: 4
  memory_limit: "8GB"
```

## Sensitive Fields and Environment Variables

**⚠️ CRITICAL**: The following fields contain sensitive information and should **NEVER** be stored in configuration files. They are automatically loaded from environment variables.

### API Keys and Credentials

| Field                  | Environment Variable   | Description               | Required For                  |
| ---------------------- | ---------------------- | ------------------------- | ----------------------------- |
| `alpaca_api_key`       | `ALPACA_API_KEY`       | Alpaca trading API key    | Live trading, data collection |
| `alpaca_secret_key`    | `ALPACA_SECRET_KEY`    | Alpaca trading secret key | Live trading, data collection |
| `alpaca_base_url`      | `ALPACA_BASE_URL`      | Alpaca API base URL       | Live trading, data collection |
| `alphavantage_api_key` | `ALPHAVANTAGE_API_KEY` | Alpha Vantage API key     | Alternative data              |
| `newsapi_key`          | `NEWSAPI_KEY`          | News API key              | Sentiment analysis            |
| `social_api_key`       | `SOCIAL_API_KEY`       | Social media API key      | Sentiment analysis            |

### Environment Variable Setup

Create a `.env` file in your project root:

```bash
# Trading API Credentials
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Data API Keys
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here
NEWSAPI_KEY=your_newsapi_key_here
SOCIAL_API_KEY=your_social_api_key_here

# Optional: Override other settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
```

### Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive data
3. **Rotate API keys** regularly
4. **Use paper trading** for development and testing
5. **Limit API key permissions** to minimum required access

## Configuration Loading

### From YAML File

```python
from trading_rl_agent.core.unified_config import UnifiedConfig

# Load from YAML file
config = UnifiedConfig.from_yaml("configs/unified_config.yaml")
```

### From Environment Variables

```python
from trading_rl_agent.core.unified_config import UnifiedConfig

# Load from environment variables (including .env file)
config = UnifiedConfig()
```

### Mixed Loading

```python
from trading_rl_agent.core.unified_config import load_config

# Load from YAML, with environment variables overriding sensitive fields
config = load_config("configs/unified_config.yaml")
```

## Configuration Validation

The Pydantic models provide automatic validation:

```python
# This will raise validation errors for invalid values
config = UnifiedConfig(
    data=DataConfig(
        real_data_ratio=1.5  # Invalid: must be <= 1.0
    )
)
```

## Configuration Updates

```python
# Update configuration programmatically
config.data.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
config.model.batch_size = 64

# Save updated configuration
config.to_yaml("configs/updated_config.yaml")
```

## CLI Integration

The unified configuration integrates with the CLI system:

```bash
# Use default configuration
trade-agent data download

# Use custom configuration file
trade-agent --config configs/custom.yaml data download

# Override with environment variables
ENVIRONMENT=development trade-agent data download
```

## Migration from Legacy Configs

The unified configuration consolidates settings from multiple legacy files:

- `configs/production.yaml` → `live` section
- `configs/development.yaml` → `environment` and `debug` settings
- `configs/cnn_lstm_training.yaml` → `model` section
- `configs/finrl_*.yaml` → `data` section

## Best Practices

1. **Use environment-specific configs**: Create separate files for dev/staging/prod
2. **Validate configurations**: Always validate before deployment
3. **Document custom settings**: Add comments for non-standard configurations
4. **Version control configs**: Track configuration changes (excluding sensitive data)
5. **Test configurations**: Validate configs in CI/CD pipelines

## Troubleshooting

### Common Issues

1. **API key not found**: Ensure environment variables are set correctly
2. **Validation errors**: Check field types and value ranges
3. **Missing dependencies**: Install required packages for specific features
4. **Permission errors**: Verify file/directory permissions for paths

### Debug Configuration

```python
# Print full configuration (excluding sensitive fields)
print(config.model_dump_json(indent=2))

# Check specific sections
print(f"Data sources: {config.data.sources}")
print(f"Model type: {config.model.type}")
print(f"API credentials available: {bool(config.alpaca_api_key)}")
```
