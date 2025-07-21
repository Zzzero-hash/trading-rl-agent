# Configuration Examples

This directory contains canonical configuration examples for the Trading RL Agent system.

## Overview

The configuration system uses Pydantic models for validation and supports:

- YAML configuration files
- Environment variable overrides
- `.env` file loading
- Type-safe configuration with automatic validation

## Configuration Files

### `local-example.yaml`

**Purpose**: Development machine configuration with paper trading

**Key Features**:

- Debug mode enabled
- Paper trading for safe development
- CPU-focused training settings
- Local file paths
- Detailed logging for debugging

**Use Case**: Local development, testing, and experimentation

### `prod-example.yaml`

**Purpose**: Production Docker deployment with live trading

**Key Features**:

- Production-optimized settings
- Live trading configuration
- GPU-accelerated training
- Distributed computing support
- Docker volume paths
- Conservative risk management

**Use Case**: Production deployment, live trading, high-performance environments

## Configuration Structure

Both configuration files follow the same structure with these main sections:

### Environment Settings

```yaml
environment: development # or production
debug: true # or false
```

### Data Configuration

- Data sources (yfinance, alpaca, alphavantage)
- Trading symbols
- Date ranges and timeframes
- Feature engineering settings
- Storage paths

### Model Configuration

- Model architecture (CNN+LSTM, RL agents)
- Training parameters
- Model persistence settings
- Device configuration

### Agent Configuration

- RL algorithm settings
- Ensemble configuration
- Evaluation and save frequencies

### Risk Management

- Position sizing limits
- Leverage constraints
- Stop-loss and take-profit settings
- VaR calculations

### Execution Configuration

- Broker settings
- Order management
- Commission and slippage
- Market hours trading

### Monitoring Configuration

- Logging settings
- Experiment tracking
- Metrics collection
- Alerting configuration

### Infrastructure Configuration

- Worker processes
- GPU settings
- Distributed computing
- Memory limits

## Required Fields

All fields marked with `# REQUIRED` in the configuration files must be provided. The system will use default values for optional fields if not specified.

## API Keys and Secrets

**IMPORTANT**: Never hardcode API keys in configuration files. Use environment variables instead:

### Environment Variable Format

The system uses the prefix `TRADE_AGENT_` for all environment variables:

```bash
export TRADE_AGENT_ALPACA_API_KEY=your_api_key
export TRADE_AGENT_ALPACA_SECRET_KEY=your_secret_key
export TRADE_AGENT_ALPACA_BASE_URL=https://api.alpaca.markets
```

### .env File

Create a `.env` file in your project root:

```env
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://api.alpaca.markets
ALPHAVANTAGE_API_KEY=your_alphavantage_key
NEWSAPI_KEY=your_newsapi_key
```

## Usage

### Loading Configuration

```python
from config import load_settings

# Load with default settings
settings = load_settings()

# Load with custom config file
settings = load_settings(config_path="config/local-example.yaml")

# Load with custom .env file
settings = load_settings(env_file=".env.production")
```

### Environment-Specific Configuration

1. **Development**: Use `local-example.yaml` as a starting point
2. **Staging**: Copy `local-example.yaml` and adjust for staging environment
3. **Production**: Use `prod-example.yaml` as a starting point

### Docker Deployment

For Docker deployments, use the production configuration and pass environment variables:

```dockerfile
# Dockerfile
ENV TRADE_AGENT_ENVIRONMENT=production
COPY config/prod-example.yaml /app/config.yaml
```

```yaml
# docker-compose.yml
services:
  trading-agent:
    environment:
      - TRADE_AGENT_ALPACA_API_KEY=${ALPACA_API_KEY}
      - TRADE_AGENT_ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - TRADE_AGENT_ALPACA_BASE_URL=${ALPACA_BASE_URL}
```

### Kubernetes Deployment

For Kubernetes, use secrets for API keys:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: trading-rl-agent-secrets
type: Opaque
data:
  alpaca-api-key: <base64-encoded-key>
  alpaca-secret-key: <base64-encoded-secret>
  alpaca-base-url: <base64-encoded-url>
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-rl-agent
spec:
  template:
    spec:
      containers:
        - name: trading-agent
          env:
            - name: TRADE_AGENT_ALPACA_API_KEY
              valueFrom:
                secretKeyRef:
                  name: trading-rl-agent-secrets
                  key: alpaca-api-key
```

## Validation

The configuration system automatically validates all settings:

```python
# This will raise validation errors for invalid values
settings = load_settings("config/invalid-config.yaml")
```

Common validation errors:

- Invalid data types (string instead of int)
- Out-of-range values (negative percentages)
- Missing required fields
- Invalid enum values

## Best Practices

1. **Environment Separation**: Use different configs for different environments
2. **Secrets Management**: Never commit API keys to version control
3. **Validation**: Always validate configurations before deployment
4. **Documentation**: Add comments to explain non-standard settings
5. **Version Control**: Track configuration changes in version control
6. **Backup**: Keep backups of production configurations
7. **Testing**: Test configurations in staging before production

## Troubleshooting

### Common Issues

1. **Configuration Not Found**: Ensure the config file path is correct
2. **Validation Errors**: Check field types and value ranges
3. **API Key Issues**: Verify environment variables are set correctly
4. **Permission Errors**: Check file and directory permissions

### Debug Mode

Enable debug mode to see detailed configuration loading:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
settings = load_settings()
```

## Migration

When updating the configuration schema:

1. Update the Pydantic models in `config.py`
2. Update the example configurations
3. Test with existing configurations
4. Document breaking changes
5. Provide migration scripts if needed

## Support

For configuration issues:

1. Check the validation error messages
2. Review the example configurations
3. Verify environment variables are set correctly
4. Check the logs for detailed error information
