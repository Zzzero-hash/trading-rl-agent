# Trading RL Agent - CLI Usage Guide

A production-grade trading system that combines CNN+LSTM supervised learning with deep reinforcement learning for algorithmic trading.

## 📦 Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### From PyPI (Future Release)

```bash
pip install trading-rl-agent
```

### Docker Installation

```bash
# Build the Docker image
docker build -t trading-rl-agent .

# Run with configuration
docker run -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data trading-rl-agent --help
```

## ⚙️ Configuration

### Quick Setup

1. **Copy example configuration**:

   ```bash
   cp config/local-example.yaml config/my-config.yaml
   ```

2. **Set up environment variables** (create `.env` file):

   ```bash
   # Required for live trading
   ALPACA_API_KEY=your_alpaca_api_key_here
   ALPACA_SECRET_KEY=your_alpaca_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets

   # Optional data sources
   ALPHAVANTAGE_API_KEY=your_alphavantage_key_here
   NEWSAPI_KEY=your_newsapi_key_here
   ```

3. **Customize configuration**:

   ```yaml
   # config/my-config.yaml
   environment: development
   debug: true

   data:
     symbols: ["AAPL", "GOOGL", "MSFT"]
     start_date: "2023-01-01"
     end_date: "2024-01-01"

   execution:
     paper_trading: true # Use paper trading for development
   ```

### Environment Variable Fallbacks

The system automatically loads configuration from multiple sources in this order:

1. Default settings (embedded)
2. YAML configuration file (if provided)
3. `.env` file (if provided)
4. Environment variables with `TRADING_RL_AGENT_` prefix

```bash
# Environment variables override config file settings
export TRADING_RL_AGENT_ENVIRONMENT=production
export TRADING_RL_AGENT_DEBUG=false
export TRADING_RL_AGENT_DATA_SYMBOLS='["AAPL","GOOGL","MSFT","TSLA"]'
```

## 🚀 Basic Usage

### Command Structure

```bash
# Main CLI command
trading-rl-agent [OPTIONS] COMMAND [ARGS]...

# Or using Python module
python main.py [OPTIONS] COMMAND [ARGS]...
```

### Global Options

```bash
--config, -c PATH          Path to configuration file
--env-file PATH            Path to environment file (.env)
--verbose, -v              Increase verbosity (use multiple times)
--help, -h                 Show help message
```

### 1. System Information

```bash
# Show version and system info
trading-rl-agent version

# Show help for all commands
trading-rl-agent --help

# Show help for specific subcommand
trading-rl-agent data --help
```

### 2. Data Pipeline

```bash
# Download all configured datasets
trading-rl-agent data all

# Download specific symbols with custom parameters
trading-rl-agent data all \
  --symbols "AAPL,GOOGL,MSFT" \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --timeframe 1h \
  --source yfinance

# Process and build datasets
trading-rl-agent data process configs/dataset_config.yaml \
  --output outputs/datasets \
  --force-rebuild

# Run complete data pipeline
trading-rl-agent data pipeline configs/pipeline_config.yaml
```

### 3. Model Training

```bash
# Train CNN+LSTM model with default config
trading-rl-agent train cnn-lstm

# Train with custom configuration
trading-rl-agent train cnn-lstm \
  --config config/local-example.yaml \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --gpu \
  --output models/cnn_lstm/

# Train RL agent (PPO, SAC, TD3)
trading-rl-agent train rl sac \
  --config config/local-example.yaml \
  --timesteps 1000000 \
  --output models/rl/

# Train with custom parameters
trading-rl-agent train rl ppo \
  --timesteps 500000 \
  --ray-address ray://localhost:10001 \
  --workers 8
```

### 4. Backtesting

```bash
# Run backtesting with default strategy
trading-rl-agent backtest strategy \
  --data-path data/historical_data.csv \
  --model models/best_model.pth \
  --initial-capital 100000 \
  --commission 0.001

# Run with custom parameters
trading-rl-agent backtest strategy \
  --data-path data/AAPL_1h.csv \
  --model models/sac_agent.zip \
  --initial-capital 50000 \
  --commission 0.0005 \
  --slippage 0.0001 \
  --output backtest_results/
```

### 5. Live Trading

```bash
# Start paper trading session
trading-rl-agent trade start \
  --config config/local-example.yaml \
  --symbols "AAPL,GOOGL,MSFT" \
  --paper \
  --initial-capital 100000

# Start live trading (REAL MONEY - use with caution!)
trading-rl-agent trade start \
  --config config/prod-example.yaml \
  --symbols "AAPL,GOOGL" \
  --model models/best_model.pth \
  --initial-capital 50000

# Start with custom parameters
trading-rl-agent trade start \
  --config config/my-config.yaml \
  --symbols "AAPL,GOOGL,MSFT,TSLA" \
  --max-position 0.05 \
  --stop-loss 0.02 \
  --take-profit 0.05 \
  --update-interval 5
```

## 🐳 Container Usage

### Docker Compose

```yaml
# docker-compose.yml
version: "3.8"
services:
  trading-agent:
    build: .
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - TRADING_RL_AGENT_ALPACA_API_KEY=${ALPACA_API_KEY}
      - TRADING_RL_AGENT_ALPACA_SECRET_KEY=${ALPACA_SECRET_KEY}
      - TRADING_RL_AGENT_ALPACA_BASE_URL=${ALPACA_BASE_URL}
    command:
      [
        "trading-rl-agent",
        "trade",
        "start",
        "--config",
        "/app/config/prod-example.yaml",
      ]
```

```bash
# Start with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f trading-agent

# Stop
docker-compose down
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-rl-agent
spec:
  replicas: 1
  selector:
    matchLabels:
      app: trading-rl-agent
  template:
    metadata:
      labels:
        app: trading-rl-agent
    spec:
      containers:
        - name: trading-agent
          image: trading-rl-agent:latest
          command: ["trading-rl-agent", "trade", "start"]
          args: ["--config", "/app/config/prod-example.yaml"]
          env:
            - name: TRADING_RL_AGENT_ALPACA_API_KEY
              valueFrom:
                secretKeyRef:
                  name: trading-rl-agent-secrets
                  key: alpaca-api-key
            - name: TRADING_RL_AGENT_ALPACA_SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: trading-rl-agent-secrets
                  key: alpaca-secret-key
          volumeMounts:
            - name: config-volume
              mountPath: /app/config
            - name: data-volume
              mountPath: /app/data
            - name: models-volume
              mountPath: /app/models
      volumes:
        - name: config-volume
          configMap:
            name: trading-rl-agent-config
        - name: data-volume
          persistentVolumeClaim:
            claimName: trading-rl-agent-data
        - name: models-volume
          persistentVolumeClaim:
            claimName: trading-rl-agent-models
```

## 📊 Example Workflows

### Development Workflow

```bash
# 1. Set up development environment
cp config/local-example.yaml config/dev-config.yaml
# Edit config/dev-config.yaml for your needs

# 2. Download test data
trading-rl-agent data all --config config/dev-config.yaml

# 3. Train model
trading-rl-agent train cnn-lstm --config config/dev-config.yaml

# 4. Test with paper trading
trading-rl-agent trade start --config config/dev-config.yaml --paper
```

### Production Workflow

```bash
# 1. Set up production environment
cp config/prod-example.yaml config/prod-config.yaml
# Edit config/prod-config.yaml for production settings

# 2. Set production environment variables
export TRADING_RL_AGENT_ENVIRONMENT=production
export TRADING_RL_AGENT_DEBUG=false

# 3. Download production data
trading-rl-agent data all --config config/prod-config.yaml

# 4. Train production model
trading-rl-agent train cnn-lstm --config config/prod-config.yaml --gpu

# 5. Start live trading
trading-rl-agent trade start --config config/prod-config.yaml
```

## 🔧 Troubleshooting

### Common Issues

1. **Configuration not found**:

   ```bash
   # Ensure config file exists and path is correct
   ls -la config/local-example.yaml
   trading-rl-agent --config config/local-example.yaml version
   ```

2. **API key issues**:

   ```bash
   # Verify environment variables are set
   echo $TRADING_RL_AGENT_ALPACA_API_KEY

   # Test API connection
   trading-rl-agent trade start --paper --symbols AAPL
   ```

3. **GPU not available**:

   ```bash
   # Check GPU availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

   # Use CPU fallback
   trading-rl-agent train cnn-lstm --device cpu
   ```

4. **Memory issues**:
   ```bash
   # Reduce batch size and workers
   trading-rl-agent train cnn-lstm --batch-size 16 --workers 2
   ```

### Debug Mode

```bash
# Enable verbose logging
trading-rl-agent --verbose --verbose data all

# Check configuration loading
trading-rl-agent --config config/local-example.yaml version
```

## 📚 Next Steps

- **Configuration**: See `config/README.md` for detailed configuration options
- **Development**: Check `CONTRIBUTING.md` for development guidelines
- **Architecture**: Review `docs/` for system architecture and design
- **Examples**: Explore `examples/` for usage examples and tutorials

## 🆘 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check the `docs/` directory for detailed guides
- **Community**: Join our Discord/Slack for community support
