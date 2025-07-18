# Getting Started with Trading RL Agent

This guide will help you get up and running with the Trading RL Agent, a production-grade hybrid reinforcement learning trading system.

## 🚀 Quick Installation

### Prerequisites

- **Python**: 3.9+ (3.12 recommended)
- **Git**: For cloning the repository
- **Docker**: Optional, for containerized deployment
- **GPU**: Optional, for accelerated training (CUDA 11.8+)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/trading-rl-agent.git
cd trading-rl-agent
```

### Step 2: Set Up Environment

Choose one of the following setup options:

#### Option A: Full Setup (Recommended)

```bash
# Full production setup with all dependencies
./setup-env.sh full
```

#### Option B: Core Setup (Fast)

```bash
# Core dependencies only
./setup-env.sh core

# Add ML dependencies later
./setup-env.sh ml
```

#### Option C: Manual Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements.dev.txt
```

### Step 3: Verify Installation

```bash
# Test basic import
python -c "import trading_rl_agent; print('✅ Package imported successfully')"

# Check CLI
python main.py version

# Show system info
python main.py info
```

## 🎯 First Steps

### 1. Download Market Data

```bash
# Download data for popular stocks
python main.py data download --symbols "AAPL,GOOGL,MSFT" --start 2023-01-01

# Download forex data
python main.py data download --symbols "EURUSD=X,GBPUSD=X" --start 2023-01-01

# Download with custom parameters
python main.py data download \
    --symbols "AAPL,GOOGL,MSFT" \
    --start 2023-01-01 \
    --end 2024-01-01 \
    --timeframe "1d" \
    --output data/market_data
```

### 2. Process and Build Datasets

```bash
# Process downloaded data into training datasets
python main.py data process --force

# Standardize data for consistent training
python main.py data standardize --method robust

# Build complete pipeline
python main.py data pipeline
```

### 3. Train Your First Model

#### CNN+LSTM Model

```bash
# Basic training
python main.py train cnn-lstm --epochs 100 --output models/cnn_lstm

# With GPU acceleration
python main.py train cnn-lstm \
    --epochs 100 \
    --gpu \
    --mixed-precision \
    --output models/cnn_lstm_gpu

# With custom parameters
python main.py train cnn-lstm \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --output models/cnn_lstm_custom
```

#### Reinforcement Learning Agent

```bash
# Train PPO agent
python main.py train rl --agent-type ppo --timesteps 1000000

# Train SAC agent
python main.py train rl --agent-type sac --timesteps 1000000

# Train with Ray cluster
python main.py train rl \
    --agent-type ppo \
    --timesteps 1000000 \
    --ray-address "ray://localhost:10001" \
    --num-workers 4
```

### 4. Evaluate Your Models

```bash
# Evaluate CNN+LSTM model
python main.py evaluate models/cnn_lstm/best_model.pth --data data/test_data.csv

# Run backtesting
python main.py backtest strategy \
    --data data/historical_data.csv \
    --model models/agent.zip \
    --initial-capital 10000

# Compare multiple models
python main.py backtest compare \
    --models "models/model1.pth,models/model2.pth" \
    --data data/test_data.csv
```

### 5. Run Paper Trading

```bash
# Start paper trading session
python main.py trade start \
    --symbols "AAPL,GOOGL" \
    --paper-trading \
    --initial-capital 100000

# Monitor trading session
python main.py trade monitor --session-id <session_id>

# Stop trading session
python main.py trade stop --session-id <session_id>
```

## 📊 Understanding the Results

### Model Performance Metrics

The system provides comprehensive performance metrics:

- **Returns**: Total return, annualized return, Sharpe ratio
- **Risk Metrics**: VaR, CVaR, maximum drawdown, volatility
- **Trading Metrics**: Win rate, profit factor, average trade
- **Risk-Adjusted Metrics**: Sortino ratio, Calmar ratio, information ratio

### Backtesting Results

Backtesting provides detailed analysis including:

- Equity curve and drawdown analysis
- Trade-by-trade breakdown
- Risk metrics and performance attribution
- Transaction cost analysis

## 🔧 Configuration

### Basic Configuration

The system uses YAML-based configuration. Create a `config.yaml` file:

```yaml
# Data configuration
data:
  symbols: ["AAPL", "GOOGL", "MSFT"]
  start_date: "2023-01-01"
  end_date: "2024-01-01"
  timeframe: "1d"
  source: "yfinance"

# Model configuration
model:
  type: "cnn_lstm"
  architecture:
    cnn_layers: [32, 64, 128]
    lstm_units: 128
    dense_layers: [64, 32]
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001

# Risk management
risk:
  max_position_size: 0.1
  stop_loss: 0.02
  take_profit: 0.04
  max_drawdown: 0.15
```

### Environment Variables

Set important environment variables:

```bash
# API keys (if using premium data sources)
export ALPHA_VANTAGE_API_KEY="your_key_here"
export YAHOO_FINANCE_API_KEY="your_key_here"

# Logging
export LOG_LEVEL="INFO"
export LOG_FILE="trading_rl_agent.log"

# Ray configuration
export RAY_ADDRESS="ray://localhost:10001"
export RAY_DISABLE_IMPORT_WARNING=1
```

## 🧪 Testing Your Setup

### Run the Test Suite

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=trading_rl_agent --cov-report=html
```

### Quick Validation

```bash
# Test data download
python main.py data download --symbols "AAPL" --start 2024-01-01 --end 2024-01-10

# Test model training (small scale)
python main.py train cnn-lstm --epochs 5 --output test_models/

# Test backtesting
python main.py backtest strategy --data data/AAPL_2024.csv --initial-capital 1000
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Ensure you're in the correct environment
source .venv/bin/activate  # or conda activate trading-rl-agent

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### 2. Ray Compatibility Issues

```bash
# Use sequential processing instead of Ray
python main.py data download --symbols "AAPL" --parallel false

# Or update Ray version
pip install "ray[default]>=2.8.0"
```

#### 3. GPU Issues

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU-only mode
python main.py train cnn-lstm --gpu false
```

#### 4. Data Download Issues

```bash
# Check internet connection
python -c "import yfinance; print(yfinance.download('AAPL', period='1d'))"

# Use different data source
python main.py data download --source "alpha_vantage" --symbols "AAPL"
```

### Getting Help

- **Documentation**: Check the [main documentation](index.md)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/trading-rl-agent/issues)
- **Discussions**: Join community discussions for questions

## 🎯 Next Steps

Now that you have the basic setup working, explore:

1. **[Advanced Training](enhanced_training_guide.md)** - Optimize your models
2. **[Risk Management](RISK_ALERT_SYSTEM.md)** - Implement proper risk controls
3. **[Live Trading](examples.md#live-trading)** - Deploy to production
4. **[Ensemble Methods](ENSEMBLE_SYSTEM_GUIDE.md)** - Combine multiple models
5. **[Custom Strategies](examples.md)** - Build your own trading strategies

---

_Happy trading! 🚀_
