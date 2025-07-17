# Trading RL Agent - Demo Commands

This document provides a series of commands to showcase the capabilities of the Trading RL Agent software.

## 🚀 Quick Start Commands

### 1. System Information
```bash
# Show version and system info
python main.py version
python main.py info

# Show help for all commands
python main.py --help
```

### 2. Data Pipeline Commands

#### Download Market Data
```bash
# Download single symbol
python main.py data download --symbols "AAPL" --start 2024-01-01 --end 2024-12-31

# Download multiple symbols
python main.py data download --symbols "AAPL,GOOGL,MSFT,TSLA" --start 2024-01-01

# Download with custom timeframe
python main.py data download --symbols "EURUSD=X" --timeframe 1h --start 2024-01-01

# Download to specific directory
python main.py data download --symbols "AAPL" --output data/market_data/
```

#### Process and Build Datasets
```bash
# Process downloaded data
python main.py data process --symbols "AAPL,GOOGL" --force

# Process with custom configuration
python main.py data process --config configs/unified_config.yaml --output processed_data/

# Run complete data pipeline
python main.py data pipeline configs/pipeline_config.yaml
```

### 3. Model Training Commands

#### Train CNN+LSTM Model
```bash
# Basic training
python main.py train cnn-lstm --epochs 100

# Training with custom parameters
python main.py train cnn-lstm \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 0.0005 \
    --gpu \
    --output models/cnn_lstm/

# Training with configuration file
python main.py train cnn-lstm --config configs/cnn_lstm_training.yaml
```

#### Train Reinforcement Learning Agents
```bash
# Train SAC agent
python main.py train rl sac --timesteps 1000000 --output models/sac/

# Train TD3 agent
python main.py train rl td3 --timesteps 500000 --output models/td3/

# Train PPO agent
python main.py train rl ppo --timesteps 750000 --output models/ppo/

# Training with Ray for distributed computing
python main.py train rl sac --ray-address ray://localhost:10001 --workers 8
```

### 4. Model Evaluation Commands

```bash
# Evaluate trained model
python main.py evaluate models/best_model.pth --data data/test_data.csv

# Evaluate with custom metrics
python main.py evaluate models/agent.zip \
    --data data/historical_data.csv \
    --metrics "sharpe_ratio,calmar_ratio,max_drawdown" \
    --output evaluation_results/
```

### 5. Backtesting Commands

```bash
# Basic backtesting
python main.py backtest strategy --data-path data/AAPL_1d.csv --model models/agent.zip

# Backtesting with custom parameters
python main.py backtest strategy \
    --data-path data/historical_data.csv \
    --model models/sac_agent.zip \
    --initial-capital 100000 \
    --commission 0.001 \
    --slippage 0.0001 \
    --output backtest_results/

# Backtesting with risk management
python main.py backtest strategy \
    --data-path data/portfolio_data.csv \
    --model models/ensemble_model.pth \
    --risk-management "var,cvar" \
    --position-sizing "kelly" \
    --output risk_managed_results/
```

### 6. Live Trading Commands

```bash
# Start paper trading
python main.py live start --paper --symbols "AAPL,GOOGL"

# Start live trading (requires API keys)
python main.py live start --symbols "AAPL,GOOGL" --config configs/production.yaml

# Monitor trading performance
python main.py live status

# Stop trading
python main.py live stop
```

## 🎯 Advanced Feature Commands

### 7. Feature Engineering

```bash
# Run market pattern generation demo
python examples/enhanced_market_patterns_demo.py

# Run ensemble trading example
python examples/ensemble_trading_example.py

# Run scenario evaluation
python examples/scenario_evaluation_example.py
```

### 8. Configuration Management

```bash
# Show configuration examples
python examples/config_example.py

# Validate configuration
python main.py config validate --file configs/production.yaml

# Generate configuration template
python main.py config template --output my_config.yaml
```

### 9. Risk Management

```bash
# Calculate VaR and CVaR
python main.py risk calculate \
    --data data/portfolio_data.csv \
    --confidence-level 0.95 \
    --time-horizon 1

# Portfolio optimization
python main.py portfolio optimize \
    --data data/multi_asset_data.csv \
    --method "efficient_frontier" \
    --constraints "max_weight:0.3,min_weight:0.05"
```

### 10. Performance Monitoring

```bash
# Run comprehensive tests
python -m pytest tests/ --cov=trading_rl_agent

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v

# Generate performance report
python main.py report performance --data data/results.csv --output reports/
```

## 🔧 Development Commands

### 11. Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
mypy src/

# Run all quality checks
python run_comprehensive_tests.py --quality-only
```

### 12. Docker Commands

```bash
# Build Docker image
docker build -t trading-rl-agent .

# Run with configuration
docker run -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data trading-rl-agent version

# Run training in container
docker run --gpus all -v $(pwd)/data:/app/data trading-rl-agent train cnn-lstm --epochs 100
```

## 📊 Example Workflows

### Complete Trading Pipeline
```bash
# 1. Download data
python main.py data download --symbols "AAPL,GOOGL,MSFT" --start 2023-01-01

# 2. Process data
python main.py data process --symbols "AAPL,GOOGL,MSFT" --force

# 3. Train CNN+LSTM model
python main.py train cnn-lstm --epochs 100 --output models/cnn_lstm/

# 4. Train RL agent
python main.py train rl sac --timesteps 500000 --output models/rl/

# 5. Evaluate models
python main.py evaluate models/best_model.pth --data data/test_data.csv

# 6. Run backtesting
python main.py backtest strategy --data-path data/AAPL_1d.csv --model models/rl/sac_agent.zip

# 7. Start paper trading
python main.py live start --paper --symbols "AAPL,GOOGL"
```

### Quick Demo Workflow
```bash
# Run quick demo
./quick_demo.sh

# Or run comprehensive demo
./demo_showcase.sh
```

## 🎯 Performance Benchmarks

The system includes several optimizations:

- **Parallel Data Fetching**: 10-50x speedup with Ray
- **Mixed Precision Training**: 2-3x faster training
- **Memory-Mapped Datasets**: 60-80% memory reduction
- **Advanced LR Scheduling**: 1.5-2x faster convergence

## 📚 Additional Resources

- **Main Documentation**: `README.md`
- **CLI Usage Guide**: `README_CLI_USAGE.md`
- **Contributing Guidelines**: `CONTRIBUTING.md`
- **Project Status**: `PROJECT_STATUS.md`
- **Example Scripts**: `examples/` directory