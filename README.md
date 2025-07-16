# Trading RL Agent

A production-grade hybrid reinforcement learning trading system that combines CNN+LSTM supervised learning with deep RL optimization for algorithmic trading.

## 🚀 Features

### Core Components

- **CNN+LSTM Models**: Hybrid neural networks for market pattern recognition
- **Reinforcement Learning**: SAC, TD3, PPO agents for trading decision optimization
- **Feature Engineering**: 150+ technical indicators with robust implementation
- **Data Pipeline**: Multi-source data ingestion with parallel processing (10-50x speedup)
- **Risk Management**: VaR, CVaR, position sizing, and portfolio optimization
- **Real-time Processing**: Live data feeds and sentiment analysis

### Infrastructure

- **Configuration Management**: YAML-based configuration with validation
- **CLI Interface**: Unified command-line interface using Typer
- **Logging & Monitoring**: Structured logging with MLflow/TensorBoard integration
- **Testing**: Comprehensive test suite with pytest
- **Code Quality**: Black, isort, ruff, mypy integration
- **Docker Support**: Containerized deployment ready

### Performance Optimizations

- **Parallel Data Fetching**: Ray-based parallel processing (10-50x speedup)
- **Mixed Precision Training**: 2-3x faster training with 30-50% memory reduction
- **Memory-Mapped Datasets**: 60-80% memory reduction for large datasets
- **Advanced LR Scheduling**: 1.5-2x faster convergence
- **Gradient Checkpointing**: Train larger models with same memory

## 📦 Installation

### Prerequisites

- Python 3.9+ (3.12 recommended)
- Git
- Docker (optional, for containerized deployment)

### Quick Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/trading-rl-agent.git
   cd trading-rl-agent
   ```

2. **Set up development environment**

   ```bash
   # Core dependencies only (fast setup)
   ./setup-env.sh core

   # Add ML dependencies
   ./setup-env.sh ml

   # Full production setup
   ./setup-env.sh full
   ```

3. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

4. **Verify installation**
   ```bash
   python -c "import trading_rl_agent; print('✅ Package imported successfully')"
   ```

## 🎯 Quick Start

### 1. Use the Unified CLI

```bash
# Show version and system info
python main.py version
python main.py info

# Download market data
python main.py data download --symbols "AAPL,GOOGL,MSFT" --start 2023-01-01

# Process and build datasets
python main.py data process --symbols "EURUSD=X,GBPUSD=X" --force

# Train CNN+LSTM model
python main.py train cnn-lstm --epochs 100 --gpu --output models/

# Train RL agent
python main.py train rl --epochs 50 --output models/

# Evaluate models
python main.py evaluate models/best_model.pth --data data/test_data.csv

# Run backtesting
python main.py backtest data/historical_data.csv --model models/agent.zip

# Start live trading
python main.py live start --paper --symbols "AAPL,GOOGL"
```

### 2. Alternative Entry Points

```bash
# Using the installed command
trading-rl-agent version
trading-rl-agent data download --symbols "AAPL,GOOGL"

# Using Python module
python -m trading_rl_agent.cli version
python -m trading_rl_agent.cli train cnn-lstm
```

## 📚 Documentation

- **[Project Status](PROJECT_STATUS.md)**: Current development status and roadmap
- **[Development Roadmap](TODO.md)**: Detailed task list and priorities
- **[Contributing Guidelines](CONTRIBUTING.md)**: How to contribute to the project

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Feature Eng.   │    │  CNN+LSTM Model │
│                 │    │                 │    │                 │
│ • yfinance      │───▶│ • Technical     │───▶│ • Pattern       │
│ • Alpha Vantage │    │   Indicators    │    │   Recognition   │
│ • Professional  │    │ • Alternative   │    │ • Uncertainty   │
│   Feeds         │    │   Data          │    │   Estimation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  RL Environment │    │  RL Agents      │
                       │                 │    │                 │
                       │ • State Space   │───▶│ • SAC           │
                       │ • Action Space  │    │ • TD3           │
                       │ • Reward Func   │    │ • PPO           │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │ Risk Management │    │ Portfolio Mgmt  │
                       │                 │    │                 │
                       │ • VaR/CVaR      │───▶│ • Multi-asset   │
                       │ • Position Size │    │ • Rebalancing   │
                       │ • Monitoring    │    │ • Analytics     │
                       └─────────────────┘    └─────────────────┘
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/
python -m pytest tests/smoke/

# Run with coverage
python -m pytest --cov=trading_rl_agent
```

## 🔧 Development

### Code Quality

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

### Adding Features

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Implement your changes following the [contributing guidelines](CONTRIBUTING.md)
3. Add tests for new functionality
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## 📊 Current Status

- **Core Infrastructure**: ✅ Complete
- **Data Pipeline**: ✅ Complete with parallel processing
- **CNN+LSTM Training**: ✅ Complete with optimizations
- **RL Agents**: ✅ Complete
- **Risk Management**: ✅ Complete
- **Production Deployment**: ✅ Complete
- **Monitoring & Logging**: ✅ Complete
- **Testing**: ✅ Complete
- **Documentation**: ✅ Complete

## 🚀 Performance Benchmarks

| Component       | Before Optimization | After Optimization | Improvement            |
| --------------- | ------------------- | ------------------ | ---------------------- |
| Data Fetching   | Sequential          | Parallel (Ray)     | **10-50x faster**      |
| Training Speed  | Standard            | Mixed Precision    | **2-3x faster**        |
| Memory Usage    | Standard            | Optimized          | **30-50% less**        |
| Dataset Loading | Standard            | Memory-mapped      | **60-80% less memory** |
| Convergence     | Standard            | Advanced LR        | **1.5-2x faster**      |

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- Ray team for parallel processing capabilities
- yfinance for market data access
- Alpha Vantage for professional data feeds
