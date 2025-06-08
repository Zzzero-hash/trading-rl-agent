# Trading RL Agent

A sophisticated reinforcement learning framework for algorithmic trading that combines deep learning models (CNN-LSTM) with reinforcement learning agents (SAC, TD3) for automated trading strategies.

## 🎯 Project Overview

**Status: Phase 1 COMPLETE ✅ | Ready for Phase 2 Deep RL Ensemble**

This project implements an end-to-end trading system featuring:
- **Live data ingestion** with sentiment analysis integration
- **CNN-LSTM hybrid models** for time-series prediction  
- **Deep RL ensemble training** (SAC, TD3, ensemble methods)
- **Comprehensive backtesting** with risk management
- **Production deployment** with monitoring and alerts

### 📊 Current Achievements
- **Model**: CNN-LSTM with 19,843 parameters, forward pass validated
- **Data**: 3,827 samples, 26 features, 3,817 sequences (length 10)
- **Training**: Basic loop functional with loss: 1.0369
- **Tests**: 5/5 integration tests passing, 75+ unit tests
- **Pipeline**: Sentiment analysis module with mock data fallback
- **Codebase**: Repository optimized and cleaned (June 8, 2025)

## 📚 Documentation Structure

- **[README.md](README.md)** (this file) - Project overview, installation, and usage
- **[ROADMAP.md](ROADMAP.md)** - Detailed development phases and milestones

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Current Status](#current-status)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Testing](#testing)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [License](#license)

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/trading-rl-agent.git
cd trading-rl-agent
./setup-env.sh

# 2. Run integration tests
python quick_integration_test.py

# 3. Train CNN-LSTM model
python src/train_cnn_lstm.py

# 4. Start Phase 2 RL training (coming next)
# python src/train_rl.py --agent sac
```

## Installation

### Local Development
```bash
# Setup virtual environment and install dependencies
./setup-env.sh

# Or manually:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Docker Development
```bash
# Build Docker image (CPU)
docker build -t trading-rl-agent .

# Build GPU/ROCm image
docker build -f Dockerfile.rocm -t trading-rl-agent:rocm .

# Run interactive container
docker run --rm -it \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/data:/app/data" \
  -w /app \
  trading-rl-agent bash
```

## Current Status & Critical Issues

### ✅ **Working Components**
- Sample data generation (3,827 samples, 26 features)
- Sentiment analysis module with mock fallback
- CNN-LSTM model (19,843 parameters)
- Data preprocessing pipeline (3,817 sequences)
- Basic training loop with loss calculation

### 🔧 **Critical TODOs Before Production**
- **✅ COMPLETED**: Repository cleanup and optimization (June 8, 2025)
- **HIGH**: Fix sentiment timestamp comparison errors
- **HIGH**: Address severe label imbalance (Class 0: 52, Class 1: 384, Class 2: 3391)
- **HIGH**: Implement proper SentimentData return types
- **MEDIUM**: Add NaN handling in preprocessing
- **MEDIUM**: Optimize sequence generation for larger datasets

## Project Structure

```text
trading-rl-agent/
├── src/                      # Source code
│   ├── main.py               # CLI entry point
│   ├── data_pipeline.py      # Data processing pipeline
│   ├── train_cnn_lstm.py     # CNN-LSTM training
│   ├── train_rl.py          # RL agent training
│   ├── agents/               # RL agents (SAC, TD3, ensemble)
│   ├── models/               # Neural network architectures
│   ├── envs/                 # Trading environments
│   ├── data/                 # Data utilities and sentiment
│   ├── configs/              # YAML configuration files
│   └── utils/                # Metrics and helper functions
├── tests/                    # Unit and integration tests
├── data/                     # Training data
├── requirements*.txt         # Dependencies
├── Dockerfile*               # Container configurations
└── setup-env.sh             # Environment setup script
```

## Usage

### CNN-LSTM Training
```bash
# Train the time-series prediction model
python src/train_cnn_lstm.py

# Generate sample data first if needed
python generate_sample_data.py
```

### RL Agent Training (Phase 2)
```bash
# Train SAC agent (when implemented)
python src/train_rl.py --agent sac --env trader_env

# Train ensemble of agents
python src/train_rl.py --agent ensemble
```

### CLI Interface
```bash
# Using the main CLI (after setup-env.sh)
trade-agent \
  --env-config src/configs/env/trader_env.yaml \
  --model-config src/configs/model/cnn_lstm.yaml \
  --trainer-config src/configs/trainer/default.yaml \
  --train
```

### Ray Distributed Training
```bash
# Start Ray cluster
ray start --head

# Distributed training
python src/train_rl.py --cluster-config ray_cluster_setup.yaml

# Hyperparameter tuning with Ray Tune
trade-agent \
  --env-config src/configs/ray/tune_search.yaml \
  --tune

ray stop
```

### Data Pipeline
```bash
# Process historical data with Ray parallelization
export RAY_ADDRESS="ray://head-node:10001"
python -m src.data.pipeline --config src/configs/data/pipeline.yaml
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_data_pipeline.py -v
pytest tests/test_cnn_lstm.py -v

# Run integration tests
python quick_integration_test.py

# Docker-based testing
./test-all.sh
```

## Development Roadmap

### ✅ Phase 1: Data & Modeling (COMPLETED)
- ✅ Data pipeline with technical indicators + sentiment
- ✅ CNN-LSTM hybrid model implementation
- ✅ Basic training loop and validation
- ✅ Integration tests passing

### 🔄 Phase 2: Deep RL Ensemble (CURRENT)
- **SAC Agent**: Implement Soft Actor-Critic in `src/agents/sac_agent.py`
- **TD3 Agent**: Twin Delayed DDPG implementation
- **Ensemble Framework**: Voting and dynamic weight adjustment
- **Testing**: Unit tests and Ray RLlib integration

### 🏦 Phase 3: Portfolio & Risk Management
- Multi-asset portfolio environment
- Risk manager with drawdown protection
- Risk-adjusted reward functions
- Transaction cost and slippage modeling

### 📊 Phase 4: Metrics & Backtesting
- Trading metrics (Sharpe, Sortino, drawdown)
- Event-driven backtesting engine
- Performance visualization and reporting
- Automated CI backtesting

### 🚀 Phase 5: Production Deployment
- Model serving API with Ray Serve
- Monitoring and alerting systems
- Docker/Kubernetes deployment
- Real-time execution with fail-safes

## Advanced Features

### Sentiment Analysis Integration
```python
from src.data.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()
sentiment = analyzer.get_symbol_sentiment("AAPL", "2024-01-01")
```

### Trading Environment
```python
from src.envs.trading_env import TradingEnv

env = TradingEnv({
    "dataset_paths": ["data/sample_data.csv"], 
    "window_size": 10,
    "initial_balance": 10000
})
obs, _ = env.reset()
```

### Ray Serve Deployment
```bash
# Start deployment
ray start --head
python -m ray serve run src.serve_deployment:deployment_graph

# Send prediction requests
curl -X POST http://127.0.0.1:8000/predictor \
  -d '{"features": [0.1, 0.2, 0.3]}'
```

## Performance Targets

- **CNN-LSTM**: Prediction accuracy > baseline (✅ functional)
- **SAC/TD3**: Outperform baseline strategies
- **Ensemble**: Reduce variance, increase return stability  
- **Backtesting**: Sharpe > 1.0, max drawdown < 15%
- **Production**: API latency < 100ms, 99% uptime

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add comprehensive tests for your changes
4. Ensure all tests pass (`pytest tests/ -v`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests for new features
- Update documentation for API changes
- Use Docker for consistent testing environment

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with PyTorch, Ray RLlib, and Gymnasium
- Technical indicators via TA-Lib
- Financial data from yfinance
- Distributed computing with Ray

---

**Next Priority**: Implement SAC agent and begin ensemble framework development for Phase 2.
