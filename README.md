# Trading RL Agent - Hybrid Reinforcement Learning Trading System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **hybrid reinforcement learning trading system** that combines CNN+LSTM supervised learning with deep RL optimization. This project is currently in active development with a focus on building a production-ready algorithmic trading framework.

## 🏗️ **Current Architecture**

This system implements a **two-tier hybrid approach**:

- **🧠 Tier 1**: CNN+LSTM models for market pattern recognition and feature extraction
- **🤖 Tier 2**: Reinforcement Learning agents (SAC/TD3/PPO) for trading decision optimization
- **⚡ Production Layer**: Real-time execution, monitoring, and risk management (planned)

### **Key Components**

- ✅ **Data Pipeline**: Multi-source data ingestion (yfinance, Alpha Vantage, synthetic)
- ✅ **Feature Engineering**: 150+ technical indicators with robust preprocessing
- ✅ **CNN+LSTM Models**: Hybrid neural networks for market pattern recognition
- ✅ **RL Environment**: Gymnasium-based trading environments
- ✅ **Configuration Management**: YAML-based configuration system
- ✅ **CLI Interface**: Unified command-line interface for all operations
- 🔄 **Training Pipeline**: CNN+LSTM training infrastructure (in progress)
- 🔄 **RL Agent Training**: SAC/TD3/PPO agent training (planned)
- 🔄 **Risk Management**: VaR, CVaR, position sizing (planned)
- 🔄 **Live Trading**: Real-time execution engine (planned)

---

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements.dev.txt
```

### **2. Basic Usage**

```python
from trading_rl_agent import ConfigManager
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder

# Initialize configuration
config = ConfigManager("configs/development.yaml")

# Build dataset with features
builder = RobustDatasetBuilder()
dataset = builder.build_dataset(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2023-01-01",
    end_date="2024-01-01"
)

print(f"Dataset shape: {dataset.shape}")
print(f"Features: {dataset.columns.tolist()}")
```

### **3. CLI Usage**

```bash
# Generate synthetic data
python cli.py generate-data configs/finrl_synthetic_data.yaml --synthetic

# Train CNN+LSTM model
python cli.py train cnn-lstm configs/cnn_lstm_training.yaml

# Run backtest
python cli.py backtest data/price_data.csv --policy "lambda p: 'buy' if p > 100 else 'sell'"

# Evaluate trained agent
python cli.py evaluate data/test_data.csv checkpoints/agent.zip --agent sac
```

---

## 📦 **Project Structure**

```
trading-rl-agent/
├── src/trading_rl_agent/          # Main package
│   ├── core/                      # Core system components
│   │   ├── config.py             # Configuration management
│   │   ├── exceptions.py         # Custom exceptions
│   │   └── logging.py            # Logging utilities
│   ├── data/                      # Data processing
│   │   ├── robust_dataset_builder.py  # Main dataset builder
│   │   ├── features.py           # Feature engineering
│   │   ├── preprocessing.py      # Data preprocessing
│   │   ├── professional_feeds.py # Professional data feeds
│   │   ├── sentiment.py          # Sentiment analysis
│   │   └── synthetic.py          # Synthetic data generation
│   ├── agents/                    # RL agents & utilities
│   │   ├── configs.py            # Agent configurations
│   │   ├── policy_utils.py       # Policy utilities
│   │   └── trainer.py            # Training utilities
│   ├── models/                    # Neural network models
│   │   └── cnn_lstm.py           # CNN+LSTM architecture
│   ├── portfolio/                 # Portfolio management (planned)
│   ├── risk/                      # Risk management (planned)
│   ├── execution/                 # Order execution (planned)
│   ├── monitoring/                # Performance monitoring (planned)
│   └── utils/                     # Shared utilities
├── configs/                       # Configuration files
│   ├── production.yaml           # Production settings
│   ├── development.yaml          # Development settings
│   ├── cnn_lstm_training.yaml    # CNN+LSTM training config
│   └── finrl_*.yaml              # Data generation configs
├── tests/                         # Test suite
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
├── cli.py                         # Main CLI interface
├── train_cnn_lstm.py             # CNN+LSTM training script
├── evaluate.py                    # Model evaluation
└── requirements.txt               # Dependencies
```

---

## 🧠 **Core Components**

### **Data Pipeline**

The robust dataset builder provides comprehensive data processing:

```python
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder

# Configure dataset
config = DatasetConfig(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    features=["sma", "ema", "rsi", "macd", "bollinger_bands"],
    window_size=50,
    normalize=True
)

# Build dataset
builder = RobustDatasetBuilder()
dataset = builder.build_dataset(config)
```

### **Feature Engineering**

Comprehensive feature engineering with 150+ technical indicators:

- **📊 Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR, etc.
- **🔗 Cross-Asset Features**: Correlation, cointegration, regime detection
- **📰 Alternative Data**: News sentiment, economic indicators
- **🕒 Temporal Features**: Sine-cosine encoding for time patterns
- **� Microstructure**: Volume profile, order flow indicators

### **CNN+LSTM Models**

Hybrid neural networks for market pattern recognition:

```python
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

# Initialize model
model = CNNLSTMModel(
    input_dim=50,  # Number of features
    config={
        "cnn_filters": [32, 64, 128],
        "lstm_units": 256,
        "dropout_rate": 0.2,
        "uncertainty_estimation": True
    }
)
```

---

## ⚙️ **Configuration**

The system uses YAML-based configuration for all components:

```yaml
# configs/development.yaml
environment: development
debug: true

data:
  data_sources:
    primary: yfinance
    backup: alpha_vantage
  feature_window: 50
  symbols: ["AAPL", "GOOGL", "MSFT"]

training:
  cnn_lstm:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    early_stopping_patience: 10

  rl:
    agent_type: sac
    total_timesteps: 1000000
    num_workers: 4

risk:
  max_position_size: 0.1
  max_leverage: 1.0
  var_confidence_level: 0.05
```

---

## 🧪 **Testing & Development**

### **Running Tests**

```bash
# Install development dependencies
pip install -r requirements.dev.txt

# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m performance   # Performance tests

# Generate coverage report
pytest --cov=src tests/ --cov-report=html
```

### **Code Quality**

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
ruff check src/ tests/
mypy src/

# Security check
bandit -r src/
```

---

## � **Current Status**

### **✅ Completed Features**

- **Data Pipeline**: Multi-source data ingestion and preprocessing
- **Feature Engineering**: 150+ technical indicators with robust implementation
- **CNN+LSTM Models**: Hybrid neural network architecture
- **Configuration System**: YAML-based configuration management
- **CLI Interface**: Unified command-line interface
- **Code Quality**: Comprehensive linting, formatting, and testing setup
- **Documentation**: API documentation and development guides

### **🔄 In Progress**

- **CNN+LSTM Training**: Complete training pipeline with monitoring
- **Integration Tests**: End-to-end workflow testing
- **Model Evaluation**: Comprehensive metrics and validation

### **📋 Planned Features**

- **RL Agent Training**: SAC, TD3, PPO agent implementation
- **Risk Management**: VaR, CVaR, Kelly criterion position sizing
- **Portfolio Management**: Multi-asset portfolio optimization
- **Live Trading**: Real-time execution engine
- **Monitoring**: Performance dashboards and alerting
- **Deployment**: Docker and Kubernetes deployment

---

## 🛠️ **Development Setup**

### **Prerequisites**

- Python 3.9+
- Git
- Virtual environment (recommended)

### **Development Environment**

```bash
# Clone and setup
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.dev.txt

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

### **Project Configuration**

The project uses several configuration files:

- **pyproject.toml**: Project metadata and tool configurations
- **requirements.txt**: Production dependencies
- **requirements.dev.txt**: Development dependencies
- **configs/**: Application configuration files

---

## 🤝 **Contributing**

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- **Development Setup**: Environment configuration and tools
- **Code Standards**: Formatting, linting, and type checking
- **Testing Guidelines**: Unit, integration, and performance tests
- **Pull Request Process**: Review and merge procedures

### **Development Workflow**

```bash
# Setup development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.dev.txt

# Run quality checks
black src/ tests/
isort src/ tests/
ruff check src/ tests/
mypy src/

# Run tests
pytest tests/ -v
```

---

## 📄 **License**

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## ⚠️ **Disclaimer**

**For educational and research purposes only.** This system is designed for algorithmic trading research and development. Always:

- 📊 **Paper trade first**: Test strategies thoroughly before using real capital
- 🧠 **Understand the risks**: Trading involves substantial risk of loss
- 👨‍💼 **Consult professionals**: Seek professional advice before deploying capital
- ⚖️ **Follow regulations**: Ensure compliance with relevant financial regulations

---

## 🆘 **Support & Documentation**

- **📖 Documentation**: [docs/](docs/) - Comprehensive guides and API reference
- **💡 Examples**: [examples/](examples/) - Working code examples and tutorials
- **🧪 Tests**: [tests/](tests/) - Reference implementations and test cases
- **🐛 Issues**: [GitHub Issues](https://github.com/your-org/trading-rl-agent/issues) - Bug reports and feature requests
- **📋 TODO**: [TODO.md](TODO.md) - Current development status and roadmap

---

**🚀 Ready to build the future of algorithmic trading with hybrid RL systems!**

---
