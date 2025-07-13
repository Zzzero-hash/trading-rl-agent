# Trading RL Agent Documentation

Welcome to the Trading RL Agent documentation! This project is a hybrid reinforcement learning trading system that combines CNN+LSTM supervised learning with deep RL optimization.

## 📚 **Documentation Overview**

### **Getting Started**

- [Getting Started Guide](getting_started.md) - Quick start guide for new users
- [Development Guide](DEVELOPMENT_GUIDE.md) - Setup development environment
- [Evaluation Guide](EVALUATION_GUIDE.md) - Model evaluation and testing

### **Core Components**

- [Data Pipeline](../src/trading_rl_agent/data/) - Data ingestion and preprocessing
- [Feature Engineering](../src/trading_rl_agent/features/) - Technical indicators and feature engineering
- [CNN+LSTM Models](../src/trading_rl_agent/models/) - Neural network architectures
- [RL Agents](../src/trading_rl_agent/agents/) - Reinforcement learning agents
- [Configuration](../src/trading_rl_agent/core/) - Configuration management

### **API Reference**

- [Data Module](../src/trading_rl_agent/data/) - Data processing and dataset building
- [Models Module](../src/trading_rl_agent/models/) - Neural network models
- [Agents Module](../src/trading_rl_agent/agents/) - RL agents and training
- [Core Module](../src/trading_rl_agent/core/) - Core utilities and configuration

### **Development**

- [Contributing Guide](../CONTRIBUTING.md) - How to contribute to the project
- [Code Quality](../docs/PRE_COMMIT_SETUP.md) - Code formatting and linting
- [Testing](../tests/) - Test suite and examples

## 🏗️ **Current Status**

### **✅ Implemented Features**

- **Data Pipeline**: Multi-source data ingestion (yfinance, Alpha Vantage, synthetic)
- **Feature Engineering**: 150+ technical indicators with robust preprocessing
- **CNN+LSTM Models**: Hybrid neural network architecture
- **Configuration System**: YAML-based configuration management
- **CLI Interface**: Unified command-line interface
- **Code Quality**: Comprehensive linting, formatting, and testing setup

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

## 🚀 **Quick Start**

```bash
# Clone repository
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements.dev.txt
```

## 📖 **Examples**

See the [examples](examples.md) page for working code examples and tutorials.

## 🆘 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-org/trading-rl-agent/issues)
- **Documentation**: [docs/](docs/) - Comprehensive guides and API reference
- **Tests**: [tests/](tests/) - Reference implementations and test cases
- **Roadmap**: [TODO.md](../TODO.md) - Current development status and roadmap

---

**Ready to build the future of algorithmic trading with hybrid RL systems!**
