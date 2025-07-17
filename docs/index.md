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
- [Testing Guide](TESTING_GUIDE.md) - Testing strategy and guidelines
- [Test Suite](../tests/) - Test suite and examples

## 🏗️ **Current Status**

### **✅ Implemented Features**

- **Data Pipeline**: Multi-source data ingestion (yfinance, Alpha Vantage, synthetic)
- **Feature Engineering**: 150+ technical indicators with robust preprocessing
- **CNN+LSTM Models**: Hybrid neural network architecture
- **Configuration System**: YAML-based configuration management
- **CLI Interface**: Unified command-line interface
- **Code Quality**: Comprehensive linting, formatting, and testing setup

### **🔄 In Progress**

- **Testing & Quality Assurance**: 8% complete (3.91% coverage)
- **CNN+LSTM Training**: Complete training pipeline with monitoring
- **Integration Tests**: End-to-end workflow testing
- **Model Evaluation**: Comprehensive metrics and validation

### **📋 Planned Features**

- **RL Agent Training**: SAC, TD3, PPO agent implementation
- **Risk Management**: VaR, CVaR, Kelly criterion position sizing
- **Portfolio Management**: Multi-asset portfolio optimization
- **Live Trading**: Real-time execution engine
- **Monitoring**: Performance dashboards and alerting

## 🧪 **Testing Status**

### **Current Coverage: 3.91%**

**Well-Tested Components:**

- ✅ Core Configuration System (82.32% coverage)
- ✅ Agent Configurations (88.06% coverage)
- ✅ Exception Handling (100% coverage)

**Needs Testing (Priority Order):**

1. 🔄 Risk Management (13.14% coverage) - **Critical Priority**
2. 🔄 CLI Interface (0% coverage) - **High Priority**
3. 🔄 Data Pipeline Components (0% coverage)
4. 🔄 Model Training Scripts (0% coverage)
5. 🔄 Portfolio Management (0% coverage)
6. 🔄 Feature Engineering (0% coverage)
7. 🔄 Evaluation Components (0% coverage)
8. 🔄 Monitoring Components (0% coverage)

**Test Infrastructure:**

- 54 test files covering unit and integration tests
- pytest framework with coverage reporting
- Comprehensive test configuration

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

# Run tests (current coverage: 3.91%)
python -m pytest --cov=trading_rl_agent
```

## 📖 **Examples**

See the [examples](examples.md) page for working code examples and tutorials.

## 🚨 **Critical Priorities**

### **Immediate Actions Required**

1. **Testing Coverage Improvement** (Priority 1)
   - Focus on CLI interface testing (0% coverage)
   - Implement data pipeline component tests
   - Add model training script tests
   - Target: Achieve 50% coverage within 2 weeks

2. **Integration Testing** (Priority 2)
   - End-to-end workflow testing
   - Cross-module integration tests
   - Performance regression testing

3. **Documentation Updates** (Priority 3)
   - Update API documentation for tested components
   - Add testing guidelines and examples
   - Improve troubleshooting guides

## 🆘 **Support**

- **Issues**: [GitHub Issues](https://github.com/your-org/trading-rl-agent/issues)
- **Documentation**: [docs/](docs/) - Comprehensive guides and API reference
- **Tests**: [tests/](tests/) - Reference implementations and test cases
- **Roadmap**: [TODO.md](../TODO.md) - Current development status and roadmap
- **Project Status**: [PROJECT_STATUS.md](../PROJECT_STATUS.md) - Detailed project status

---

**Ready to build the future of algorithmic trading with hybrid RL systems!**

**Note**: This project is in active development. Current test coverage is 3.91% and improving. Please check the [Project Status](../PROJECT_STATUS.md) for the latest updates.
