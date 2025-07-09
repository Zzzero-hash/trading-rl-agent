# Trading RL Agent - Production-Grade Hybrid Trading System

[![Coverage Status](https://codecov.io/gh/Zzzero-hash/trading-rl-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/Zzzero-hash/trading-rl-agent)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready hybrid reinforcement learning trading system** that combines CNN+LSTM supervised learning with deep RL optimization, featuring comprehensive risk management, real-time execution, and enterprise-grade monitoring.

## 🏗️ **Architecture Overview**

This system implements a sophisticated **two-tier hybrid approach**:

- **🧠 Tier 1**: Deep Learning (CNN+LSTM) for market pattern recognition and uncertainty quantification
- **🤖 Tier 2**: Reinforcement Learning (SAC/TD3/PPO) for trading decision optimization with risk controls
- **⚡ Production Layer**: Real-time execution, monitoring, and risk management

### **Key Innovations**

- ✅ **Uncertainty-Weighted Actions**: CNN+LSTM confidence scores guide RL position sizing
- ✅ **Multi-Asset Portfolio Optimization**: Modern Portfolio Theory integration
- ✅ **Production-Grade Risk Management**: VaR, CVaR, Kelly criterion position sizing
- ✅ **Real-Time Execution**: Sub-second order execution with smart routing
- ✅ **Enterprise Monitoring**: MLflow, Prometheus, and custom dashboards

---

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Clone repository
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Quick setup (recommended)
./setup-production.sh full

# Or manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-production.txt
```

### **2. Basic Usage**

```python
from trading_rl_agent import ConfigManager, PortfolioManager
from trading_rl_agent.agents import EnsembleAgent
from trading_rl_agent.data import DataPipeline

# Initialize system with production config
config = ConfigManager("configs/production.yaml")
data_pipeline = DataPipeline(config.data)
portfolio = PortfolioManager(initial_capital=100000)

# Create hybrid agent
agent = EnsembleAgent(config.agent)

# Train on historical data
agent.train(data_pipeline.get_training_data())

# Execute live trading (paper trading enabled by default)
portfolio.start_live_trading(agent, data_pipeline)
```

### **3. Configuration**

```yaml
# configs/production.yaml
environment: production
debug: false

data:
  data_sources:
    primary: alpaca
    backup: yfinance
  real_time_enabled: true
  feature_window: 50

agent:
  agent_type: sac
  ensemble_size: 3
  total_timesteps: 1000000

risk:
  max_position_size: 0.1
  max_leverage: 1.0
  var_confidence_level: 0.05

execution:
  broker: alpaca
  paper_trading: false # Set to true for paper trading
  order_timeout: 60
```

---

## 📦 **Project Structure**

```
trading-rl-agent/
├── src/trading_rl_agent/          # Main package
│   ├── core/                      # Core system components
│   ├── agents/                    # RL agents & ensemble methods
│   ├── data/                      # Data ingestion & processing
│   ├── features/                  # Feature engineering pipeline
│   ├── models/                    # CNN+LSTM architectures
│   ├── portfolio/                 # Portfolio management
│   ├── risk/                      # Risk management & VaR
│   ├── execution/                 # Order execution engine
│   ├── monitoring/                # Performance monitoring
│   └── utils/                     # Shared utilities
├── configs/                       # Configuration management
├── tests/                         # Comprehensive test suite
├── docs/                          # Documentation
├── scripts/                       # Utility scripts
└── k8s/                          # Kubernetes deployment
```

---

## 🧠 **Core Components**

### **Hybrid CNN+LSTM + RL Architecture**

```python
# Market Intelligence Layer
cnn_lstm = CNNLSTMModel(
    cnn_filters=[32, 64, 128],
    lstm_units=256,
    dropout_rate=0.2,
    uncertainty_estimation=True
)

# RL Decision Layer
agent = SACAgent(
    state_space=env.observation_space,
    action_space=env.action_space,
    cnn_lstm_features=True  # Enhanced state representation
)

# Risk Management Integration
risk_manager = RiskManager(
    max_position_size=0.1,
    var_confidence_level=0.05,
    kelly_position_sizing=True
)
```

### **Advanced Feature Engineering**

 - **📊 Technical Indicators**: 150+ technical indicators powered by pandas-ta
- **🔗 Cross-Asset Features**: Correlation and regime detection
- **📰 Alternative Data**: News sentiment, economic indicators
- **🕒 Real-Time Processing**: Sub-second feature calculation

### **Production Risk Management**

- **📉 Value at Risk (VaR)**: Monte Carlo and historical simulation
- **📊 Portfolio Optimization**: Modern Portfolio Theory
- **💰 Position Sizing**: Kelly criterion with safety constraints
- **⚠️ Real-Time Monitoring**: Automated alerts and circuit breakers

---

## ⚙️ **Usage Examples**

### **Training a Multi-Asset Agent**

```python
from trading_rl_agent import ConfigManager
from trading_rl_agent.envs import PortfolioEnv
from trading_rl_agent.agents import EnsembleAgent

# Setup multi-asset environment
config = ConfigManager()
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA"]
env = PortfolioEnv(symbols=symbols, initial_balance=100000)

# Train ensemble agent
agent = EnsembleAgent(
    agents=["sac", "td3", "ppo"],
    ensemble_method="weighted"
)
agent.train(env, total_timesteps=1000000)

# Evaluate performance
results = agent.evaluate(env, episodes=100)
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### **Real-Time Risk Monitoring**

```python
from trading_rl_agent.risk import RiskManager
from trading_rl_agent.monitoring import MetricsCollector

# Initialize risk monitoring
risk_manager = RiskManager()
metrics = MetricsCollector()

# Monitor portfolio in real-time
while True:
    portfolio_weights = portfolio.get_weights()
    risk_metrics = risk_manager.calculate_metrics(portfolio_weights)

    # Check risk limits
    violations = risk_manager.check_limits(risk_metrics)
    if violations:
        print(f"⚠️ Risk violations detected: {violations}")

    # Log metrics
    metrics.log_metrics(risk_metrics)

    time.sleep(60)  # Check every minute
```

### **Backtesting with Transaction Costs**

```python
from trading_rl_agent.backtesting import Backtester

# Setup realistic backtesting
backtester = Backtester(
    start_date="2020-01-01",
    end_date="2024-01-01",
    initial_balance=100000,
    commission_rate=0.001,
    slippage_model="linear"
)

# Run backtest
results = backtester.run(
    agent=agent,
    symbols=["AAPL", "GOOGL", "MSFT"],
    rebalance_frequency="daily"
)

# Analyze results
print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
print(f"Win Rate: {results.win_rate:.2%}")
```

---

## 🧪 **Testing & Validation**

Before running the test suite, install the dependencies listed in
`requirements-test.txt`. These include **numpy**, **pandas**, **scipy**,
**torch**, **ray[rllib]**, **gymnasium**, **stable-baselines3**, **sb3-contrib**
and the full **pytest** toolchain.

```bash
pip install -r requirements-test.txt
```

### **Comprehensive Test Suite**

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests
pytest -m performance   # Performance tests

# Generate coverage report
pytest --cov=src tests/ --cov-report=html
```

### **Verify Full Workflow**

Run the minimal training example and evaluate the resulting checkpoint to
confirm that data ingestion, model training and evaluation work end-to-end.

```bash
# Train on the bundled sample dataset (or generated synthetic data)
python scripts/train_sample.py

# Evaluate the trained agent
python evaluate_agent.py \
    --data outputs/sample_data.csv \
    --checkpoint outputs/ppo_agent_checkpoint.zip \
    --agent ppo \
    --output outputs/evaluation.json
```

### **Performance Benchmarking**

```bash
# Benchmark against baselines
python scripts/benchmark.py --agent sac --baseline buy_and_hold

# Stress testing
python scripts/stress_test.py --scenarios crisis,volatility,trending

# Model validation
python scripts/validate_models.py --walk-forward --out-of-sample
```

---

## 🐳 **Production Deployment**

### **Docker Deployment**

```bash
# Build production image
docker build -f docker/Dockerfile.prod -t trading-rl-agent:latest .

# Run with docker-compose
docker-compose up -d

# Check services
docker-compose ps
```

### **Kubernetes Deployment**

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment
kubectl get pods -n trading-rl
kubectl logs -f deployment/trading-rl-agent -n trading-rl
```

### **Environment Variables**

```bash
# Required environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # Paper trading

# Optional
export REDIS_URL="redis://localhost:6379"
export POSTGRES_URL="postgresql://user:pass@localhost/trading"
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

---

## 📊 **Performance Monitoring**

### **Real-Time Dashboards**

- **📈 Trading Performance**: P&L, Sharpe ratio, drawdown tracking
- **⚠️ Risk Metrics**: VaR, position sizes, correlation analysis
- **🔧 System Health**: Latency, memory usage, error rates
- **📡 Market Data**: Real-time prices, volume, volatility

### **MLflow Integration**

```python
import mlflow

# Track experiments
with mlflow.start_run():
    mlflow.log_params(config.dict())
    mlflow.log_metrics({"sharpe_ratio": 1.45, "max_drawdown": 0.08})
    mlflow.log_model(agent, "model")
```

---

## 🛡️ **Risk Management Features**

### **Position Sizing**

- **Kelly Criterion**: Optimal position sizing based on historical performance
- **Risk Budgeting**: Allocation based on risk contribution
- **Volatility Targeting**: Dynamic position sizing based on market volatility

### **Portfolio Risk**

- **Value at Risk (VaR)**: 1-day and 10-day VaR calculations
- **Expected Shortfall (CVaR)**: Tail risk measurement
- **Maximum Drawdown**: Real-time drawdown monitoring
- **Correlation Analysis**: Cross-asset correlation tracking

### **Execution Risk**

- **Slippage Control**: Market impact minimization
- **Order Timeout**: Automatic order cancellation
- **Partial Fill Handling**: Smart order management
- **Circuit Breakers**: Automatic trading halts

---

## 🔍 **Current Status**

- ✅ **733 comprehensive tests** covering all components
- ✅ **Production-ready architecture** with enterprise patterns
- ✅ **Multi-asset portfolio support** with risk management
- ✅ **Real-time execution engine** with sub-second latency
- ✅ **Comprehensive monitoring** with alerts and dashboards
- ✅ **Docker & Kubernetes** deployment ready

### **Recent Updates**

- 🆕 **Restructured Architecture**: Production-grade modular design
- 🆕 **Enhanced Risk Management**: VaR, CVaR, Kelly criterion
- 🆕 **Real-Time Features**: Live data feeds and execution
- 🆕 **Multi-Asset Support**: Portfolio optimization and rebalancing

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
./setup-production.sh development

# Run quality checks
black src/ tests/
isort src/ tests/
flake8 src/ tests/
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

---

**🚀 Ready to revolutionize algorithmic trading with production-grade RL systems!**

---

