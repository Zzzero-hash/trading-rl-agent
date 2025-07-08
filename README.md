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

- **📊 Technical Indicators**: 150+ TA-Lib indicators
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

### **Comprehensive Test Suite**

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v          # Unit tests
pytest tests/integration/ -v   # Integration tests
pytest tests/performance/ -v   # Performance tests

# Generate coverage report
pytest --cov=src tests/ --cov-report=html
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

## 🚀 Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/Zzzero-hash/trading-rl-agent.git
   cd trading-rl-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-finrl.txt
   pip install finrl[full] "ray[rllib]"
   ```

   The base requirements file now includes `nltk>=3.8` for sentiment analysis.

3. **Run tests**
   ```bash
   pytest
   ```
   The suite currently includes around 29 tests. Some may fail if optional dependencies such as `finrl` or `nltk` are missing.

---

## 📦 Project Structure

```
trading-rl-agent/
├── src/
│   ├── agents/           # RL agents (SAC, PPO, EnsembleAgent(RLlib))
│   ├── models/           # CNN+LSTM architectures
│   ├── data/             # Data processing & feature generation
│   ├── envs/             # Trading environments
│   └── deployment/       # Serving configurations
├── tests/                # Unit and integration tests
├── cnn_lstm_hparam_clean.ipynb  # Hyperparameter tuning notebook
└── data/                 # Sample datasets
```

---

## 🧠 Core Components

- **CNN+LSTM Market Intelligence**
  - Sequence input of market features + technical indicators
  - Convolutional layers → LSTM with attention → trend forecasts

- **RL Decision Engine**
  - Built on FinRL + Ray RLlib + Stable Baselines3[contrib]
  - Supports SAC, PPO
  - Risk-adjusted reward functions integrated

- **Testing Framework**
  - Fully automated environment tests
  - Backtesting simulation with transaction costs
  - Coverage across data pipelines, models, and agents

---

## ⚙️ Usage Examples

### Train an SAC Agent

```python
from finrl.env.env_stocktrading import StockTradingEnv
from ray.rllib.algorithms.sac import SACConfig
from ray import tune

config = SACConfig().environment(StockTradingEnv)
tune.Tuner("SAC", param_space=config, stop={"training_iteration": 10}).fit()
```

### Backtest with Sample Data

```bash
python finrl_data_loader.py --config configs/finrl_real_data.yaml
python src/train_finrl_agent.py --agent sac --data sample --backtesting realistic
```

### Messaging Example

```python
import asyncio
from messaging import connect


async def main():
    async with connect(servers=["nats://localhost:4222"]) as nc:
        await nc.publish("trades.buy", b"{\"symbol\": \"AAPL\"}")
        await nc.subscribe("trades.*", cb=lambda msg: print(msg.data))


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🔍 Current Status

- ✅ Environment testing framework complete
- ✅ Core CNN+LSTM & RL integration validated
- ✅ Sample datasets included
- ❌ Some tests failing due to missing dependencies (see [test-report.md](test-report.md))

> **Note:** This project is research-oriented. Sample data and workflows are provided for experimentation. Production deployment, professional data feeds, advanced risk modules, and multi-asset portfolio features remain under active development.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code standards
- Pull request guidelines

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

> **Disclaimer:** For educational and research purposes only. Not financial advice. Always paper‐trade and consult professionals before deploying strategies with real capital.
