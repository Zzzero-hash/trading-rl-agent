# Getting Started with Trading RL Agent

Welcome to the Trading RL Agent - a production-ready hybrid CNN+LSTM + Reinforcement Learning system for algorithmic trading.

## 🏗️ System Overview

This is a **two-tier hybrid system** that combines:

- **Tier 1**: Deep Learning (CNN+LSTM) for market pattern recognition and feature extraction
- **Tier 2**: Reinforcement Learning (SAC/TD3/Ensemble) for trading decision optimization

**Current Status**: Core modules validated with ~733 tests. Sample data is provided and large datasets are optional. Automated quality checks help maintain the codebase.

## 🚀 Quick Setup

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (optional, for GPU training)

### Installation

```bash
# Clone and setup
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Setup environment
./setup-env.sh full

# Run tests to verify
pytest
```

## Quick Start Tutorial

### 1. Generate Production Dataset

```bash
# Generate the full 1.37M record production dataset
python build_advanced_dataset.py

# This creates:
# - Advanced feature engineering (78 technical indicators)
# - 19 real market instruments (stocks, forex, crypto)
# - Synthetic data augmentation
# - Real-time integration compatibility
```

### 2. Initialize the Hybrid System

```python
from src.envs.trading_env import TradingEnv
from src.agents.sac_agent import SACAgent
from src.models.cnn_lstm_model import CNNLSTMModel

# Load production dataset
env = TradingEnv(
    data_paths=["data/advanced_trading_dataset_*.csv"],
    initial_balance=10000,
    window_size=50,
    transaction_cost=0.001,
    use_cnn_lstm_features=True  # Enable hybrid mode
)

# Initialize hybrid CNN+LSTM feature extractor
cnn_lstm = CNNLSTMModel(
    input_features=78,
    cnn_filters=[32, 64, 128],
    lstm_units=256,
    dropout_rate=0.2
)

# Initialize RL agent with hybrid features
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dim=512,  # Larger network for hybrid features
    learning_rate=3e-4
)
```

### 3. Train the Hybrid System

```python
# Train with hybrid architecture
training_results = agent.train(
    env=env,
    episodes=2000,
    max_steps=1000,
    save_frequency=100,
    use_ray_tune=True,  # Hyperparameter optimization
    hybrid_mode=True    # Enable CNN+LSTM integration
)
```

### 4. Evaluate Hybrid Performance

```python
from evaluate_agent import evaluate_agent

# Evaluate the hybrid system
results = evaluate_agent(
    agent=agent,
    env=env,
    episodes=10,
    use_hybrid_features=True
)

print(f"Hybrid System Performance:")
print(f"Average Return: {results['avg_return']:.2f}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2f}")
print(f"Win Rate: {results['win_rate']:.2f}")
```

## ⚙️ Configuration

The hybrid system uses environment-specific configurations:

### Production Environment Configuration

```yaml
# config/production_env_config.yaml
environment:
  initial_balance: 100000
  window_size: 50
  transaction_cost: 0.001
  max_position: 1.0
  normalize_observations: true
  use_cnn_lstm_features: true
  feature_engineering_level: "advanced"

hybrid_model:
  cnn_filters: [32, 64, 128]
  lstm_units: 256
  dropout_rate: 0.2
  batch_normalization: true
```

### Hybrid Agent Configuration

```yaml
# config/hybrid_sac_config.yaml
agent:
  algorithm: "SAC"
  hidden_dim: 512
  learning_rate: 0.0003
  batch_size: 256
  memory_size: 1000000
  tau: 0.005
  gamma: 0.99
  alpha: 0.2

training:
  episodes: 2000
  max_steps: 1000
  save_frequency: 100
  use_ray_tune: true
  hyperparameter_optimization: true
```

## 🧪 Testing & Validation

```bash
# Run comprehensive test suite
pytest tests/ -v

# Run specific test categories
pytest tests/test_agents/ -v  # RL agent tests
pytest tests/test_models/ -v  # CNN+LSTM model tests
pytest tests/test_envs/ -v    # Environment tests
pytest tests/test_integration/ -v  # Hybrid system integration

# Generate test coverage report
pytest --cov=src tests/
```

## 📊 Performance Monitoring

```bash
# Monitor training progress
tensorboard --logdir=logs/

# Evaluate model performance
python evaluate_agent.py --config config/production_config.yaml

# Generate performance reports
python scripts/generate_performance_report.py
```

## 🔄 Next Steps

### Production Deployment

1. Review [Architecture Overview](ARCHITECTURE_OVERVIEW.md) for system design details
2. Read [Advanced Dataset Documentation](ADVANCED_DATASET_DOCUMENTATION.md) for data pipeline management
3. Study [Experiment Outputs Management](EXPERIMENT_OUTPUTS_MANAGEMENT.md) for production workflows

### Development & Research

1. Explore [Notebook Best Practices](NOTEBOOK_BEST_PRACTICES.md) for ML development workflows
2. Check [API Reference](api_reference.md) for detailed class and method documentation
3. Review [Examples](examples.md) for advanced use cases and patterns

### Phase 3 Development (Multi-Asset Portfolio)

1. Multi-asset environment development
2. Portfolio optimization algorithms
3. Cross-asset correlation modeling
4. Risk management enhancements

## 🆘 Getting Help

- **Documentation**: Comprehensive guides in [`docs/`](.)
- **Examples**: Working code samples in [`examples/`](../examples/)
- **Tests**: Reference implementations in [`tests/`](../tests/)
- **Issues**: GitHub issue tracker for bug reports and feature requests

**Current Achievement**: ~733 tests defined with sample data. Larger datasets are optional. Automated quality checks ensure consistency. Metrics are illustrative ✨
