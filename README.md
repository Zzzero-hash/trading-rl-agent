# Trading RL Agent

[![Coverage Status](https://codecov.io/gh/Zzzero-hash/trading-rl-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/Zzzero-hash/trading-rl-agent)

A research-focused system combining CNN+LSTM market intelligence with reinforcement learning (RL) optimization.
_Current status: core functionality validated, environment testing framework complete, and all tests passing._

---

## 🚀 Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/Zzzero-hash/trading-rl-agent.git
   cd trading-rl-agent
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements-finrl.txt
   pip install finrl[full] "ray[rllib]"
   ```

3. **Run tests**
   ```bash
   pytest
   ```
   All ~733 tests should pass, validating the core environment and integration.

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
├── build_production_dataset.py  # Dataset generation script
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
python build_production_dataset.py --symbols AAPL,MSFT --start 2024-01-01
python src/train_finrl_agent.py --agent sac --data sample --backtesting realistic
```

---

## 🔍 Current Status

- ✅ Environment testing framework complete
- ✅ Core CNN+LSTM & RL integration validated
- ✅ Sample datasets included
- ✅ ~733 tests passing

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
