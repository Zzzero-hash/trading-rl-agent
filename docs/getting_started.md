# Getting Started with Trading RL Agent

Welcome to the Trading RL Agent, a production-ready system for algorithmic trading that combines deep learning and reinforcement learning.

## 🏗️ System Overview

This project provides a robust framework for developing and testing trading strategies. It features:

- **A flexible system configuration**: Easily configure all aspects of the system, from data processing to agent hyperparameters.
- **A powerful `Trainer` class**: The main entry point for training and evaluating agents.
- **Support for multiple RL agents**: Includes support for SAC, PPO, and TD3.
- **Comprehensive feature engineering**: Generate a wide range of technical indicators and other features.

## 🚀 Quick Setup

### Prerequisites

- Python 3.10+
- An environment manager like `conda` or `venv`

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-repo/trading-rl-agent.git
    cd trading-rl-agent
    ```

2.  **Create and activate a virtual environment**:

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the tests to verify the installation**:
    ```bash
    pytest
    ```

## ⚙️ Basic Usage

The primary way to use this project is through the `Trainer` class. Here's a basic example:

```python
from trading_rl_agent.agents.trainer import Trainer
from trading_rl_agent.core.config import SystemConfig

# 1. Create a SystemConfig object
system_config = SystemConfig()

# 2. Initialize the Trainer
trainer = Trainer(system_cfg=system_config, save_dir="outputs/basic_example")

# 3. Run the training process
trainer.train()

print("Training complete! Check the 'outputs/basic_example' directory for results.")
```

## 🧪 Testing & Validation

To ensure the system is working correctly, run the comprehensive test suite:

```bash
pytest -vv
```

## Next Steps

- **Explore the code**: The main logic is in the `src/trading_rl_agent` directory.
- **Customize the configuration**: Modify the `SystemConfig` object to experiment with different settings.
- **Review the documentation**: Check out the other guides in this `docs` directory for more information.

---

For legal and safety notes see the [project disclaimer](disclaimer.md).
