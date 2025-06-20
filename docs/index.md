# Trading RL Agent Documentation

Welcome to the Trading RL Agent documentation! This project implements a comprehensive reinforcement learning system for algorithmic trading.

```{toctree}
:maxdepth: 2
:caption: Contents:

getting_started
architecture
api_reference
examples
contributing
```

## Quick Start

```python
from src.envs.trading_env import TradingEnv
from src.agents.sac_agent import SACAgent

# Initialize environment
env = TradingEnv(data_paths=['data/sample_data.csv'])

# Create agent
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

# Train the agent
agent.train(env, episodes=1000)
```

## Features

- **Multiple RL Algorithms**: SAC, TD3, and Ensemble methods
- **Advanced Market Data**: Real-time and historical data integration
- **Feature Engineering**: Technical indicators and sentiment analysis
- **Comprehensive Testing**: >92% code coverage with extensive test suite
- **Production Ready**: Docker containers and Kubernetes deployment
- **Hyperparameter Optimization**: Ray Tune integration

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
