# Trading RL Agent Documentation

Production-ready hybrid CNN+LSTM + Reinforcement Learning system for algorithmic trading.

```{toctree}
:maxdepth: 2
:caption: Getting Started:

getting_started
ARCHITECTURE_OVERVIEW
```

```{toctree}
:maxdepth: 2
:caption: Development:

DEVELOPMENT_GUIDE
EVALUATION_GUIDE
ADVANCED_DATASET_DOCUMENTATION
examples
```

```{toctree}
:maxdepth: 2
:caption: Migration & Setup:

RAY_RLLIB_MIGRATION
PRE_COMMIT_SETUP
```

```{toctree}
:maxdepth: 2
:caption: API Reference:

api_reference
```

## Quick Start

```python
from src.envs.trading_env import TradingEnv
from src.agents.sac_agent import SACAgent

# Initialize hybrid environment with production dataset
env = TradingEnv(
    data_paths=['data/advanced_trading_dataset_*.csv'],
    use_cnn_lstm_features=True,
    window_size=50
)

# Create and train SAC agent
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)
agent.train(env, episodes=1000)
```

## System Status

- ✅ **Test suite with 733 cases** covering core functionality
- ✅ **Sample dataset included**; large historical datasets are optional
- ✅ **Hybrid CNN+LSTM + SAC** architecture production-ready
- ✅ **Ray RLlib 2.38.0+** compatibility (SAC with **FinRL** integration; custom TD3 retained for experimentation)
- ✅ **Automated quality checks** integrated
  _Metrics are illustrative and rely on sample data._
