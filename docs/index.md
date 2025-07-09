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

```{toctree}
:maxdepth: 2
:caption: Additional Guides:

MIGRATION_GUIDE
ROADMAP
CLEANUP_PLAN
COMPREHENSIVE_TESTING_FRAMEWORK
```

## Quick Start

```python
from trading_rl_agent import ConfigManager, PortfolioManager
from trading_rl_agent.agents import SACAgent
from trading_rl_agent.data.pipeline import load_cached_csvs

cfg = ConfigManager("configs/production.yaml").load_config()
data = load_cached_csvs("data/raw")

agent = SACAgent(cfg.agent)
agent.train(data)

portfolio = PortfolioManager(initial_capital=100000)
portfolio.start_live_trading(agent)
```

## System Status

- ✅ **Test suite with 733 cases** covering core functionality
- ✅ **Sample dataset included**; large historical datasets are optional
- ✅ **Hybrid CNN+LSTM + SAC** architecture production-ready
- ✅ **Ray RLlib 2.38.0+** compatibility (SAC with **FinRL** integration; custom TD3 retained for experimentation)
- ✅ **Automated quality checks** integrated
  _Metrics are illustrative and rely on sample data._
