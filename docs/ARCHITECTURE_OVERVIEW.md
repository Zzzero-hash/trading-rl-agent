# Architecture Overview

This document outlines the high level architecture of the trading reinforcement learning system. It focuses on how data is ingested, transformed into features, and fed into the custom `TradingEnv` used by the RL agents. The interactions between the Soft Actor-Critic (SAC), TD3, and ensemble components are shown in the diagram below.

The diagram source is stored in
[`img/architecture_pipeline.gv`](img/architecture_pipeline.gv). To view it as an
image, first generate the PNG or decode the provided base64 file:

```bash
# Option 1: render directly from Graphviz
dot -Tpng docs/img/architecture_pipeline.gv -o docs/img/architecture_pipeline.png

# Option 2: decode the base64 representation
base64 -d docs/img/architecture_pipeline.png.b64 > docs/img/architecture_pipeline.png
```

After running either command, you can open `img/architecture_pipeline.png` which
will produce the following diagram:

![Pipeline](img/architecture_pipeline.png)

## Data Ingestion

- Source files are prepared using `build_datasets.py` and related scripts.
- Raw market data is read, merged with sentiment data, and stored in CSV files under `data/`.

## Feature Engineering

- Features are generated in `src/data/features.py`.
- Common technical indicators (SMA, EMA, RSI, Bollinger Bands) and sentiment metrics are added.
- Feature generation is used both for supervised training and for the RL environment when `include_features` is enabled.

## Trading Environment

- Implementation: [`src/envs/trading_env.py`](../src/envs/trading_env.py).
- Wraps market data into a Gym-compatible environment.
- Supports discrete and continuous actions for compatibility with TD3.
- Optionally injects CNN‑LSTM predictions into the observation space.

## RL Agents

- **SAC Agent** – [`src/agents/sac_agent.py`](../src/agents/sac_agent.py)
  - Used for distributed training via Ray RLlib.
  - Provides stochastic policy with entropy regularization.
- **TD3 Agent** – [`src/agents/td3_agent.py`](../src/agents/td3_agent.py)
  - Custom implementation for local experiments.
  - Uses deterministic policies with twin critics and delayed actor updates.

## Ensemble Interaction

- **Ensemble Agent** – [`src/agents/ensemble_agent.py`](../src/agents/ensemble_agent.py)
  - Combines the outputs of SAC and TD3 agents.
  - Designed to reduce variance and improve stability.
  - Can be extended to include additional models.

The agents consume observations from `TradingEnv`, take actions, and optionally contribute to ensemble decisions. This modular setup makes it straightforward to swap out or update individual components while keeping the overall pipeline intact.
