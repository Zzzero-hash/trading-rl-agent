# RL Agents with CNN+LSTM Model

This directory contains implementations of reinforcement learning agents that integrate with the CNN+LSTM model for processing financial time series data.

## Overview

The implementations in this directory provide two different approaches to integrating the CNN+LSTM model with reinforcement learning algorithms:

1. **PPO Agent** - Implementation using Stable-Baselines3
2. **SAC Agent** - Implementation using Ray RLlib

Both implementations are designed to work with the same CNN+LSTM model architecture, allowing for consistent feature extraction across different RL algorithms.

## Agent Implementations

### PPO Agent (`ppo_agent.py`)

The PPO (Proximal Policy Optimization) agent implementation uses Stable-Baselines3 as the RL framework. It includes:

- `CNNLSTMFeaturesExtractor`: Feature extractor that wraps the CNN+LSTM model
- `CNNLSTMPolicy`: Custom policy using the CNN+LSTM feature extractor
- `PPOAgent`: High-level interface for creating and using PPO agents

[Detailed PPO Agent Documentation](../../docs/agents/ppo_agent.md)

### SAC Agent (`sac_agent.py`)

The SAC (Soft Actor-Critic) agent implementation uses Ray RLlib as the RL framework. It includes:

- `CNNLSTMFeatureExtractor`: Feature extractor that wraps the CNN+LSTM model
- `CNNLSTMSACModel`: Custom SAC model using the CNN+LSTM feature extractor
- `SACAgent`: High-level interface for creating and using SAC agents

[Detailed SAC Agent Documentation](../../docs/agents/sac_agent.md)

## Common CNN+LSTM Integration

Both agents integrate with the same CNN+LSTM model architecture:

- **Input**: Multi-asset financial time series data with shape `(batch_size, time_steps, assets, features)`
- **CNN Processing**: Spatial feature extraction from cross-sectional market data
- **LSTM Processing**: Temporal sequence modeling across time steps
- **Output**: Fixed-size feature vector for policy and value networks

This shared architecture ensures consistency in feature extraction regardless of the chosen RL algorithm.

## Usage Recommendations

Based on research recommendations:

1. **Primary**: Use the SAC agent for most trading applications due to its sample efficiency and stability
2. **Secondary**: Use the PPO agent as a comparison baseline or when working with simpler environments

## Configuration

Both agents support extensive configuration options for the CNN+LSTM model and their respective RL algorithms. See the detailed documentation for each agent for specific configuration options.

## Examples

See the detailed documentation for each agent for usage examples:

- [PPO Agent Examples](../../docs/agents/ppo_agent.md#usage-example)
- [SAC Agent Examples](../../docs/agents/sac_agent.md#usage-example)
