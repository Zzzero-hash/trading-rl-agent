# Hybrid CNN+LSTM Architecture for Trading RL Agent

## Overview

This document describes the design of a hybrid CNN+LSTM neural network architecture for processing financial time series data in a reinforcement learning trading agent. The architecture combines the spatial feature extraction capabilities of Convolutional Neural Networks (CNNs) with the temporal sequence modeling of Long Short-Term Memory (LSTM) networks to effectively capture both cross-sectional and temporal patterns in market data.

## Input Specifications

Based on the requirements, the model will process:

- **Time Window**: 30 time steps of daily bars (approximately 1 month of data)
- **Assets**: 100+ assets simultaneously
- **Features**: OHLCV data with common technical indicators (RSI, MACD, Bollinger Bands, etc.)

### Input Tensor Structure

```
Input Shape: (batch_size, time_steps, assets, features)
Where:
- batch_size: Variable batch size for training/inference
- time_steps: 30 (daily bars)
- assets: 100+ assets
- features: OHLCV + technical indicators (approximately 10-15 features)
```

## Architecture Design

### 1. CNN Component

The CNN component is designed to extract spatial features from the cross-sectional market data at each time step.

#### Structure:

1. **Input Reshaping Layer**
   - Reshapes input from (batch_size, time_steps, assets, features) to (batch_size \* time_steps, assets, features)
   - Allows processing of each time step independently

2. **Convolutional Layers**
   - **Conv1D Layer 1**: 64 filters, kernel size 3, ReLU activation
   - **Conv1D Layer 2**: 32 filters, kernel size 3, ReLU activation
   - **Conv1D Layer 3**: 16 filters, kernel size 3, ReLU activation
   - Purpose: Extract local patterns and relationships between assets

3. **Pooling Layer**
   - Global Average Pooling to reduce dimensionality
   - Output shape: (batch_size \* time_steps, 16)

4. **Reshape Layer**
   - Reshapes output back to (batch_size, time_steps, 16)
   - Prepares data for temporal processing by LSTM

#### Justification:

- 1D convolutions are appropriate for time series data where we want to capture relationships between assets at each time step
- Multiple convolutional layers with decreasing filter counts follow common CNN design patterns
- Global Average Pooling reduces overfitting compared to fully connected layers while maintaining spatial information

### 2. LSTM Component

The LSTM component captures temporal dependencies across the 30-day time window.

#### Structure:

1. **LSTM Layer 1**: 128 units, return sequences=True
2. **Dropout Layer**: 0.2 dropout rate for regularization
3. **LSTM Layer 2**: 64 units, return sequences=False
4. **Dropout Layer**: 0.2 dropout rate for regularization

#### Justification:

- Stacked LSTM layers can learn complex temporal patterns
- Dropout layers prevent overfitting during training
- The final LSTM layer outputs a fixed-size vector representing the temporal state

### 3. Integration Architecture

The CNN and LSTM components are integrated sequentially:

```
Input → CNN Feature Extractor → Temporal Sequence → LSTM Processor → Latent Representation
```

#### Data Flow:

1. Input data is processed by the CNN component at each time step
2. The CNN outputs a feature vector for each time step
3. These feature vectors form a temporal sequence for the LSTM
4. The LSTM processes this sequence to produce a final latent representation

### 4. Output Layer

The output layer is designed to be compatible with multiple RL algorithms (PPO, SAC, DDPG, TD3).

#### Structure:

1. **Dense Layer 1**: 128 units, ReLU activation
2. **Dense Layer 2**: 64 units, ReLU activation
3. **Output Layer**:
   - For continuous action spaces: Linear activation with 10-20 dimensions for portfolio allocation
   - For value functions: Single unit with linear activation

#### Flexibility Features:

- Modular design allows for different output heads for policy and value functions
- Configurable output dimensions to support different portfolio sizes
- Shared latent representation reduces computational overhead

## Architecture Diagram

```mermaid
graph TD
    A[Input: (batch, 30, 100+, features)] --> B[CNN Feature Extractor]
    B --> C[Temporal Sequence: (batch, 30, 16)]
    C --> D[LSTM Processor]
    D --> E[Latent Representation: (batch, 64)]
    E --> F[Output Layers]
    F --> G[Policy Output: (batch, 10-20)]
    F --> H[Value Output: (batch, 1)]
```

## Industry Standards and Best Practices

### Financial Time Series Modeling

- **Feature Engineering**: The architecture assumes pre-computed technical indicators, following industry practice where feature engineering is often done as a preprocessing step
- **Multi-Asset Processing**: Processing multiple assets simultaneously allows the model to learn cross-asset relationships, which is crucial for portfolio optimization
- **Temporal Horizon**: 30-day windows balance short-term trading signals with longer-term market trends

### Deep Learning Design

- **CNN for Spatial Features**: Using CNNs for cross-sectional feature extraction is well-established in financial literature
- **LSTM for Temporal Dependencies**: LSTMs are proven effective for financial time series modeling due to their ability to capture long-term dependencies
- **Regularization**: Dropout layers prevent overfitting, which is critical in financial applications where models can easily overfit to historical patterns

### Reinforcement Learning Integration

- **Modular Output**: Separate policy and value outputs support both on-policy (PPO) and off-policy (SAC, DDPG, TD3) algorithms
- **Continuous Action Space**: Direct output for continuous portfolio weights aligns with modern portfolio optimization approaches
- **Latent Representation**: Shared latent space enables efficient learning across multiple RL objectives

## Computational Efficiency Considerations

### Real-Time Trading Requirements

- **Batch Processing**: Architecture supports variable batch sizes for both training and inference
- **Reduced Dimensionality**: CNN processing reduces the high-dimensional asset space to manageable feature vectors
- **Sequential Processing**: LSTM layers process temporal sequences efficiently

### Optimization Strategies

- **Model Pruning**: Convolutional layers can be pruned for inference to reduce computational load
- **Quantization**: Model can be quantized for deployment on resource-constrained environments
- **Caching**: Intermediate representations can be cached for repeated predictions with updated data

## Training Considerations

### Loss Functions

- **Policy Gradient Methods**: Compatible with PPO's clipped surrogate objective
- **Actor-Critic Methods**: Supports value function learning for SAC, DDPG, and TD3
- **Multi-Task Learning**: Can jointly optimize policy and value functions

### Regularization Techniques

- **Dropout**: Applied in LSTM layers to prevent overfitting
- **Batch Normalization**: Can be added to dense layers for stable training
- **Gradient Clipping**: Recommended for LSTM training to prevent gradient explosion

## Future Extensions

### Model Enhancements

- **Attention Mechanisms**: Self-attention could be added to capture long-range dependencies
- **Multi-Scale Processing**: Multiple CNN branches for different time granularities
- **External Features**: Integration of macroeconomic indicators or alternative data sources

### Ensemble Methods

- **Model Averaging**: Multiple CNN+LSTM models with different initializations
- **Snapshot Ensembles**: Multiple models saved at different training stages

## Conclusion

This hybrid CNN+LSTM architecture provides a robust foundation for a trading RL agent that can process multi-asset financial time series data. The design balances the need for spatial feature extraction with temporal pattern recognition while maintaining compatibility with multiple RL algorithms. The modular structure allows for future enhancements while the efficiency considerations make it suitable for real-time trading applications.

## RL Agent Integrations

This CNN+LSTM model has been integrated with two popular reinforcement learning algorithms:

1. **PPO Agent**: Implementation using Stable-Baselines3
   - [PPO Agent Documentation](./agents/ppo_agent.md)
   - [Source Code](../src/agents/ppo_agent.py)

2. **SAC Agent**: Implementation using Ray RLlib
   - [SAC Agent Documentation](./agents/sac_agent.md)
   - [Source Code](../src/agents/sac_agent.py)

Both implementations are designed to work with the same CNN+LSTM model architecture, allowing for consistent feature extraction across different RL algorithms. Based on research recommendations, the SAC agent is recommended as the primary choice with the PPO agent as a secondary option for comparison.
