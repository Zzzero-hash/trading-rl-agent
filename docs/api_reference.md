# API Reference

This section provides detailed API documentation for all modules and classes.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   src.agents
   src.envs
   src.data
   src.utils
   src.optimization
```

## Agents

### SAC Agent

```{eval-rst}
.. automodule:: src.agents.sac_agent
   :members:
   :undoc-members:
   :show-inheritance:
```

### TD3 Agent

_Experimental: maintained only for custom research and testing_

```{eval-rst}
.. automodule:: src.agents.td3_agent
   :members:
   :undoc-members:
   :show-inheritance:
```

### Ensemble Utilities

```{eval-rst}
.. automodule:: src.agents.policy_utils
   :members:
   :undoc-members:
   :show-inheritance:
```

### Agent Configurations

```{eval-rst}
.. automodule:: src.agents.configs
   :members:
   :undoc-members:
   :show-inheritance:
```

## Environments

### Trading Environment

```{eval-rst}
.. automodule:: src.envs.trading_env
   :members:
   :undoc-members:
   :show-inheritance:
```

### Trader Environment

```{eval-rst}
.. automodule:: src.envs.trader_env
   :members:
   :undoc-members:
   :show-inheritance:
```

## Data Processing

### Features

```{eval-rst}
.. automodule:: src.data.features
   :members:
   :undoc-members:
   :show-inheritance:
```

### Historical Data

```{eval-rst}
.. automodule:: src.data.historical
   :members:
   :undoc-members:
   :show-inheritance:
```

### Pipeline

```{eval-rst}
.. automodule:: src.data.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
```

### Sentiment Analysis

```{eval-rst}
.. automodule:: src.data.sentiment
   :members:
   :undoc-members:
   :show-inheritance:
```

### Candlestick Patterns

```{eval-rst}
.. automodule:: src.data.candle_patterns
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utilities

### Metrics

```{eval-rst}
.. automodule:: src.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:
```

### Cluster Management

```{eval-rst}
.. automodule:: src.utils.cluster
   :members:
   :undoc-members:
   :show-inheritance:
```

### Quantization

This project relies on PyTorch's built-in dynamic quantization utilities.

```{eval-rst}
.. autofunction:: torch.quantization.quantize_dynamic
```

## NLP Utilities

```{eval-rst}
.. automodule:: src.nlp
   :members:
   :undoc-members:
   :show-inheritance:
```

## Optimization

### CNN-LSTM Optimization

```{eval-rst}
.. automodule:: src.optimization.cnn_lstm_optimization
   :members:
   :undoc-members:
   :show-inheritance:
```

### RL Optimization

```{eval-rst}
.. automodule:: src.optimization.rl_optimization
   :members:
   :undoc-members:
   :show-inheritance:
```

### Model Summary

```{eval-rst}
.. automodule:: src.optimization.model_utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## Training Modules

### CNN-LSTM Training

```{eval-rst}
.. automodule:: src.training.cnn_lstm
   :members:
   :undoc-members:
   :show-inheritance:
```

### Supervised Model

```{eval-rst}
.. automodule:: src.supervised_model
   :members:
   :undoc-members:
   :show-inheritance:
```

## Deployment

### Serve Deployment

```{eval-rst}
.. automodule:: src.serve_deployment
   :members:
   :undoc-members:
   :show-inheritance:
```
