# Streamlined Optimization Structure

## Core Files
- `src/optimization/cnn_lstm_optimization.py` - Main CNN-LSTM hyperparameter optimization
- `src/optimization/rl_optimization.py` - RL agent hyperparameter optimization  
- `src/optimization/model_summary.py` - Model analysis and profiling utilities

## Usage
```python
# CNN-LSTM optimization
from src.optimization.cnn_lstm_optimization import optimize_cnn_lstm
results = optimize_cnn_lstm(features, targets, num_samples=20)

# RL optimization  
from src.optimization.rl_optimization import optimize_sac_hyperparams
results = optimize_sac_hyperparams(env_config, num_samples=10)

# Model analysis
from src.optimization.model_summary import ModelSummarizer
summarizer = ModelSummarizer(model)
print(summarizer.get_summary())
```

## Configuration
- Hyperparameter spaces defined in optimization modules
- Ray Tune integration for distributed optimization
- Automatic GPU detection and configuration
