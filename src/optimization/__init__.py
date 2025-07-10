"""Streamlined Optimization and Model Analysis Utilities.

This module provides clean, working implementations of:
1. CNN-LSTM hyperparameter optimization
2. RL agent hyperparameter optimization
3. Model inspection and profiling utilities

Example usage:
>>> from trading_rl_agent.optimization import optimize_cnn_lstm, get_model_summary
>>> results = optimize_cnn_lstm(features, targets, num_samples=20)
>>> summary = get_model_summary(model, input_size=(1, 10))
>>> print(summary)
"""

from .cnn_lstm_optimization import optimize_cnn_lstm
from .model_utils import (
    detect_gpus,
    get_model_summary,
    optimal_gpu_config,
    profile_model_inference,
    run_hyperparameter_optimization,
)

# Import RL optimization if available
try:
    from .rl_optimization import optimize_sac_hyperparams

    __all__ = [
        "detect_gpus",
        "get_model_summary",
        "optimal_gpu_config",
        "optimize_cnn_lstm",
        "optimize_sac_hyperparams",
        "profile_model_inference",
        "run_hyperparameter_optimization",
    ]
except ImportError:
    __all__ = [
        "detect_gpus",
        "get_model_summary",
        "optimal_gpu_config",
        "optimize_cnn_lstm",
        "profile_model_inference",
        "run_hyperparameter_optimization",
    ]
