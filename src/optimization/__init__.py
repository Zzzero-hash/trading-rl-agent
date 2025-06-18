"""Streamlined Optimization and Model Analysis Utilities.

This module provides clean, working implementations of:
1. CNN-LSTM hyperparameter optimization
2. RL agent hyperparameter optimization  
3. Model analysis and profiling utilities

Example usage:
>>> from src.optimization import optimize_cnn_lstm, ModelSummarizer
>>> results = optimize_cnn_lstm(features, targets, num_samples=20)
>>> summarizer = ModelSummarizer(model)
>>> print(summarizer.get_summary())
"""

from .cnn_lstm_optimization import optimize_cnn_lstm
from .model_summary import (
    ModelSummarizer,
    profile_model_inference,
    detect_gpus,
    optimal_gpu_config
)

# Import RL optimization if available
try:
    from .rl_optimization import optimize_sac_hyperparams
    __all__ = [
        "optimize_cnn_lstm",
        "optimize_sac_hyperparams", 
        "ModelSummarizer",
        "profile_model_inference",
        "detect_gpus",
        "optimal_gpu_config"
    ]
except ImportError:
    __all__ = [
        "optimize_cnn_lstm",
        "ModelSummarizer", 
        "profile_model_inference",
        "detect_gpus",
        "optimal_gpu_config"
    ]
