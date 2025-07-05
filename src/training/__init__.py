"""Training utilities and experiment orchestration.

This package contains the training implementations for both RL agents and
supervised CNN-LSTM models. Use :mod:`src.training.cli` as the command line
interface to launch training or Ray Tune sweeps.
"""

from .cnn_lstm import (
    CNNLSTMTrainer,
    TrainingConfig,
    create_example_config,
)
from . import rl

__all__ = [
    "CNNLSTMTrainer",
    "TrainingConfig",
    "create_example_config",
    "rl",
]
