"""
Training module for trading RL agent.

This module contains training utilities, optimizers, and trainers for
machine learning models used in trading strategies.
"""

from .optimized_trainer import (
    AdvancedLRScheduler,
    MixedPrecisionTrainer,
    OptimizedTrainingManager,
)

# Import enhanced training components with error handling
try:
    from .train_cnn_lstm_enhanced import EnhancedCNNLSTMTrainer
except ImportError:
    EnhancedCNNLSTMTrainer = None  # type: ignore[misc,assignment]

from .model_registry import (
    ModelMetadata,
    ModelRegistry,
    PerformanceGrade,
)
from .preprocessor_manager import (
    PreprocessingPipeline,
    PreprocessorManager,
    PreprocessorMetadata,
)
from .unified_manager import (
    TrainingConfig,
    TrainingResult,
    UnifiedTrainingManager,
)

__all__ = [
    "AdvancedLRScheduler",
    "EnhancedCNNLSTMTrainer",
    "MixedPrecisionTrainer",
    "ModelMetadata",
    "ModelRegistry",
    "OptimizedTrainingManager",
    "PerformanceGrade",
    "PreprocessingPipeline",
    "PreprocessorManager",
    "PreprocessorMetadata",
    "TrainingConfig",
    "TrainingResult",
    "UnifiedTrainingManager",
]
