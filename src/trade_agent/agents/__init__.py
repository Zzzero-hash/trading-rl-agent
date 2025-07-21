"""Agents package - Contains reinforcement learning agents and training utilities."""

# Import configs
# Import agents and trainer
# Import advanced policy optimization components
from .advanced_policy_optimization import (
    TRPO,
    AdaptiveLearningRateScheduler,
    AdvancedPPO,
    MultiObjectiveOptimizer,
    NaturalPolicyGradient,
    PolicyOptimizationComparison,
)
from .advanced_trainer import AdvancedTrainer, MultiObjectiveTrainer
from .benchmark_framework import (
    BenchmarkConfig,
    BenchmarkFramework,
    run_quick_benchmark,
)
from .configs import (
    AdvancedPPOConfig,
    EnsembleConfig,
    MultiObjectiveConfig,
    NaturalPolicyGradientConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    TRPOConfig,
)
from .ensemble_evaluator import EnsembleEvaluator

# Import new ensemble components
from .ensemble_trainer import EnsembleTrainer
from .policy_utils import (
    CallablePolicy,
    EnsembleAgent,
    WeightedEnsembleAgent,
    weighted_policy_mapping,
)
from .trainer import Trainer

__all__ = [
    "TRPO",
    "AdaptiveLearningRateScheduler",
    "AdvancedPPO",
    "AdvancedPPOConfig",
    "AdvancedTrainer",
    "BenchmarkConfig",
    "BenchmarkFramework",
    "CallablePolicy",
    "EnsembleAgent",
    "EnsembleConfig",
    "EnsembleEvaluator",
    "EnsembleTrainer",
    "MultiObjectiveConfig",
    "MultiObjectiveOptimizer",
    "MultiObjectiveTrainer",
    "NaturalPolicyGradient",
    "NaturalPolicyGradientConfig",
    "PPOConfig",
    "PolicyOptimizationComparison",
    "SACConfig",
    "TD3Config",
    "TRPOConfig",
    "Trainer",
    "WeightedEnsembleAgent",
    "run_quick_benchmark",
    "weighted_policy_mapping",
]
