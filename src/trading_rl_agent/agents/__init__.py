"""Agents package - Contains reinforcement learning agents and training utilities."""

# Configs are imported separately by each agent to avoid circular dependencies

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
from .benchmark_framework import BenchmarkConfig, BenchmarkFramework, run_quick_benchmark
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
    # Advanced policy optimization
    "AdvancedPPO",
    "AdvancedTrainer",
    # Benchmarking
    "BenchmarkConfig",
    "BenchmarkFramework",
    # Core agents
    "CallablePolicy",
    "EnsembleAgent",
    "EnsembleEvaluator",
    "EnsembleTrainer",
    # Multi-objective optimization
    "MultiObjectiveOptimizer",
    "MultiObjectiveTrainer",
    "NaturalPolicyGradient",
    # Policy optimization utilities
    "PolicyOptimizationComparison",
    "Trainer",
    "WeightedEnsembleAgent",
    # Benchmarking utilities
    "run_quick_benchmark",
    "weighted_policy_mapping",
]
