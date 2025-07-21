"""
Evaluation Framework for Trading RL Agent

This module provides comprehensive evaluation capabilities including:
- Walk-forward analysis for robust model validation
- Performance metrics calculation
- Model comparison utilities
- Statistical significance testing
- Visualization and reporting tools
- Agent scenario evaluation with synthetic data
"""

from .metrics_calculator import MetricsCalculator
from .model_evaluator import ModelEvaluator
from .scenario_evaluator import AgentScenarioEvaluator, MarketScenario, ScenarioResult
from .statistical_tests import StatisticalTests
from .walk_forward_analyzer import WalkForwardAnalyzer

__all__ = [
    "AgentScenarioEvaluator",
    "MarketScenario",
    "MetricsCalculator",
    "ModelEvaluator",
    "ScenarioResult",
    "StatisticalTests",
    "WalkForwardAnalyzer",
]
