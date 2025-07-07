import numpy as np
import pytest

from src.risk import RiskfolioConfig, RiskfolioRiskManager


@pytest.fixture
def risk_manager():
    cfg = RiskfolioConfig(max_position=0.5, min_position=-0.5, var_limit=0.005)
    return RiskfolioRiskManager(cfg)


def test_position_limit_violation(risk_manager):
    # Action beyond max_position should be invalid
    action = 1.0
    returns = np.array([0.001, -0.002, 0.003])
    assert not risk_manager.validate_action(action, returns)


def test_var_limit_violation(risk_manager):
    # Override calculate_risk to return high Var
    def high_var(returns):
        return {"var": 0.01}
    risk_manager.calculate_risk = high_var
    assert not risk_manager.validate_action(0.0, np.array([0.0]))


def test_valid_action_within_limits():
    # Default var is 0.01, var_limit set to 0.02 allows valid actions
    cfg = RiskfolioConfig(max_position=1.0, min_position=-1.0, var_limit=0.02)
    rm = RiskfolioRiskManager(cfg)
    assert rm.validate_action(0.0, np.array([0.0, 0.001]))


def test_risk_adjusted_action_halts():
    # Invalid action should be adjusted to zero (halt)
    cfg = RiskfolioConfig(max_position=0.5, min_position=-0.5, var_limit=0.005)
    rm = RiskfolioRiskManager(cfg)
    action = 1.0
    adjusted = rm.risk_adjusted_action(action, np.array([0.001, -0.002]))
    assert adjusted == 0.0


def test_risk_adjusted_action_allows_valid():
    # Valid action remains unchanged
    cfg = RiskfolioConfig(max_position=2.0, min_position=-2.0, var_limit=0.02)
    rm = RiskfolioRiskManager(cfg)
    action = 1.0
    adjusted = rm.risk_adjusted_action(action, np.array([0.0, 0.001]))
    assert adjusted == action
