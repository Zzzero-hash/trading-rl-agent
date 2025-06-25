import pytest

from src.risk.production_risk_manager import ProductionRiskManager


@pytest.fixture
def risk_manager():
    config = {
        "max_position": 0.5,
        "min_position": -0.5,
        "var_limit": 0.005,
        "circuit_breaker_rules": {},
    }
    return ProductionRiskManager(config)


def test_position_limit_violation(risk_manager):
    # Action beyond max_position should be invalid
    action = 1.0
    portfolio = {}
    assert not risk_manager.validate_action(action, portfolio)


def test_var_limit_violation(risk_manager):
    # Override calculate_risk to return high Var
    def high_var(action, portfolio):
        return {"var": 0.01, "drawdown": 0.0}

    risk_manager.calculate_risk = high_var
    assert not risk_manager.validate_action(0.0, {})


def test_valid_action_within_limits():
    # Default var is 0.01, var_limit set to 0.02 allows valid actions
    config = {
        "max_position": 1.0,
        "min_position": -1.0,
        "var_limit": 0.02,
        "circuit_breaker_rules": {},
    }
    rm = ProductionRiskManager(config)
    assert rm.validate_action(0.0, {})


def test_risk_adjusted_action_halts():
    # Invalid action should be adjusted to zero (halt)
    config = {
        "max_position": 0.5,
        "min_position": -0.5,
        "var_limit": 0.005,
        "circuit_breaker_rules": {},
    }
    rm = ProductionRiskManager(config)
    action = 1.0
    adjusted = rm.risk_adjusted_action(action, {})
    assert adjusted == 0.0


def test_risk_adjusted_action_allows_valid():
    # Valid action remains unchanged
    config = {
        "max_position": 2.0,
        "min_position": -2.0,
        "var_limit": 0.02,
        "circuit_breaker_rules": {},
    }
    rm = ProductionRiskManager(config)
    action = 1.0
    adjusted = rm.risk_adjusted_action(action, {})
    assert adjusted == action
