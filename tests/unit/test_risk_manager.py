import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trade_agent.risk.manager import RiskManager
from trade_agent.risk.position_sizer import kelly_position_size

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "src.envs.finrl_trading_env" not in sys.modules:
    sys.modules["src.envs.finrl_trading_env"] = types.SimpleNamespace(
        register_env=lambda: None,
    )

if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"),
    ]
    sys.modules["trading_rl_agent"] = pkg

if "nltk.sentiment.vader" not in sys.modules:
    dummy = types.ModuleType("nltk.sentiment.vader")

    class DummySIA:
        def polarity_scores(self, _):
            """
            Return a neutral sentiment score for the given text.

            Parameters:
                text (str): The input text to analyze.

            Returns:
                dict: A dictionary with a single key 'compound' set to 0.0, indicating neutral sentiment.
            """
            return {"compound": 0.0}

    dummy.SentimentIntensityAnalyzer = DummySIA
    sys.modules["nltk.sentiment.vader"] = dummy
base = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
for pkg in ["features", "portfolio", "risk"]:
    key = f"trading_rl_agent.{pkg}"
    if key not in sys.modules:
        mod = types.ModuleType(key)
        mod.__path__ = [str(base / pkg)]
        sys.modules[key] = mod

pytestmark = pytest.mark.unit


def _sample_returns():
    """
    Generate reproducible sample daily return series for two assets over 60 days.

    Returns:
        dict: A dictionary with keys "A" and "B", each mapping to a pandas Series of simulated daily returns for 60 consecutive dates.
    """
    np.random.seed(42)  # Seed the random number generator for reproducibility
    dates = pd.date_range("2023-01-01", periods=60)
    a = pd.Series(np.random.normal(0, 0.01, len(dates)), index=dates)
    b = pd.Series(np.random.normal(0, 0.02, len(dates)), index=dates)
    return {"A": a, "B": b}


def test_var_and_cvar():
    """
    Test that RiskManager calculates portfolio VaR and CVaR correctly for a sample portfolio.

    Asserts that VaR is non-negative, CVaR is at least as large as VaR, CVaR exceeds VaR by at least 10%,
    and VaR falls within a reasonable daily range for the provided synthetic returns.
    """
    rm = RiskManager()
    rm.update_returns_data(_sample_returns())
    weights = {"A": 0.5, "B": 0.5}
    var = rm.calculate_portfolio_var(weights)
    cvar = rm.calculate_portfolio_cvar(weights)
    assert var >= 0
    assert cvar >= var
    # Verify CVaR is meaningfully larger than VaR for typical distributions
    assert cvar > var * 1.1  # CVaR should be at least 10% higher than VaR
    # Verify reasonable magnitude for the given volatility
    assert 0.001 <= var <= 0.1  # Reasonable range for daily VaR


def test_correlation_and_concentration():
    """
    Test the calculation of correlation and concentration risk metrics in the RiskManager.

    Verifies that correlation and concentration risks are within valid bounds for typical and edge-case portfolio weights, including maximum and minimum concentration scenarios.
    """
    rm = RiskManager()
    data = _sample_returns()
    rm.update_returns_data(data)
    weights = {"A": 0.6, "B": 0.4}
    corr = rm.calculate_correlation_risk(weights)
    conc = rm.calculate_concentration_risk(weights)
    assert 0 <= corr <= 1
    assert 0 <= conc <= 1

    # Test extreme concentration (single asset)
    extreme_weights = {"A": 1.0, "B": 0.0}
    extreme_conc = rm.calculate_concentration_risk(extreme_weights)
    assert extreme_conc == 1.0  # Maximum concentration

    # Test equal weights (minimum concentration for 2 assets)
    equal_weights = {"A": 0.5, "B": 0.5}
    equal_conc = rm.calculate_concentration_risk(equal_weights)
    assert equal_conc == 0.5  # Expected value for equal 2-asset portfolio


def test_kelly_position_size():
    """
    Test the Kelly position size calculation of the RiskManager for various scenarios.

    Verifies that the calculated position size is within expected bounds for profitable, unprofitable, and
    high win rate scenarios, and that the method adheres to the properties of the Kelly formula.
    """
    rm = RiskManager()
    # Test with profitable scenario (positive expected return)
    size = rm.calculate_kelly_position_size(0.1, 0.6, 0.05, 0.02)
    assert 0 <= size <= 0.25

    # Test with unprofitable scenario (negative expected return)
    size_negative = rm.calculate_kelly_position_size(-0.05, 0.4, 0.02, 0.03)
    assert size_negative == 0  # Should return 0 for negative expected return

    # Test with very high win rate
    size_high_wr = rm.calculate_kelly_position_size(0.15, 0.9, 0.10, 0.05)
    assert 0 < size_high_wr <= 0.25  # Should be positive but capped

    # Verify Kelly formula properties
    expected_return, win_rate, avg_win, avg_loss = 0.08, 0.55, 0.04, 0.03
    kelly_raw = (expected_return * win_rate - (1 - win_rate) * avg_loss) / avg_win
    size_calculated = rm.calculate_kelly_position_size(
        expected_return,
        win_rate,
        avg_win,
        avg_loss,
    )
    if kelly_raw > 0:
        assert size_calculated > 0  # Should be positive when Kelly formula is positive


def test_portfolio_drawdown_and_function():
    rm = RiskManager()
    rm.update_returns_data(_sample_returns())
    weights = {"A": 0.5, "B": 0.5}
    dd = rm.calculate_portfolio_drawdown(weights)
    assert 0 <= dd <= 1

    # Module-level Kelly function
    size = kelly_position_size(0.1, 0.6, 0.05, 0.02)
    assert 0 < size <= 0.25
