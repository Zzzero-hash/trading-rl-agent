import sys
from pathlib import Path
import types
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "structlog" not in sys.modules:
    stub = types.SimpleNamespace(
        BoundLogger=object,
        stdlib=types.SimpleNamespace(
            ProcessorFormatter=object,
            BoundLogger=object,
            LoggerFactory=lambda: None,
            filter_by_level=lambda *a, **k: None,
            add_logger_name=lambda *a, **k: None,
            add_log_level=lambda *a, **k: None,
            PositionalArgumentsFormatter=lambda: None,
            wrap_for_formatter=lambda f: f,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda **_: None,
            StackInfoRenderer=lambda **_: None,
            format_exc_info=lambda **_: None,
            UnicodeDecoder=lambda **_: None,
        ),
        dev=types.SimpleNamespace(ConsoleRenderer=lambda **_: None),
        configure=lambda **_: None,
        get_logger=lambda name=None: logging.getLogger(name),
    )
    sys.modules["structlog"] = stub

if "src.envs.finrl_trading_env" not in sys.modules:
    sys.modules["src.envs.finrl_trading_env"] = types.SimpleNamespace(register_env=lambda: None)

if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent")]
    sys.modules["trading_rl_agent"] = pkg

if "nltk.sentiment.vader" not in sys.modules:
    dummy = types.ModuleType("nltk.sentiment.vader")
    class DummySIA:
        def polarity_scores(self, text):
            """
            Return a neutral sentiment score for the given text.
            
            Parameters:
                text (str): The input text to analyze.
            
            Returns:
                dict: A dictionary with a single key "compound" set to 0.0, indicating neutral sentiment.
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
import pytest

pytestmark = pytest.mark.unit


def test_execution_package_missing():
    """
    Test that importing 'trading_rl_agent.execution' raises a ModuleNotFoundError.
    """
    with pytest.raises(ModuleNotFoundError):
        import trading_rl_agent.execution  # noqa: F401


def test_monitoring_package_missing():
    """
    Test that importing the 'trading_rl_agent.monitoring' subpackage raises a ModuleNotFoundError.
    """
    with pytest.raises(ModuleNotFoundError):
        import trading_rl_agent.monitoring  # noqa: F401
