import logging
from pathlib import Path
import sys
import types

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
    sys.modules["src.envs.finrl_trading_env"] = types.SimpleNamespace(
        register_env=lambda: None
    )

if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [
        str(Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent")
    ]
    sys.modules["trading_rl_agent"] = pkg

if "nltk.sentiment.vader" not in sys.modules:
    dummy = types.ModuleType("nltk.sentiment.vader")

    class DummySIA:
        def polarity_scores(self, text):
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

from datetime import datetime

import pytest

from trading_rl_agent.monitoring import AlertManager, Dashboard, MetricsCollector
from trading_rl_agent.risk.manager import RiskMetrics

pytestmark = pytest.mark.unit


def test_metricscollector_log_dict():
    collector = MetricsCollector()
    data = {"sharpe_ratio": 1.2}
    collector.log_metrics(data)
    assert collector.get_latest() == data
    assert collector.history[-1]["sharpe_ratio"] == 1.2


def test_metricscollector_log_dataclass():
    metrics = RiskMetrics(
        portfolio_var=1.0,
        portfolio_cvar=1.5,
        max_drawdown=0.1,
        current_drawdown=0.05,
        leverage=1.0,
        sharpe_ratio=1.2,
        sortino_ratio=1.1,
        beta=0.3,
        correlation_risk=0.2,
        concentration_risk=0.1,
        timestamp=datetime.utcnow(),
    )
    collector = MetricsCollector()
    collector.log_metrics(metrics)
    latest = collector.get_latest()
    assert isinstance(latest, dict)
    assert latest["portfolio_var"] == 1.0


def test_alert_manager_send():
    am = AlertManager()
    am.send_alert("test alert")
    assert am.alerts[-1] == "test alert"


def test_dashboard_update_and_latest():
    dash = Dashboard()
    metrics = {"return": 0.05}
    dash.update(metrics)
    assert dash.get_latest() == metrics
