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
import numpy as np
import pytest
from gymnasium import spaces
from ray.rllib.policy.policy import Policy

from trading_rl_agent.agents.policy_utils import CallablePolicy, weighted_policy_mapping, WeightedEnsembleAgent

pytestmark = pytest.mark.unit


def test_callable_policy_simple():
    policy = CallablePolicy(spaces.Box(-1, 1, (1,), dtype=np.float32),
                            spaces.Box(-1, 1, (1,), dtype=np.float32),
                            lambda obs: np.array([0.5], dtype=np.float32))
    acts, _, _ = policy.compute_actions([np.zeros(1, dtype=np.float32)])
    assert acts.shape == (1, 1)
    assert np.isclose(acts[0, 0], 0.5)


def test_weighted_policy_mapping():
    mapping = weighted_policy_mapping({"a": 1.0, "b": 1.0})
    choice = mapping("agent0")
    assert choice in {"a", "b"}


def test_weighted_ensemble_agent(monkeypatch):
    obs_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
    act_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

    class DummyPolicy(Policy):
        def __init__(self, name):
            super().__init__(obs_space, act_space, {})
            self.name = name
        def compute_single_action(self, obs, **kwargs):
            val = 1.0 if self.name == "a" else -1.0
            return np.array([val], dtype=np.float32), [], {}
        def compute_actions(self, obs_batch, **kwargs):
            acts = [self.compute_single_action(obs)[0] for obs in obs_batch]
            return np.stack(acts), [], {}

    policies = {"a": DummyPolicy("a"), "b": DummyPolicy("b")}
    agent = WeightedEnsembleAgent(policies, {"a": 1.0, "b": 1.0})
    action = agent.select_action(np.zeros(1, dtype=np.float32))
    assert action.shape == (1,)
    assert action[0] in {1.0, -1.0}
