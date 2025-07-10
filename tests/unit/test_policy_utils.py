import logging
import sys
import types
from pathlib import Path

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
        def polarity_scores(self, text):
            """
            Return a dictionary with a fixed compound sentiment score of 0.0 for the given text.

            Parameters:
                text (str): The input text to analyze.

            Returns:
                dict: A dictionary with a single key "compound" set to 0.0.
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
import numpy as np
import pytest
from gymnasium import spaces
from ray.rllib.policy.policy import Policy

from trading_rl_agent.agents.policy_utils import (
    CallablePolicy,
    WeightedEnsembleAgent,
    weighted_policy_mapping,
)

pytestmark = pytest.mark.unit


def test_callable_policy_simple():
    """
    Test that CallablePolicy returns the expected fixed action for a simple input.

    Verifies that a CallablePolicy using a lambda returning a constant action produces the correct action shape and value when given a single observation.
    """
    policy = CallablePolicy(
        spaces.Box(-1, 1, (1,), dtype=np.float32),
        spaces.Box(-1, 1, (1,), dtype=np.float32),
        lambda obs: np.array([0.5], dtype=np.float32),
    )
    acts, _, _ = policy.compute_actions([np.zeros(1, dtype=np.float32)])
    assert acts.shape == (1, 1)
    assert np.isclose(acts[0, 0], 0.5)


def test_weighted_policy_mapping():
    """
    Test that `weighted_policy_mapping` returns a valid policy key from the provided weighted mapping.

    Asserts that the mapping function, when called with an agent ID, selects one of the expected policy names.
    """
    mapping = weighted_policy_mapping({"a": 1.0, "b": 1.0})
    choice = mapping("agent0")
    assert choice in {"a", "b"}


def test_weighted_ensemble_agent(monkeypatch):
    """
    Test that WeightedEnsembleAgent selects an action from the expected set when using dummy policies with fixed outputs.

    Verifies that the agent's selected action is either 1.0 or -1.0 and has the correct shape when policies with deterministic outputs are used in the ensemble.
    """
    obs_space = spaces.Box(-1, 1, (1,), dtype=np.float32)
    act_space = spaces.Box(-1, 1, (1,), dtype=np.float32)

    class DummyPolicy(Policy):
        def __init__(self, name):
            """
            Initialize a DummyPolicy instance with a given name.

            Parameters:
                name (str): The name assigned to this policy instance.
            """
            super().__init__(obs_space, act_space, {})
            self.name = name

        def compute_single_action(self, obs, **kwargs):
            """
            Compute a single action based on the policy's name.

            Returns:
                tuple: A tuple containing the action as a NumPy array, an empty list, and an empty dictionary.
            """
            val = 1.0 if self.name == "a" else -1.0
            return np.array([val], dtype=np.float32), [], {}

        def compute_actions(self, obs_batch, **kwargs):
            """
            Compute actions for a batch of observations.

            Parameters:
                obs_batch (iterable): A batch of observations to process.

            Returns:
                tuple: A tuple containing the stacked actions as a NumPy array, an empty list, and an empty dictionary.
            """
            acts = [self.compute_single_action(obs)[0] for obs in obs_batch]
            return np.stack(acts), [], {}

    policies = {"a": DummyPolicy("a"), "b": DummyPolicy("b")}
    agent = WeightedEnsembleAgent(policies, {"a": 1.0, "b": 1.0})
    action = agent.select_action(np.zeros(1, dtype=np.float32))
    assert action.shape == (1,)
    assert action[0] in {1.0, -1.0}
