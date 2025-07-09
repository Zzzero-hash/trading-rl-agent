"""Unit tests for PPOAgent wrapper."""

import logging
import os
from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import pytest

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

from trading_rl_agent.agents.ppo_agent import PPOAgent


@pytest.mark.unit
class TestPPOAgent:
    """Basic tests for PPOAgent."""

    def test_initialization(self):
        agent = PPOAgent(state_dim=4, action_dim=2)
        assert agent.state_dim == 4
        assert agent.action_dim == 2
        assert hasattr(agent, "model")

    def test_select_action(self):
        agent = PPOAgent(state_dim=3, action_dim=1)
        state = np.zeros(3, dtype=np.float32)
        action = agent.select_action(state)
        assert action.shape == (1,)
        action_det = agent.select_action(state, evaluate=True)
        assert action_det.shape == (1,)

    def test_save_and_load(self):
        agent = PPOAgent(state_dim=3, action_dim=1)
        state = np.random.randn(3).astype(np.float32)
        action_before = agent.select_action(state, evaluate=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ppo_test_model")
            agent.save(path)
            new_agent = PPOAgent(state_dim=3, action_dim=1)
            new_agent.load(path)
            action_after = new_agent.select_action(state, evaluate=True)
            np.testing.assert_array_almost_equal(action_before, action_after)
