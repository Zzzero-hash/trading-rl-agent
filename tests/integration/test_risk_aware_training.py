import sys
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

if "structlog" not in sys.modules:
    import types
    import logging

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

from trading_rl_agent.agents.trainer import Trainer

pytestmark = pytest.mark.integration

def test_risk_aware_training(tmp_path):
    df = pd.DataFrame({
        "open": [1.0] * 20,
        "high": [1.0] * 20,
        "low": [1.0] * 20,
        "close": [1.0] * 20,
        "volume": [1.0] * 20,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    env_cfg = {"dataset_paths": [str(csv_path)], "window_size": 5}
    trainer_cfg = {
        "algorithm": "ppo",
        "num_iterations": 1,
        "ray_config": {"env": "TraderEnv"},
        "risk_management": {"enabled": True, "var_limit": 0.0001},
    }

    with (
        patch("ray.init"),
        patch("ray.is_initialized", return_value=False),
        patch("ray.shutdown"),
        patch("ray.tune.registry.register_env") as reg_env,
        patch("trading_rl_agent.envs.finrl_trading_env.register_env"),
        patch("trading_rl_agent.agents.trainer.tune.Tuner") as tuner_cls,
    ):
        tuner = Mock()
        tuner.fit.return_value = Mock()
        tuner_cls.return_value = tuner

        trainer = Trainer(env_cfg, {}, trainer_cfg, save_dir=str(tmp_path))
        env_creator = reg_env.call_args[0][1]
        env = env_creator(env_cfg)
        assert hasattr(env, "risk_manager")
        trainer.train()
        assert tuner_cls.called
        assert tuner.fit.called

