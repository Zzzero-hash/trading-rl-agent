import sys
from pathlib import Path
from unittest.mock import Mock, patch
import types
import importlib.util

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

import pandas as pd
import pytest

base_path = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"

spec_syn = importlib.util.spec_from_file_location(
    "trading_rl_agent.data.synthetic", base_path / "data" / "synthetic.py"
)
synthetic_mod = importlib.util.module_from_spec(spec_syn)
spec_syn.loader.exec_module(synthetic_mod)  # type: ignore
fetch_synthetic_data = synthetic_mod.fetch_synthetic_data

spec_feat = importlib.util.spec_from_file_location(
    "trading_rl_agent.data.features", base_path / "data" / "features.py"
)
features_mod = importlib.util.module_from_spec(spec_feat)
spec_feat.loader.exec_module(features_mod)  # type: ignore
generate_features = features_mod.generate_features

spec_trainer = importlib.util.spec_from_file_location(
    "trading_rl_agent.agents.trainer", base_path / "agents" / "trainer.py"
)
trainer_mod = importlib.util.module_from_spec(spec_trainer)
spec_trainer.loader.exec_module(trainer_mod)  # type: ignore
Trainer = trainer_mod.Trainer

spec_pm = importlib.util.spec_from_file_location(
    "trading_rl_agent.portfolio.manager", base_path / "portfolio" / "manager.py"
)
pm_mod = importlib.util.module_from_spec(spec_pm)
spec_pm.loader.exec_module(pm_mod)  # type: ignore
PortfolioManager = pm_mod.PortfolioManager


@pytest.mark.integration
@pytest.mark.e2e
def test_data_ingestion_training_portfolio(tmp_path, benchmark):
    """End-to-end test: data ingestion → training → portfolio update."""
    # ---------------------- Data Ingestion ----------------------
    def create_data():
        return fetch_synthetic_data(n_samples=50)

    import time
    start = time.perf_counter()
    df = create_data()
    df = generate_features(df)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    pipeline_time = time.perf_counter() - start
    benchmark.extra_info["pipeline_time"] = pipeline_time

    # ----------------------- Agent Training ---------------------
    env_cfg = {"dataset_paths": [str(csv_path)], "window_size": 5, "initial_balance": 1000}
    model_cfg = {"architecture": "ppo"}
    trainer_cfg = {"algorithm": "ppo", "num_iterations": 1, "ray_config": {"env": "TraderEnv"}}

    with (
        patch("ray.init"),
        patch("ray.is_initialized", return_value=False),
        patch("trading_rl_agent.envs.finrl_trading_env.register_env"),
        patch("ray.shutdown"),
        patch("trading_rl_agent.agents.trainer.tune.Tuner") as tuner_cls,
    ):
        tuner = Mock()
        tuner.fit.return_value = Mock()
        tuner_cls.return_value = tuner

        trainer = Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=str(tmp_path))
        benchmark.pedantic(trainer.train, rounds=1, iterations=1)

        assert tuner_cls.call_count >= 1
        assert tuner.fit.call_count >= 1

    # ------------------- Portfolio Update ----------------------
    pm = PortfolioManager(1000.0)
    assert pm.execute_trade("TEST", 5, 10.0)
    pm.update_prices({"TEST": 12.0})

    assert pm.performance_history
    last = pm.performance_history[-1]
    assert last["total_value"] > 1000.0
