import os
import types
from unittest import mock
import pytest

from src.agents.configs import EnsembleConfig
from src.agents import trainer as trainer_module


def test_ensemble_config_alias_and_validation():
    cfg = EnsembleConfig(agent_configs={"sac": {"enabled": True, "config": None}})
    assert cfg.agents == {"sac": {"enabled": True, "config": None}}


def test_ensemble_config_empty_agents():
    with pytest.raises(ValueError):
        EnsembleConfig(agents={})


def test_trainer_algorithm_case_insensitive(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(trainer_module.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(trainer_module.ray, "init", lambda **kw: calls.setdefault("init", kw))
    monkeypatch.setattr(trainer_module, "register_env", lambda: calls.setdefault("reg", True))
    monkeypatch.setattr(trainer_module.ray, "shutdown", lambda: calls.setdefault("shutdown", True))

    monkeypatch.setattr(trainer_module, "PPOTrainer", types.SimpleNamespace(__name__="PPOTrainer"))
    monkeypatch.setattr(trainer_module, "DQNTrainer", types.SimpleNamespace(__name__="DQNTrainer"))

    class FakeTuner:
        def __init__(self, algo_cls, param_space=None, run_config=None):
            calls["algo"] = algo_cls

        def fit(self):
            calls["fit"] = True

    monkeypatch.setattr(trainer_module.tune, "Tuner", FakeTuner)

    env_cfg = {}
    model_cfg = {}
    trainer_cfg = {"algorithm": "DQN", "num_iterations": 1, "ray_config": {}}

    trainer = trainer_module.Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=str(tmp_path))
    assert trainer.algorithm == "dqn"
    trainer.train()
    assert calls["algo"].__name__ == "DQNTrainer"
    assert calls.get("fit")
    assert calls.get("shutdown")
