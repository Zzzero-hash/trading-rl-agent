import types
from unittest import mock

from trading_rl_agent.agents import trainer as trainer_module


def test_dqn_algorithm_selected(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr(trainer_module.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(
        trainer_module.ray, "init", lambda **kw: calls.setdefault("init", kw)
    )
    monkeypatch.setattr(
        trainer_module, "register_env", lambda: calls.setdefault("reg", True)
    )
    monkeypatch.setattr(
        trainer_module.ray, "shutdown", lambda: calls.setdefault("shutdown", True)
    )

    monkeypatch.setattr(
        trainer_module, "DQNTrainer", types.SimpleNamespace(__name__="DQN")
    )
    tuner = types.SimpleNamespace(fit=lambda: calls.setdefault("fit", True))
    monkeypatch.setattr(trainer_module.tune, "Tuner", lambda *a, **k: tuner)

    env_cfg = {}
    model_cfg = {}
    trainer_cfg = {"algorithm": "dqn"}
    t = trainer_module.Trainer(env_cfg, model_cfg, trainer_cfg, save_dir=str(tmp_path))
    assert t.algorithm == "dqn"
    t.train()
    assert calls.get("fit")
    assert calls.get("shutdown")
