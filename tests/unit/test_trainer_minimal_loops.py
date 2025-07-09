import types
from unittest.mock import Mock, patch

import pytest

from trading_rl_agent.agents import trainer as trainer_module

pytestmark = pytest.mark.unit


def _make_configs(tmp_path, sample_csv_file):
    env_cfg = {"dataset_paths": [sample_csv_file]}
    model_cfg = {}
    trainer_cfg = {}
    return env_cfg, model_cfg, trainer_cfg


def test_train_invokes_tuner(tmp_path, sample_csv_file):
    env_cfg, model_cfg, trainer_cfg = _make_configs(tmp_path, sample_csv_file)
    calls = {}
    with (
        patch.object(trainer_module.ray, "is_initialized", return_value=False),
        patch.object(
            trainer_module.ray, "init", lambda **kw: calls.setdefault("init", kw)
        ),
        patch.object(trainer_module, "register_env", lambda: None),
        patch.object(
            trainer_module.ray, "shutdown", lambda: calls.setdefault("shutdown", True)
        ),
        patch.object(trainer_module.tune, "Tuner") as tuner_cls,
    ):
        tuner = types.SimpleNamespace(fit=lambda: calls.setdefault("fit", True))
        tuner_cls.return_value = tuner
        t = trainer_module.Trainer(
            env_cfg, model_cfg, trainer_cfg, save_dir=str(tmp_path)
        )
        t.train()
    assert calls.get("fit") and calls.get("shutdown")


def test_evaluate_and_test_not_implemented(tmp_path, sample_csv_file):
    env_cfg, model_cfg, trainer_cfg = _make_configs(tmp_path, sample_csv_file)
    with (
        patch.object(trainer_module.ray, "is_initialized", return_value=False),
        patch.object(trainer_module.ray, "init"),
        patch.object(trainer_module, "register_env"),
    ):
        t = trainer_module.Trainer(
            env_cfg, model_cfg, trainer_cfg, save_dir=str(tmp_path)
        )
        with pytest.raises(NotImplementedError):
            t.evaluate()
        with pytest.raises(NotImplementedError):
            t.test()
