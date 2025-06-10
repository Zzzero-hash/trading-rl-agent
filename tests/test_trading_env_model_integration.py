import numpy as np
import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.integration

from src.envs.trading_env import TradingEnv
from src.supervised_model import TrendPredictor, ModelConfig, save_model


@pytest.fixture
def dummy_model(tmp_path):
    cfg = ModelConfig(cnn_filters=[1], cnn_kernel_sizes=[1], lstm_units=1)
    model = TrendPredictor(input_dim=2, config=cfg)
    for p in model.parameters():
        torch.nn.init.constant_(p, 0.0)
    path = tmp_path / "model.pt"
    save_model(model, path)
    return str(path)


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "open": np.linspace(1, 2, 60),
        "high": np.linspace(1, 2, 60),
        "low": np.linspace(1, 2, 60),
        "close": np.linspace(1, 2, 60),
        "volume": np.ones(60),
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_reset_includes_model_pred(sample_csv, dummy_model):
    env = TradingEnv({"dataset_paths": [sample_csv], "window_size": 5, "model_path": dummy_model})
    obs, _ = env.reset()
    assert "model_pred" in obs
    assert obs["model_pred"].shape == (1,)


def test_step_includes_model_pred(sample_csv, dummy_model):
    env = TradingEnv({"dataset_paths": [sample_csv], "window_size": 5, "model_path": dummy_model})
    env.reset()
    obs, _, _, _, _ = env.step(0)
    assert "model_pred" in obs
    assert obs["model_pred"].shape == (1,)
