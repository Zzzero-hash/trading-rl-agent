import pytest
import torch
import yaml

from trading_rl_agent.models.cnn_lstm import CNNLSTMModel, _load_config


def test_load_config_from_path(tmp_path):
    cfg = {"cnn_filters": [4], "cnn_kernel_sizes": [2], "lstm_units": 8, "dropout": 0.0}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.dump(cfg))
    loaded = _load_config(str(path))
    assert loaded == cfg


def test_forward_invalid_features():
    model = CNNLSTMModel(
        input_dim=3,
        config={"cnn_filters": [4], "cnn_kernel_sizes": [2]},
    )
    x = torch.randn(2, 5, 4)  # wrong feature dimension
    with pytest.raises(ValueError):
        model(x)
