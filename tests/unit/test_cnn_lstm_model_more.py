import torch
import yaml
import pytest

from src.models import CNNLSTMModel

pytestmark = pytest.mark.unit


def test_attention_forward_pass():
    cfg = {"cnn_filters": [2], "cnn_kernel_sizes": [2], "lstm_units": 4, "dropout": 0.0}
    model = CNNLSTMModel(input_dim=3, config=cfg, use_attention=True)
    x = torch.randn(1, 5, 3)
    out = model(x)
    assert out.shape == (1, 1)


def test_invalid_config(tmp_path):
    bad_cfg = tmp_path / "cfg.yaml"
    bad_cfg.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError):
        CNNLSTMModel(input_dim=2, config=str(bad_cfg))


def test_feature_mismatch():
    model = CNNLSTMModel(input_dim=4, config={"cnn_filters": [2], "cnn_kernel_sizes": [2], "lstm_units": 4, "dropout":0.0})
    bad_input = torch.randn(1, 5, 3)
    with pytest.raises(ValueError):
        model(bad_input)
