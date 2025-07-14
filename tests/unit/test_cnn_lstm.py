import torch

from trading_rl_agent.models import CNNLSTMModel


def test_model_instantiation_from_yaml():
    config = {
        "cnn_filters": [16, 32],
        "cnn_kernel_sizes": [3, 3],
        "lstm_units": 64,
        "dropout": 0.1,
    }
    model = CNNLSTMModel(input_dim=5, config=config)
    assert isinstance(model, CNNLSTMModel)


def test_forward_pass():
    cfg = {
        "cnn_filters": [4],
        "cnn_kernel_sizes": [2],
        "lstm_units": 8,
        "dropout": 0.0,
    }
    model = CNNLSTMModel(input_dim=3, config=cfg)
    x = torch.randn(2, 10, 3)
    out = model(x)
    assert out.shape == (2, 1)
