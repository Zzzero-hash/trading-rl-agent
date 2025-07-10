import numpy as np
import pandas as pd
import pytest
import torch

from trading_rl_agent.supervised_model import (
    ModelConfig,
    TrendPredictor,
    _to_tensor,
    evaluate_model,
    predict_features,
)


def test_to_tensor_various_types():
    arr = np.array([[1, 2]], dtype=np.float64)
    df = pd.DataFrame(arr)
    tens = torch.tensor(arr, dtype=torch.float32)

    assert _to_tensor(arr).dtype == torch.float32
    assert _to_tensor(df).dtype == torch.float32
    t = _to_tensor(tens)
    assert t.dtype == torch.float32
    assert t.shape == tens.shape


def test_evaluate_model_classification_metrics():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            self.task = "classification"
            self.config = ModelConfig(output_size=1, task="classification")

        def forward(self, x):
            return torch.ones(x.size(0), 1)

    x = np.zeros((4, 3, 1), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)
    model = DummyModel()
    metrics = evaluate_model(model, x, y)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_predict_features_invalid_shape():
    cfg = ModelConfig()
    model = TrendPredictor(input_dim=1, config=cfg)
    bad_input = torch.randn(4)
    with pytest.raises(ValueError):
        predict_features(model, bad_input)
