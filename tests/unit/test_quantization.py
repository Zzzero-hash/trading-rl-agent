import io

import torch
from torch import nn
from torch.quantization import quantize_dynamic


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def _model_size(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


def test_dynamic_quantization_changes_layer_types():
    model = SimpleModel()
    q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    assert isinstance(q_model.fc1, torch.nn.quantized.dynamic.Linear)
    assert isinstance(q_model.fc2, torch.nn.quantized.dynamic.Linear)


def test_dynamic_quantization_size_reasonable():
    model = SimpleModel()
    q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    orig = _model_size(model)
    quant = _model_size(q_model)
    assert quant > 0
    assert quant / orig < 2
