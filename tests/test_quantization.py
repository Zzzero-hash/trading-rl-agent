import torch
import pytest
import torch.nn as nn

from src.utils.quantization import quantize_model, dynamic_quantization, compare_model_sizes


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def test_dynamic_quantization_changes_layer_types():
    model = SimpleModel()
    q_model = dynamic_quantization(model)
    assert isinstance(q_model.fc1, torch.nn.quantized.dynamic.Linear)
    assert isinstance(q_model.fc2, torch.nn.quantized.dynamic.Linear)


def test_quantize_model_dynamic():
    model = SimpleModel()
    q_model = quantize_model(model, quantization_type="dynamic", backend="fbgemm")
    assert isinstance(q_model.fc1, torch.nn.quantized.dynamic.Linear)
    assert isinstance(q_model.fc2, torch.nn.quantized.dynamic.Linear)


def test_compare_model_sizes_reduces_size():
    model = SimpleModel()
    q_model = dynamic_quantization(model)
    metrics = compare_model_sizes(model, q_model)
    assert metrics["quantized_size_mb"] < metrics["original_size_mb"]
    assert metrics["size_reduction_percent"] >= 0

def test_quantize_model_invalid_type():
    model = SimpleModel()
    with pytest.raises(ValueError):
        quantize_model(model, quantization_type="unknown")

