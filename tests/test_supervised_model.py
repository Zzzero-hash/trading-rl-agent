import numpy as np
import torch

from src.supervised_model import TrendPredictor, ModelConfig, TrainingConfig, train_supervised


def test_model_output_shape():
    cfg = ModelConfig(cnn_filters=[4], cnn_kernel_sizes=[2], lstm_units=8)
    model = TrendPredictor(input_dim=3, config=cfg)
    x = torch.randn(2, 5, 3)
    out = model(x)
    assert out.shape == (2, 1)


def test_training_step_reduces_loss():
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = x.sum(axis=1)
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2)
    _, history = train_supervised(x, y, model_cfg, train_cfg)
    assert history["train_loss"][0] > history["train_loss"][-1]


def test_validation_accuracy_perfect_when_same_data():
    x = np.random.randn(10, 3, 1).astype(np.float32)
    y = (x.sum(axis=1) > 0).astype(np.float32)
    model_cfg = ModelConfig(task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=2, batch_size=2, val_split=0.5)
    _, history = train_supervised(x, y, model_cfg, train_cfg)
    assert 0.0 <= history["val_acc"][-1] <= 1.0
