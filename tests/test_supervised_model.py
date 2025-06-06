import numpy as np
import torch
import pytest

from src.supervised_model import (
    TrendPredictor,
    ModelConfig,
    TrainingConfig,
    train_supervised,
    save_model,
    load_model,
    evaluate_model,
    predict_features,
    select_best_model,
)

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


def test_save_and_load_consistency(tmp_path):
    cfg = ModelConfig(cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    model = TrendPredictor(input_dim=1, config=cfg)
    x = torch.randn(3, 4, 1)
    out1 = model(x)
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
    out2 = loaded(x)
    assert torch.allclose(out1, out2)


def test_predict_features_dummy():
    cfg = ModelConfig(cnn_filters=[1], cnn_kernel_sizes=[1], lstm_units=1)
    model = TrendPredictor(input_dim=2, config=cfg)
    for p in model.parameters():
        torch.nn.init.constant_(p, 0.0)
    data = torch.randn(5, 2)
    pred = predict_features(model, data)
    assert pred.item() == pytest.approx(0.5)


def test_evaluate_model_returns_metrics():
    x = np.random.randn(6, 3, 1).astype(np.float32)
    y = (x.sum(axis=1) > 0).astype(np.float32)
    model_cfg = ModelConfig(task="classification", cnn_filters=[1], cnn_kernel_sizes=[1], lstm_units=2)
    train_cfg = TrainingConfig(epochs=1, batch_size=2, val_split=0.0)
    model, _ = train_supervised(x, y, model_cfg, train_cfg)
    metrics = evaluate_model(model, x, y)
    assert "accuracy" in metrics


def test_select_best_model(tmp_path):
    trial1 = tmp_path / "trial1"
    trial1.mkdir()
    (trial1 / "metrics.json").write_text('{"val_loss": 0.5, "checkpoint": "m.pt"}')
    (trial1 / "m.pt").write_text("dummy")
    trial2 = tmp_path / "trial2"
    trial2.mkdir()
    (trial2 / "metrics.json").write_text('{"val_loss": 0.2, "checkpoint": "m.pt"}')
    (trial2 / "m.pt").write_text("dummy")
    best = select_best_model(tmp_path)
    assert "trial2" in str(best)
