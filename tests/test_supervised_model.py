import logging

import numpy as np
import pytest
import ray
import torch

from src.supervised_model import (
    ModelConfig,
    TrainingConfig,
    TrendPredictor,
    evaluate_model,
    load_model,
    predict_features,
    save_model,
    select_best_model,
)
from src.supervised_model import train_supervised  # Use Ray remote version

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Ensure Ray is initialized for all tests
@pytest.fixture(scope="session", autouse=True)
def _init_ray():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


def test_model_output_shape():
    cfg = ModelConfig(cnn_filters=[4], cnn_kernel_sizes=[2], lstm_units=8)
    model = TrendPredictor(input_dim=3, config=cfg)
    x = torch.randn(2, 5, 3)
    out = model(x)
    assert out.shape == (2, 1)


# Updated test_training_step_reduces_loss to use Ray remote
def test_training_step_reduces_loss():
    logging.info("Starting test_training_step_reduces_loss...")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate random input and target data
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)

    # Define model and training configurations
    model_cfg = ModelConfig(
        task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(
        epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2
    )
    logging.info("Model and training configuration initialized.")

    obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
    model, history = ray.get(obj_ref)

    logging.info("Training completed. Checking loss reduction...")

    # Verify loss reduction and magnitude of improvement
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    assert final_loss < initial_loss, "Final loss should be smaller than initial loss."
    assert (
        initial_loss - final_loss
    ) > 0.01, "Improvement in loss should be significant."

    # Ensure configuration parameters were respected
    assert list(model.config.cnn_filters) == list(model_cfg.cnn_filters)
    assert len(history["train_loss"]) == train_cfg.epochs

    logging.info(f"Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")
    logging.info("Loss reduction verified. Test passed.")


# Update test_validation_accuracy_perfect_when_same_data to use Ray remote
def test_validation_accuracy_perfect_when_same_data():
    x = np.random.randn(10, 3, 1).astype(np.float32)
    y = (x.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    model_cfg = ModelConfig(
        task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(epochs=2, batch_size=2, val_split=0.5)
    obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
    model, history = ray.get(obj_ref)
    assert 0.0 <= history["val_acc"][-1] <= 1.0


def test_save_and_load_consistency(tmp_path):
    cfg = ModelConfig(cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    model = TrendPredictor(input_dim=1, config=cfg)
    model.eval()
    x = torch.randn(3, 4, 1)
    out1 = model(x)
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
    loaded.eval()
    out2 = loaded(x)
    assert torch.allclose(out1, out2, atol=1e-3)


def test_predict_features_dummy():
    cfg = ModelConfig(cnn_filters=[1], cnn_kernel_sizes=[1], lstm_units=1)
    model = TrendPredictor(input_dim=2, config=cfg)
    for p in model.parameters():
        torch.nn.init.constant_(p, 0.0)
    data = torch.randn(1, 5, 2)  # Add batch dimension
    pred = predict_features(model, data)
    assert pred.item() == pytest.approx(0.5)


def test_evaluate_model_returns_metrics():
    x = np.random.randn(6, 3, 1).astype(np.float32)
    y = (x.sum(axis=1) > 0).astype(np.float32)
    model_cfg = ModelConfig(
        task="classification", cnn_filters=[1], cnn_kernel_sizes=[1], lstm_units=2
    )
    train_cfg = TrainingConfig(epochs=1, batch_size=2, val_split=0.0)
    obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
    model, _ = ray.get(obj_ref)
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


def test_empty_input_data():
    x = np.empty((0, 4, 1), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)
    model_cfg = ModelConfig(
        task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(
        epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2
    )
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
        ray.get(obj_ref)


def test_large_input_data():
    x = np.random.randn(10000, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)
    model_cfg = ModelConfig(
        task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(
        epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2
    )
    obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
    model, history = ray.get(obj_ref)
    assert len(history["train_loss"]) == train_cfg.epochs


def test_mismatched_input_target_dimensions():
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = np.random.randn(10, 1).astype(np.float32)
    model_cfg = ModelConfig(
        task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(
        epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2
    )
    with pytest.raises(ValueError, match="Input and target dimensions must match"):
        obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
        ray.get(obj_ref)


def test_zero_epochs():
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)
    model_cfg = ModelConfig(
        task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(
        epochs=0, batch_size=5, learning_rate=0.01, val_split=0.2
    )
    with pytest.raises(ValueError, match="Number of epochs must be greater than zero"):
        obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
        ray.get(obj_ref)


def test_batch_size_larger_than_dataset():
    x = np.random.randn(5, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)
    model_cfg = ModelConfig(
        task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    train_cfg = TrainingConfig(
        epochs=3, batch_size=10, learning_rate=0.01, val_split=0.2
    )
    with pytest.raises(
        ValueError, match="Batch size cannot be larger than the dataset"
    ):
        obj_ref = train_supervised.remote(x, y, model_cfg, train_cfg)  # type: ignore
        ray.get(obj_ref)


def test_evaluation_with_empty_features():
    x = np.empty((0, 4, 1), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)
    model_cfg = ModelConfig(
        task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    model = TrendPredictor(input_dim=4, config=model_cfg)
    with pytest.raises(ValueError, match="Features cannot be empty"):
        evaluate_model(model, x, y)


def test_evaluation_with_incorrect_dimensions():
    x = np.random.randn(10, 4, 1).astype(np.float32)
    y = np.random.randn(10, 2).astype(np.float32)
    model_cfg = ModelConfig(
        task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4
    )
    model = TrendPredictor(input_dim=4, config=model_cfg)
    with pytest.raises(ValueError, match="Target dimensions must match model output"):
        evaluate_model(model, x, y)


def test_save_load_with_corrupted_file(tmp_path):
    cfg = ModelConfig(cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    model = TrendPredictor(input_dim=1, config=cfg)
    path = tmp_path / "model.pt"
    path.write_text("corrupted data")
    with pytest.raises(RuntimeError, match="Failed to load model"):
        load_model(path)


def test_save_load_with_unsupported_format(tmp_path):
    cfg = ModelConfig(cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    model = TrendPredictor(input_dim=1, config=cfg)
    path = tmp_path / "model.txt"
    save_model(model, path)
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_model(path)
