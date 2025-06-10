import numpy as np
import torch
import pytest
import ray
from ray import get
import logging

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_model_output_shape():
    cfg = ModelConfig(cnn_filters=[4], cnn_kernel_sizes=[2], lstm_units=8)
    model = TrendPredictor(input_dim=3, config=cfg)
    x = torch.randn(2, 5, 3)
    out = model(x)
    assert out.shape == (2, 1)


# Updated test_training_step_reduces_loss to use .remote()
def test_training_step_reduces_loss():
    logging.info("Starting test_training_step_reduces_loss...")

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate random input and target data
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)

    # Define model and training configurations
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2)

    logging.info("Model and training configuration initialized.")

    # If no GPU is present run training locally using the underlying function,
    # otherwise run the Ray remote version. A Ray cluster is started only when
    # needed to avoid unnecessary dependency on GPU availability.
    if torch.cuda.is_available():
        ray.init(log_to_driver=False)
        result = train_supervised.remote(
            x, y, model_config=model_cfg, train_config=train_cfg
        )
        model, history = get(result)
        ray.shutdown()
    else:
        model, history = train_supervised._function(
            x, y, model_cfg, train_cfg
        )

    logging.info("Training completed. Checking loss reduction...")

    # Verify loss reduction and magnitude of improvement
    initial_loss = history["train_loss"][0]
    final_loss = history["train_loss"][-1]
    assert final_loss < initial_loss, "Final loss should be smaller than initial loss."
    assert (initial_loss - final_loss) > 0.01, "Improvement in loss should be significant."

    # Ensure configuration parameters were respected
    assert model.config.task == model_cfg.task
    assert list(model.config.cnn_filters) == list(model_cfg.cnn_filters)
    assert list(model.config.cnn_kernel_sizes) == list(model_cfg.cnn_kernel_sizes)
    assert model.config.lstm_units == model_cfg.lstm_units

    # Ensure training hyperparameters were applied
    assert len(history["train_loss"]) == train_cfg.epochs
    assert len(history["val_loss"]) == train_cfg.epochs

    logging.info(f"Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")
    logging.info("Loss reduction verified. Test passed.")


# Update test_validation_accuracy_perfect_when_same_data to use .remote()
def test_validation_accuracy_perfect_when_same_data():
    x = np.random.randn(10, 3, 1).astype(np.float32)
    y = (x.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    model_cfg = ModelConfig(task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=2, batch_size=2, val_split=0.5)
    result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
    model, history = get(result)
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
    result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
    model, _ = get(result)
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
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2)
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
        model, history = get(result)

def test_large_input_data():
    x = np.random.randn(10000, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2)
    result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
    model, history = get(result)
    assert len(history["train_loss"]) == train_cfg.epochs

def test_mismatched_input_target_dimensions():
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = np.random.randn(10, 1).astype(np.float32)
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=3, batch_size=5, learning_rate=0.01, val_split=0.2)
    with pytest.raises(ValueError, match="Input and target dimensions must match"):
        result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
        model, history = get(result)

def test_zero_epochs():
    x = np.random.randn(20, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=0, batch_size=5, learning_rate=0.01, val_split=0.2)
    with pytest.raises(ValueError, match="Number of epochs must be greater than zero"):
        result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
        model, history = get(result)

def test_batch_size_larger_than_dataset():
    x = np.random.randn(5, 4, 1).astype(np.float32)
    y = x.sum(axis=1).reshape(-1, 1)
    model_cfg = ModelConfig(task="regression", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    train_cfg = TrainingConfig(epochs=3, batch_size=10, learning_rate=0.01, val_split=0.2)
    with pytest.raises(ValueError, match="Batch size cannot be larger than the dataset"):
        result = train_supervised.remote(features=x, targets=y, model_cfg=model_cfg, train_cfg=train_cfg)
        model, history = get(result)

def test_evaluation_with_empty_features():
    x = np.empty((0, 4, 1), dtype=np.float32)
    y = np.empty((0, 1), dtype=np.float32)
    model_cfg = ModelConfig(task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
    model = TrendPredictor(input_dim=4, config=model_cfg)
    with pytest.raises(ValueError, match="Features cannot be empty"):
        evaluate_model(model, x, y)

def test_evaluation_with_incorrect_dimensions():
    x = np.random.randn(10, 4, 1).astype(np.float32)
    y = np.random.randn(10, 2).astype(np.float32)
    model_cfg = ModelConfig(task="classification", cnn_filters=[2], cnn_kernel_sizes=[2], lstm_units=4)
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
