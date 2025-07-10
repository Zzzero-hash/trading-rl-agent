import json

import numpy as np

from trading_rl_agent.optimization import cnn_lstm_optimization as mod
from trading_rl_agent.optimization.cnn_lstm_optimization import (
    create_simple_dataset,
    get_default_search_space,
    optimize_cnn_lstm_streamlined,
    simple_grid_search,
    train_single_trial,
)


def test_get_default_search_space_ray_unavailable(monkeypatch):
    # Simulate Ray unavailable
    monkeypatch.setattr(mod, "RAY_AVAILABLE", False)
    space = get_default_search_space()
    # Should be lists
    assert isinstance(space, dict)
    assert all(isinstance(v, list) for v in space.values())
    assert "learning_rate" in space


def test_get_default_search_space_ray_available(monkeypatch):
    # Simulate Ray available
    monkeypatch.setattr(mod, "RAY_AVAILABLE", True)
    # Reload module to ensure globals updated
    import importlib

    importlib.reload(mod)
    space = mod.get_default_search_space()
    # Should contain Ray Tune samplers
    assert isinstance(space, dict)
    val = space.get("learning_rate")
    # Ray Tune sampler classes have sample method
    assert hasattr(val, "sample") or callable(val)


def test_create_simple_dataset_2d_and_padding():
    # Test 2D features
    features = np.arange(12).reshape(6, 2)
    targets = np.arange(6)
    ds = create_simple_dataset(features, targets, sequence_length=4)
    x0, y0 = ds[0]
    assert x0.shape == (4, 1)
    assert y0.shape == (1,)
    # Test 3D-like input (reshaped)
    feats = np.arange(8).reshape(2, 2, 2)
    tgts = np.array([0, 1])
    ds2 = create_simple_dataset(feats, tgts, sequence_length=3)
    x1, y1 = ds2[1]
    assert x1.shape == (3, 1)


def test_train_single_trial_basic():
    # Basic training run
    np.random.seed(0)
    features = np.random.randn(20, 4)
    targets = np.random.randn(20)
    config = {
        "learning_rate": 0.005,
        "batch_size": 4,
        "lstm_units": 8,
        "dropout": 0.2,
        "sequence_length": 5,
    }
    metrics = train_single_trial(config, features, targets, max_epochs=3)
    assert isinstance(metrics, dict)
    assert "val_loss" in metrics
    assert metrics["val_loss"] >= 0
    assert "train_loss" in metrics
    assert "epochs_trained" in metrics


def test_simple_grid_search_structure():
    features = np.random.randn(30, 2)
    targets = np.arange(30)
    results = simple_grid_search(
        features,
        targets,
        num_samples=2,
        max_epochs_per_trial=1,
    )
    assert isinstance(results, dict)
    assert results.get("method") == "simple_grid_search"
    assert "best_config" in results
    assert isinstance(results["all_results"], list)
    assert len(results["all_results"]) == 2


def test_optimize_cnn_lstm_streamlined_saves_json(tmp_path, monkeypatch):
    # Force simple grid search
    monkeypatch.setattr(mod, "RAY_AVAILABLE", False)
    # Prepare data
    features = np.random.randn(15, 3)
    targets = np.random.randn(15)
    out_dir = tmp_path / "opt_out"
    # Run optimization
    results = optimize_cnn_lstm_streamlined(
        features,
        targets,
        num_samples=1,
        max_epochs_per_trial=1,
        output_dir=str(out_dir),
    )
    assert "best_score" in results
    # Check JSON file
    json_path = out_dir / "optimization_results.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["method"] == "simple_grid_search" or "best_score" in data
