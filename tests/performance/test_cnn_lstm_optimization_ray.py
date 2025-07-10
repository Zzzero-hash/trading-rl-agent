import json

import numpy as np
import pandas as pd
import pytest

from trading_rl_agent.optimization import cnn_lstm_optimization as mod


def test_ray_tune_optimization_unavailable(monkeypatch):
    # When Ray unavailable, should fallback to simple_grid_search
    monkeypatch.setattr(mod, "RAY_AVAILABLE", False)
    features = np.random.randn(10, 3)
    targets = np.random.randn(10)
    result = mod.ray_tune_optimization(
        features,
        targets,
        num_samples=2,
        max_epochs_per_trial=1,
    )
    assert isinstance(result, dict)
    assert result.get("method") == "simple_grid_search"
    assert "best_config" in result


def test_ray_tune_optimization_available(monkeypatch):
    # Simulate Ray available and tune.run
    monkeypatch.setattr(mod, "RAY_AVAILABLE", True)
    # Ensure ray is initialized
    monkeypatch.setattr(mod.ray, "is_initialized", lambda: True)

    # Create fake analysis
    class FakeTrial:
        last_result = {"val_loss": 0.123}

    class FakeAnalysis:
        def get_best_config(self, metric, mode):
            return {"param": "value"}

        def get_best_trial(self, metric, mode):
            return FakeTrial()

        def dataframe(self):
            return pd.DataFrame([{"param": "value"}])

    # Mock tune.run
    monkeypatch.setattr(mod.tune, "run", lambda *args, **kwargs: FakeAnalysis())

    features = np.random.randn(12, 4)
    targets = np.random.randn(12)
    custom_space = {"custom_param": 42}
    resources = {"cpu": 1, "gpu": 0}
    result = mod.ray_tune_optimization(
        features,
        targets,
        num_samples=3,
        max_epochs_per_trial=2,
        custom_search_space=custom_space,
        ray_resources_per_trial=resources,
    )
    assert result["method"] == "ray_tune"
    assert result["best_config"] == {"param": "value"}
    assert pytest.approx(result["best_score"], 0.123) == 0.123
    assert isinstance(result["all_results"], list)
    assert result["all_results"][0]["param"] == "value"


def test_train_single_trial_exception(monkeypatch):
    # Simulate error in dataset creation
    monkeypatch.setattr(
        mod,
        "create_simple_dataset",
        lambda f, t, seq: (_ for _ in ()).throw(ValueError("bad")),
    )
    features = np.random.randn(5, 2)
    targets = np.random.randn(5)
    config = {"sequence_length": 3}
    metrics = mod.train_single_trial(config, features, targets, max_epochs=1)
    assert metrics["val_loss"] == float("inf")
    assert "error" in metrics
    assert "bad" in metrics["error"]


def test_optimize_cnn_lstm_streamlined_ray_branch(tmp_path, monkeypatch):
    # Test branch using ray_tune_optimization
    monkeypatch.setattr(mod, "RAY_AVAILABLE", True)
    # Monkeypatch ray_tune_optimization
    fake = {
        "best_score": 0.5,
        "best_config": {"x": 1},
        "all_results": [],
        "method": "fake",
    }
    monkeypatch.setattr(mod, "ray_tune_optimization", lambda f, t, num, ep: fake)

    features = np.random.randn(20, 5)
    targets = np.random.randn(20)
    out_dir = tmp_path / "ray_out"
    # Use num_samples>10 to pick Ray branch
    results = mod.optimize_cnn_lstm_streamlined(
        features,
        targets,
        num_samples=11,
        max_epochs_per_trial=1,
        output_dir=str(out_dir),
    )
    assert results == fake
    json_path = out_dir / "optimization_results.json"
    assert json_path.exists()
    # Check JSON content
    data = json.loads(json_path.read_text())
    assert data["best_score"] == 0.5
