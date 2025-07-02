import types
import src.optimization.model_utils as mu


def test_optimal_gpu_config_no_gpu(monkeypatch):
    monkeypatch.setattr(mu, "detect_gpus", lambda: {"available": False, "count": 0, "devices": []})
    cfg = mu.optimal_gpu_config(model_params=1000, batch_size=4)
    assert cfg["use_gpu"] is False
    assert "No GPUs available" in cfg["reason"]


def test_optimal_gpu_config_memory_constraint(monkeypatch):
    fake_info = {
        "available": True,
        "count": 1,
        "devices": [{"index": 0, "name": "GPU", "total_memory": 200, "memory_free": 50}],
    }
    monkeypatch.setattr(mu, "detect_gpus", lambda: fake_info)
    cfg = mu.optimal_gpu_config(model_params=10_000_000, batch_size=64, sequence_length=100, feature_dim=30)
    assert cfg["use_gpu"] is True
    assert cfg["recommended_batch_size"] <= 64
    assert cfg["mixed_precision"] or cfg["recommended_batch_size"] < 64
