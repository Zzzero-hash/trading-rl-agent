import ray
import types

from src.utils.cluster import init_ray, get_available_devices


def test_get_available_devices(monkeypatch):
    fake_resources = {"CPU": 8.0, "GPU": 2.0}

    def fake_available():
        return fake_resources

    monkeypatch.setattr(ray, "available_resources", fake_available)
    devices = get_available_devices()
    assert devices == {"CPU": 8.0, "GPU": 2.0}
