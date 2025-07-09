import pandas as pd
import ray
import yaml
import pytest
import types
import sys
from pathlib import Path

if "structlog" not in sys.modules:
    stub = types.SimpleNamespace(
        BoundLogger=object,
        stdlib=types.SimpleNamespace(
            ProcessorFormatter=object,
            BoundLogger=object,
            LoggerFactory=lambda: None,
            filter_by_level=lambda *a, **k: None,
            add_logger_name=lambda *a, **k: None,
            add_log_level=lambda *a, **k: None,
            PositionalArgumentsFormatter=lambda: None,
            wrap_for_formatter=lambda f: f,
        ),
        processors=types.SimpleNamespace(
            TimeStamper=lambda **_: None,
            StackInfoRenderer=lambda **_: None,
            format_exc_info=lambda **_: None,
            UnicodeDecoder=lambda **_: None,
        ),
        dev=types.SimpleNamespace(ConsoleRenderer=lambda **_: None),
        configure=lambda **_: None,
        get_logger=lambda name=None: None,
    )
    sys.modules["structlog"] = stub

base_path = Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent"
if "trading_rl_agent" not in sys.modules:
    pkg = types.ModuleType("trading_rl_agent")
    pkg.__path__ = [str(base_path)]
    sys.modules["trading_rl_agent"] = pkg
if "trading_rl_agent.data" not in sys.modules:
    data_pkg = types.ModuleType("trading_rl_agent.data")
    data_pkg.__path__ = [str(base_path / "data")]
    sys.modules["trading_rl_agent.data"] = data_pkg
    pkg.data = data_pkg

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "trading_rl_agent.data.pipeline",
    Path(__file__).resolve().parents[2] / "src" / "trading_rl_agent" / "data" / "pipeline.py",
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)  # type: ignore
sys.modules["trading_rl_agent.data.pipeline"] = pipeline
run_pipeline = pipeline.run_pipeline


@pytest.fixture(autouse=True)
def dummy_loaders(monkeypatch):
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01")],
        "open":      [1],
        "high":      [2],
        "low":       [3],
        "close":     [4],
        "volume":    [5],
    })
    monkeypatch.setattr(
        pipeline,
        "load_yfinance",
        lambda *args, **kwargs: df,
    )
    monkeypatch.setattr(
        pipeline,
        "load_alphavantage",
        lambda *args, **kwargs: df,
    )
    monkeypatch.setattr(
        pipeline,
        "load_ccxt",
        lambda *args, **kwargs: df,
    )
    return df


def test_pipeline_uses_new_loaders(tmp_path, dummy_loaders):
    cfg = {
        "start": "2024-01-01",
        "end": "2024-01-02",
        "yfinance_symbols": ["AAPL"],
        "alphavantage_symbols": ["IBM"],
        "ccxt": {"binance": ["BTC/USDT"]},
        "output_dir": str(tmp_path / "out"),
        "to_csv": False,
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    ray.init(local_mode=True, log_to_driver=False)
    results = ray.get(run_pipeline.remote(str(cfg_path)))
    ray.shutdown()

    assert "yfinance_AAPL" in results
    assert "alphavantage_IBM" in results
    assert "binance_BTCUSDT" in results
    for df in results.values():
        assert df.equals(dummy_loaders)
