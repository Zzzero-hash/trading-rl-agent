import pandas as pd
import pytest
import ray
import yaml

from trading_rl_agent.data.pipeline import run_pipeline


@pytest.fixture(autouse=True)
def dummy_fetch(monkeypatch):
    """Monkey-patch all data fetchers to return a simple DataFrame."""
    dummy_df = pd.DataFrame(
        {
            "open": [1] * 30,
            "high": [2] * 30,
            "low": [3] * 30,
            "close": [4] * 30,
            "volume": [5] * 30,
        }
    )
    monkeypatch.setattr(
        "src.data.pipeline.fetch_historical_data", lambda *_, **__: dummy_df
    )
    monkeypatch.setattr(
        "src.data.pipeline.fetch_synthetic_data", lambda *_, **__: dummy_df
    )
    monkeypatch.setattr("src.data.pipeline.fetch_live_data", lambda *_, **__: dummy_df)
    return dummy_df


def test_run_pipeline(tmp_path, dummy_fetch):
    # Create a temporary pipeline config
    cfg = {
        "start": "2021-01-01",
        "end": "2021-01-02",
        "timestep": "minute",
        "coinbase_perp_symbols": ["SYM"],
        "oanda_fx_symbols": ["SYM"],
        "synthetic_symbols": ["SYN"],
        "live_symbols": ["LIV"],
        "to_csv": False,
        "output_dir": str(tmp_path / "raw"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    ray.init(local_mode=True, log_to_driver=False)
    results = run_pipeline(str(cfg_path))
    ray.shutdown()

    # Check that all expected keys are present
    assert "coinbase_SYM" in results
    assert "oanda_SYM" in results
    assert "synthetic_SYN" in results
    assert "live_LIV" in results

    # Verify the returned DataFrames match the dummy
    assert results["coinbase_SYM"].equals(dummy_fetch)
    assert results["oanda_SYM"].equals(dummy_fetch)

    # The output directory should exist but contain no CSV files when to_csv=False
    raw_dir = tmp_path / "raw"
    assert raw_dir.exists()
    assert not any(raw_dir.iterdir())
