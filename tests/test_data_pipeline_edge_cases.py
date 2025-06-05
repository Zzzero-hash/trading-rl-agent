# tests/test_data_pipeline_edge_cases.py
import yaml
import pytest
import pandas as pd
from pathlib import Path
from src.data.pipeline import run_pipeline

@pytest.fixture(autouse=True)
def dummy_fetch(monkeypatch):
    """Monkey-patch all data fetch functions to return a dummy DataFrame."""
    dummy_df = pd.DataFrame(
        {
            "open": [10] * 30,
            "high": [15] * 30,
            "low": [5] * 30,
            "close": [12] * 30,
            "volume": [100] * 30,
        }
    )
    monkeypatch.setattr(
        "src.data.pipeline.fetch_historical_data",
        lambda symbol, start, end, timestep: dummy_df,
    )
    monkeypatch.setattr(
        "src.data.pipeline.fetch_synthetic_data",
        lambda symbol, start, end, timestep: dummy_df,
    )
    monkeypatch.setattr(
        "src.data.pipeline.fetch_live_data",
        lambda symbol, start, end, timestep: dummy_df,
    )
    return dummy_df


def test_no_symbols(tmp_path, dummy_fetch):
    # Config with no symbols and to_csv=False
    cfg = {
        "start": "2021-01-01",
        "end": "2021-01-02",
        "synthetic_symbols": [],
        "live_symbols": [],
        "to_csv": False,
        "output_dir": str(tmp_path / "raw"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    results = run_pipeline(str(cfg_path))
    assert results == {}, "Expected empty results when no symbols are configured"

    raw_dir = Path(cfg["output_dir"])
    assert raw_dir.exists(), "Output directory should be created"
    assert not any(raw_dir.iterdir()), "Directory should be empty when to_csv=False"


def test_to_csv_creates_files(tmp_path, dummy_fetch):
    # Config with one coinbase symbol and to_csv=True
    cfg = {
        "start": "2021-01-01",
        "end": "2021-01-02",
        "coinbase_perp_symbols": ["ABC"],
        "oanda_fx_symbols": [],
        "synthetic_symbols": ["SYN"],
        "live_symbols": ["LIV"],
        "to_csv": True,
        "output_dir": str(tmp_path / "out"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    results = run_pipeline(str(cfg_path))
    key = "coinbase_ABC"
    assert key in results, f"Results should contain key '{key}'"
    assert results[key].equals(dummy_fetch), "Returned DataFrame should match dummy"
    assert "synthetic_SYN" in results
    assert "live_LIV" in results

    out_dir = Path(cfg["output_dir"])
    csv_file = out_dir / f"{key}.csv"
    assert csv_file.exists(), "CSV file should be created when to_csv=True"

    # Read back and verify contents
    df_csv = pd.read_csv(csv_file, index_col=0)
    assert list(df_csv.columns) == ["open", "high", "low", "close", "volume"], "CSV columns mismatch"
    pd.testing.assert_frame_equal(df_csv, dummy_fetch)


def test_invalid_config_path_raises(tmp_path):
    # Non-existent file should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        run_pipeline(str(tmp_path / "nope.yaml"))
