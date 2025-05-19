import pandas as pd
import yaml
import pytest
from trading_rl_agent.data.pipeline import run_pipeline

@pytest.fixture(autouse=True)
def dummy_fetch(monkeypatch):
    """
    Monkey-patch fetch_historical_data to return a dummy DataFrame for any symbol.
    """
    dummy_df = pd.DataFrame(
        {
            "open": [1],
            "high": [2],
            "low": [3],
            "close": [4],
            "volume": [5],
        },
        index=[pd.Timestamp("2021-01-01")],
    )
    monkeypatch.setattr(
        "trading_rl_agent.data.pipeline.fetch_historical_data", lambda symbol, start, end, timestep: dummy_df
    )
    return dummy_df


def test_run_pipeline(tmp_path, dummy_fetch):
    # Create a temporary pipeline config
    cfg = {
        "start": "2021-01-01",
        "end": "2021-01-02",
        "timestep": "day",
        "coinbase_perp_symbols": ["SYM"],
        "oanda_fx_symbols": ["SYM"],
        "to_csv": False,
        "output_dir": str(tmp_path / "raw"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    results = run_pipeline(str(cfg_path))

    # Check that both coinbase and oanda keys are present
    assert "coinbase_SYM" in results
    assert "oanda_SYM" in results

    # Verify the returned DataFrames match the dummy
    assert results["coinbase_SYM"].equals(dummy_fetch)
    assert results["oanda_SYM"].equals(dummy_fetch)

    # The output directory should exist but contain no CSV files when to_csv=False
    raw_dir = tmp_path / "raw"
    assert raw_dir.exists()
    assert not any(raw_dir.iterdir())