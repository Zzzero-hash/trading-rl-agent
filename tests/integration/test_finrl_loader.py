import yaml

from finrl_data_loader import load_synthetic_data


def test_load_synthetic_data(tmp_path):
    cfg = {
        "synthetic": {"n_days": 10, "mu": 0.0001, "sigma": 0.01, "n_symbols": 1},
        "tech_indicators": ["macd"],
        "output": str(tmp_path / "synth.csv"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    with Path(cfg_path).open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    df = load_synthetic_data(str(cfg_path))
    assert not df.empty
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    assert "macd" in df.columns
