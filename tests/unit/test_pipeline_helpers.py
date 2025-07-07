import pandas as pd
import pytest

from src.data.pipeline import load_cached_csvs


def test_load_cached_csvs_empty(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    df = load_cached_csvs(tmp_path)
    assert df.empty


def test_load_cached_csvs_sources(sample_csv_file, tmp_path):
    df = pd.read_csv(sample_csv_file)
    dest = tmp_path / "dataset.csv"
    df.to_csv(dest)
    combined = load_cached_csvs(tmp_path)
    assert "source" in combined.columns
    assert combined["source"].iloc[0] == "dataset"


def test_load_cached_csvs_missing_dir(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_cached_csvs(missing)
