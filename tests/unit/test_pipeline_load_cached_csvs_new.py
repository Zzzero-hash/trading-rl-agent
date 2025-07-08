import pandas as pd
import pytest

from trading_rl_agent.data.pipeline import load_cached_csvs

pytestmark = pytest.mark.unit


def test_load_cached_csvs_multiple(sample_csv_file, tmp_path):
    df = pd.read_csv(sample_csv_file)
    sub = tmp_path / "data"
    sub.mkdir()
    path1 = sub / "a.csv"
    path2 = sub / "b.csv"
    df.to_csv(path1, index=False)
    df.to_csv(path2, index=False)
    combined = load_cached_csvs(sub)
    assert len(combined) == len(df) * 2
    assert set(combined["source"].unique()) == {"a", "b"}
