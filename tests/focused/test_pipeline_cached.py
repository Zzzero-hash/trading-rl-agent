import pandas as pd
import pytest

pytest.importorskip("talib")

from src.data.pipeline import load_cached_csvs


def test_load_cached_csvs_combines_files(tmp_path):
    df1 = pd.DataFrame({"open": [1, 2], "close": [3, 4]})
    df2 = pd.DataFrame({"open": [5], "close": [6]})
    (tmp_path / "a.csv").write_text(df1.to_csv())
    (tmp_path / "b.csv").write_text(df2.to_csv())

    combined = load_cached_csvs(str(tmp_path))

    assert len(combined) == len(df1) + len(df2)
    assert set(combined["source"]) == {"a", "b"}
    pd.testing.assert_series_equal(
        combined.loc[0, ["open", "close"]].astype(float), df1.loc[0].astype(float)
    )


def test_load_cached_csvs_missing_dir(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_cached_csvs(str(missing))
