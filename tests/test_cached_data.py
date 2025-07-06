from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("talib")

from src.data.pipeline import load_cached_csvs


def test_load_cached_csvs(tmp_path):
    df1 = pd.DataFrame(
        {"open": [1], "high": [2], "low": [1], "close": [2], "volume": [10]}
    )
    df2 = pd.DataFrame(
        {"open": [3], "high": [4], "low": [3], "close": [4], "volume": [20]}
    )

    file1 = tmp_path / "coinbase_BTC.csv"
    file2 = tmp_path / "oanda_EURUSD.csv"
    df1.to_csv(file1)
    df2.to_csv(file2)

    combined = load_cached_csvs(tmp_path)

    assert len(combined) == 2
    assert set(combined["source"]) == {"coinbase_BTC", "oanda_EURUSD"}
    assert combined.iloc[0]["open"] == 1
    assert combined.iloc[1]["open"] == 3


def test_load_cached_csvs_missing(tmp_path):
    missing_dir = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        load_cached_csvs(missing_dir)
