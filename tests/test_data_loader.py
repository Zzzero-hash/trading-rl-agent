import pandas as pd
import numpy as np

from src.data_pipeline import load_data


def test_load_csv_parses_columns_and_dtypes(tmp_path):
    data = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=3, freq="D"),
        "open": np.arange(3, dtype=float),
        "high": np.arange(3, dtype=float) + 1,
        "low": np.arange(3, dtype=float) - 1,
        "close": np.arange(3, dtype=float) + 2,
        "volume": np.arange(3, dtype=int),
    })
    csv = tmp_path / "data.csv"
    data.to_csv(csv, index=False)

    loaded = load_data({"type": "csv", "path": str(csv)})

    assert list(loaded.columns) == list(data.columns)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        assert pd.api.types.is_numeric_dtype(loaded[col])

