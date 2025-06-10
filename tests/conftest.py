import pytest
import pandas as pd
from generate_sample_data import generate_sample_price_data

@pytest.fixture(scope="session")
def sample_csv_path(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    file_path = data_dir / "sample.csv"
    df = generate_sample_price_data(symbol="TEST", days=30, start_price=100.0, volatility=0.01)
    df = df.drop(columns=["timestamp", "symbol"])
    df.to_csv(file_path, index=False)
    return str(file_path)
