"""
Test data generation utilities for integration tests.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def generate_synthetic_test_data(path=None, days=30, num_assets=3):
    if path is None:
        path = Path("data/synthetic_test_data.csv")
    else:
        path = Path(path)
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    data = {
        'timestamp': [d.strftime('%Y-%m-%d') for d in dates for _ in range(num_assets)],
        'symbol': [f"SYM{i}" for i in range(num_assets)] * days,
        'open': np.random.uniform(100, 200, days * num_assets),
        'high': np.random.uniform(100, 200, days * num_assets),
        'low': np.random.uniform(90, 199, days * num_assets),
        'close': np.random.uniform(100, 200, days * num_assets),
        'volume': np.random.randint(1000, 10000, days * num_assets),
        'label': np.random.choice([0, 1, 2], days * num_assets)
    }
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)

def cleanup_synthetic_test_data(path="data/synthetic_test_data.csv"):
    p = Path(path)
    if p.exists():
        p.unlink()
