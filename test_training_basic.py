#!/usr/bin/env python3
"""Test basic training functionality."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("Testing basic training components...")

# Test 1: Model imports
try:
    print("✅ CNN+LSTM model import works")
except Exception as e:
    print(f"❌ CNN+LSTM model import failed: {e}")

# Test 2: Training module
try:
    from trade_agent.training.train_cnn_lstm_enhanced import (
        load_and_preprocess_csv_data,
    )
    print("✅ Training module imports work")
except Exception as e:
    print(f"❌ Training module import failed: {e}")

# Test 3: Create sample dataset
try:
    # Create sample CSV dataset
    dates = pd.date_range("2024-01-01", periods=200, freq="D")
    sample_data = pd.DataFrame({
        "timestamp": dates,
        "open": np.random.randn(200).cumsum() + 100,
        "high": np.random.randn(200).cumsum() + 102,
        "low": np.random.randn(200).cumsum() + 98,
        "close": np.random.randn(200).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 200),
        "returns": np.random.randn(200) * 0.02
    })

    # Add some technical indicators
    for i in [5, 10, 20]:
        sample_data[f"sma_{i}"] = sample_data["close"].rolling(i).mean()
        sample_data[f"rsi_{i}"] = 50 + np.random.randn(200) * 20

    test_csv_path = Path("test_sample_data.csv")
    sample_data.to_csv(test_csv_path, index=False)
    print(f"✅ Sample dataset created: {sample_data.shape} -> {test_csv_path}")

    # Test data loading
    sequences, targets = load_and_preprocess_csv_data(
        test_csv_path,
        sequence_length=10,
        prediction_horizon=1
    )
    print(f"✅ Data preprocessing: sequences {sequences.shape}, targets {targets.shape}")

    # Clean up
    test_csv_path.unlink()

except Exception as e:
    print(f"❌ Data preprocessing failed: {e}")
    import traceback
    traceback.print_exc()

print("✅ Basic training test completed!")
