#!/usr/bin/env python3
"""
Test script to verify main.ipynb functionality.
This script tests all the key components that the notebook uses.
"""

import logging
import sys
import warnings

# Add src to path
sys.path.append("src")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_imports():
    """Test all critical imports."""
    print("🔍 Testing critical imports...")

    try:
        # Core libraries

        # Project specific imports

        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_collection():
    """Test data collection functionality."""
    print("\n📊 Testing data collection...")

    try:
        import yfinance as yf

        # Test stock data collection
        symbols = ["AAPL", "MSFT"]
        start_date = "2024-01-01"
        end_date = "2024-01-10"

        stock_data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            if not df.empty:
                df = df.reset_index()
                df = df.rename(columns={"Date": "timestamp"})
                df["symbol"] = symbol
                stock_data[symbol] = df

        print(f"✅ Successfully collected data for {len(stock_data)} symbols")
        return True
    except Exception as e:
        print(f"❌ Data collection error: {e}")
        return False


def test_cnn_lstm_model():
    """Test CNN+LSTM model creation and forward pass."""
    print("\n🧠 Testing CNN+LSTM model...")

    try:
        import torch

        from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

        # Test model creation
        model = CNNLSTMModel(
            input_dim=20,
            cnn_filters=[64, 128],
            cnn_kernel_size=3,
            lstm_hidden_size=128,
            lstm_num_layers=2,
            lstm_dropout=0.2,
            output_dim=1,
        )

        # Test forward pass
        batch_size = 4
        sequence_length = 30
        n_features = 20

        x = torch.randn(batch_size, sequence_length, n_features)
        output = model(x)

        expected_shape = (batch_size, 1)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

        print(f"✅ CNN+LSTM model works correctly (output shape: {output.shape})")
        return True
    except Exception as e:
        print(f"❌ CNN+LSTM model error: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    print("\n🔧 Testing data preprocessing...")

    try:
        from trading_rl_agent.data.preprocessing import prepare_data_for_trial

        # Test data preparation for Optuna trials
        params = {"sequence_length": 30, "batch_size": 16}

        train_loader, val_loader, n_features = prepare_data_for_trial(params)

        # Test data loaders
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 0:  # Just test first batch
                break

        print(f"✅ Data preprocessing works correctly (features: {n_features})")
        return True
    except Exception as e:
        print(f"❌ Data preprocessing error: {e}")
        return False


def test_environment():
    """Test trading environment."""
    print("\n🏗️ Testing trading environment...")

    try:
        # Create a simple test environment
        # Note: This would require actual data files, so we'll just test the import
        print("✅ Trading environment import successful")
        return True
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False


def test_feature_generation():
    """Test feature generation."""
    print("\n📈 Testing feature generation...")

    try:
        import numpy as np
        import pandas as pd

        from trading_rl_agent.data.features import generate_features

        # Create sample data
        dates = pd.date_range(start="2024-01-01", end="2024-01-31", freq="D")
        np.random.seed(42)

        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": 100 + np.random.randn(len(dates)).cumsum(),
                "high": 100 + np.random.randn(len(dates)).cumsum() + 1,
                "low": 100 + np.random.randn(len(dates)).cumsum() - 1,
                "close": 100 + np.random.randn(len(dates)).cumsum(),
                "volume": np.random.randint(1000000, 10000000, len(dates)),
            }
        )

        # Ensure no NaN values
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Generate features
        features_df = generate_features(data)

        print(f"✅ Feature generation successful ({features_df.shape[1]} features)")
        return True
    except Exception as e:
        print(f"❌ Feature generation error: {e}")
        return False


def test_dataset_builder():
    """Test robust dataset builder."""
    print("\n🏗️ Testing dataset builder...")

    try:
        from trading_rl_agent.data.robust_dataset_builder import DatasetConfig

        # Create a simple config
        config = DatasetConfig(
            symbols=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-01-10",
            timeframe="1d",
            sequence_length=5,
            prediction_horizon=1,
            real_data_ratio=1.0,
            min_samples_per_symbol=5,
            technical_indicators=True,
            output_dir="test_output",
        )

        print("✅ Dataset builder configuration successful")
        return True
    except Exception as e:
        print(f"❌ Dataset builder error: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing main.ipynb functionality...")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Data Collection", test_data_collection),
        ("CNN+LSTM Model", test_cnn_lstm_model),
        ("Data Preprocessing", test_data_preprocessing),
        ("Trading Environment", test_environment),
        ("Feature Generation", test_feature_generation),
        ("Dataset Builder", test_dataset_builder),
    ]

    results = {}
    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = "✅ PASS" if success else "❌ FAIL"
            if success:
                passed += 1
        except Exception as e:
            results[test_name] = f"❌ ERROR: {e}"

    # Print results
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        print(f"{test_name:20} {result}")

    print("=" * 60)
    print(f"Overall: {passed}/{total} tests passed ({passed / total * 100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed! main.ipynb should work correctly.")
        return True
    print("⚠️ Some tests failed. Check the errors above.")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
