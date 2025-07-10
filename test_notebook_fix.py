#!/usr/bin/env python3
"""
Test script to verify main.ipynb components work correctly.
This script tests the key functionality that the notebook uses.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append("/workspace/src")

def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import matplotlib.pyplot as plt
        import seaborn as sns
        import yfinance as yf
        print("✅ Basic libraries imported")
        
        # Test trading_rl_agent imports
        from trading_rl_agent.data.features import generate_features
        print("✅ generate_features imported")
        
        # Test CNN+LSTM model
        from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
        print("✅ CNNLSTMModel imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_generation():
    """Test data generation functionality."""
    print("\nTesting data generation...")
    
    try:
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 100 + np.random.randn(len(dates)).cumsum() + 1,
            'low': 100 + np.random.randn(len(dates)).cumsum() - 1,
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'symbol': 'AAPL',
            'data_source': 'stock_real',
            'asset_class': 'stock'
        })
        
        # Test feature generation
        from trading_rl_agent.data.features import generate_features
        featured_data = generate_features(data, ma_windows=[5, 10, 20], rsi_window=14)
        print(f"✅ Generated {featured_data.shape[1]} features")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation error: {e}")
        return False

def test_cnn_lstm_model():
    """Test CNN+LSTM model creation and forward pass."""
    print("\nTesting CNN+LSTM model...")
    
    try:
        import torch
        from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
        
        # Create model
        model = CNNLSTMModel(
            input_dim=10,
            output_size=1,
            sequence_length=60
        )
        
        # Test forward pass
        batch_size = 32
        seq_len = 60
        input_dim = 10
        
        x = torch.randn(batch_size, seq_len, input_dim)
        output = model(x)
        
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"✅ Forward pass works: input {x.shape} -> output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ CNN+LSTM model error: {e}")
        return False

def test_dataset_creation():
    """Test dataset creation and saving."""
    print("\nTesting dataset creation...")
    
    try:
        # Create sample dataset
        dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 100 + np.random.randn(len(dates)).cumsum() + 1,
            'low': 100 + np.random.randn(len(dates)).cumsum() - 1,
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates)),
            'symbol': 'AAPL',
            'data_source': 'stock_real',
            'asset_class': 'stock'
        })
        
        # Add features
        from trading_rl_agent.data.features import generate_features
        featured_data = generate_features(data, ma_windows=[5, 10, 20], rsi_window=14)
        
        # Save dataset
        output_dir = Path("data")
        output_dir.mkdir(exist_ok=True)
        
        sample_data_path = output_dir / "sample_data.csv"
        featured_data.to_csv(sample_data_path, index=False)
        
        print(f"✅ Dataset created and saved to {sample_data_path}")
        print(f"✅ Dataset shape: {featured_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing main.ipynb components...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_generation,
        test_cnn_lstm_model,
        test_dataset_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The notebook should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)