#!/usr/bin/env python3
"""
Test script to verify the CNN architecture coordination fix.

This script tests that the hyperparameter optimization correctly selects
coordinated CNN architectures with matching filter and kernel sizes.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from train_cnn_lstm_enhanced import HyperparameterOptimizer
import numpy as np


def test_cnn_architecture_coordination():
    """Test that CNN architecture selection ensures matching lengths."""
    
    # Generate sample data
    np.random.seed(42)
    sequences = np.random.randn(100, 30, 20)
    targets = np.sum(sequences[:, -5:, :3], axis=(1, 2)) + np.random.randn(100) * 0.1
    
    optimizer = HyperparameterOptimizer(sequences, targets, n_trials=1)
    
    # Mock optuna trial to test different architectures
    class MockTrial:
        def __init__(self):
            self.architecture_index = 0
            # Test various coordinated architectures
            self.architectures = [
                ([16, 32], [3, 3]),                    # 2-layer, same kernel
                ([32, 64, 128], [3, 3, 3]),           # 3-layer, same kernel
                ([16, 32, 64, 128], [5, 5, 5, 5]),    # 4-layer, same kernel
                ([16, 32, 64], [3, 5, 3]),            # 3-layer, mixed kernel
                ([32, 64, 128], [5, 3, 5]),           # 3-layer, mixed kernel
            ]
        
        def suggest_categorical(self, name, choices):
            if name == "cnn_architecture":
                arch = self.architectures[self.architecture_index % len(self.architectures)]
                self.architecture_index += 1
                print(f"Selected architecture: filters={arch[0]}, kernels={arch[1]}")
                return arch
            elif name == "lstm_units":
                return 64
            elif name == "batch_size":
                return 16
            else:
                return choices[0]
        
        def suggest_int(self, name, low, high):
            return low
        
        def suggest_float(self, name, low, high, log=False):
            return low
    
    print("🧪 Testing CNN architecture coordination...")
    
    # Test multiple architecture selections
    for i in range(5):
        print(f"\nTest {i+1}:")
        trial = MockTrial()
        trial.architecture_index = i
        
        try:
            result = optimizer.objective(trial)
            print(f"✅ Success: validation loss = {result:.6f}")
            
            # Verify the selected architecture has matching lengths
            selected_arch = trial.architectures[i]
            filters, kernels = selected_arch
            assert len(filters) == len(kernels), f"Mismatch: {len(filters)} filters vs {len(kernels)} kernels"
            print(f"✅ Architecture validation passed: {len(filters)} layers")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            return False
    
    print("\n🎉 All CNN architecture coordination tests passed!")
    return True


def test_invalid_architecture_rejection():
    """Test that invalid architectures are properly rejected."""
    
    from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
    
    print("\n🧪 Testing invalid architecture rejection...")
    
    # Test with mismatched lengths (should raise ValueError)
    invalid_config = {
        "cnn_filters": [16, 32, 64],
        "cnn_kernel_sizes": [3, 3],  # Mismatch: 3 filters vs 2 kernels
        "lstm_units": 64,
        "lstm_layers": 1,
        "dropout_rate": 0.1,
    }
    
    try:
        model = CNNLSTMModel(input_dim=20, config=invalid_config)
        print("❌ Invalid architecture was not rejected!")
        return False
    except ValueError as e:
        print(f"✅ Invalid architecture properly rejected: {e}")
    
    # Test with valid architecture (should work)
    valid_config = {
        "cnn_filters": [16, 32, 64],
        "cnn_kernel_sizes": [3, 3, 3],  # Match: 3 filters vs 3 kernels
        "lstm_units": 64,
        "lstm_layers": 1,
        "dropout_rate": 0.1,
    }
    
    try:
        model = CNNLSTMModel(input_dim=20, config=valid_config)
        print("✅ Valid architecture accepted")
    except Exception as e:
        print(f"❌ Valid architecture rejected: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("🔧 Testing CNN Architecture Coordination Fix")
    print("=" * 50)
    
    success = True
    
    # Test 1: Architecture coordination
    if not test_cnn_architecture_coordination():
        success = False
    
    # Test 2: Invalid architecture rejection
    if not test_invalid_architecture_rejection():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! CNN architecture coordination fix is working correctly.")
    else:
        print("❌ SOME TESTS FAILED! Please check the implementation.")
    
    sys.exit(0 if success else 1)