#!/usr/bin/env python3
"""Test script to verify CUDA fix for classification targets."""

import sys
import os
sys.path.append('.')

from tests.test_train_cnn_lstm import TestCNNLSTMTrainer
import torch

def test_cuda_fix():
    """Test if the CUDA error is fixed."""
    print("Testing CUDA fix for classification targets...")
    
    # Initialize test
    test_instance = TestCNNLSTMTrainer()
    test_instance.setUp()
    
    try:
        # Test train_epoch
        print("Running test_train_epoch...")
        test_instance.test_train_epoch()
        print("✅ test_train_epoch PASSED")
        
        # Test validate
        print("Running test_validate...")
        test_instance.test_validate()
        print("✅ test_validate PASSED")
        
        # Test integration
        print("Running test_full_training_pipeline...")
        from tests.test_train_cnn_lstm import TestTrainingIntegration
        integration_test = TestTrainingIntegration()
        integration_test.test_full_training_pipeline()
        print("✅ test_full_training_pipeline PASSED")
        
        print("\n🎉 ALL TESTS PASSED - CUDA errors fixed!")
        
    except Exception as e:
        print(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    test_cuda_fix()
