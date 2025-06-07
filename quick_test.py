#!/usr/bin/env python3
"""Quick test script to verify our core implementations work."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports work without heavy dependencies."""
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
        
        # Test if our code can be imported without executing
        from agents.sac_agent import Actor, Critic, ReplayBuffer
        print("✅ SAC components imported successfully")
        
        from agents.td3_agent import Actor as TD3Actor
        print("✅ TD3 components imported successfully")
        
        from models.cnn_lstm import CNNLSTMModel
        print("✅ CNN-LSTM model imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_model_creation():
    """Test creating models with minimal dependencies."""
    try:
        import torch
        import numpy as np
        
        # Test CNN-LSTM
        from models.cnn_lstm import CNNLSTMModel
        model = CNNLSTMModel(input_dim=10, output_size=3)
        print("✅ CNN-LSTM model created successfully")
        
        # Test simple forward pass
        dummy_input = torch.randn(1, 5, 10)  # batch, seq, features
        output = model(dummy_input)
        print(f"✅ CNN-LSTM forward pass successful: {output.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Trading RL Agent Core Components\n")
    
    success = True
    success &= test_basic_imports()
    print()
    success &= test_model_creation()
    
    if success:
        print("\n🎉 All core tests passed! Dependencies are working correctly.")
    else:
        print("\n💥 Some tests failed. Check dependencies.")
    
    sys.exit(0 if success else 1)
