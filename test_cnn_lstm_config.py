#!/usr/bin/env python3
"""Test CNNLSTMConfig to verify parameter names"""

import sys
import os
sys.path.insert(0, 'src')

def test_cnn_lstm_config():
    print("Testing CNNLSTMConfig parameter compatibility...")
    
    try:
        from src.models.cnn_lstm import CNNLSTMConfig, CNNLSTMModel
        
        # Test with correct parameter names
        config = CNNLSTMConfig(
            input_dim=10,
            lstm_units=32,  # Should work
            cnn_filters=[16, 32],
            cnn_kernel_sizes=[3, 3],
            dropout=0.1,
            output_size=3,  # Should work
            use_attention=False
        )
        print("✅ CNNLSTMConfig created successfully with correct parameters")
        
        # Test model creation
        model = CNNLSTMModel(
            input_dim=config.input_dim,
            output_size=config.output_size,
            config=config.to_dict(),
            use_attention=config.use_attention
        )
        print(f"✅ CNNLSTMModel created successfully with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        import torch
        x = torch.randn(4, 20, 10)  # batch_size=4, seq_len=20, features=10
        output = model(x)
        print(f"✅ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cnn_lstm_config()
    exit(0 if success else 1)
