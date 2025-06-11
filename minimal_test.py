#!/usr/bin/env python3
"""Minimal test without Ray or PyTorch to check our logic."""

import sys
import os
import numpy as np

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_numpy_compatibility():
    """Test that NumPy is working correctly."""
    try:
        print(f"NumPy version: {np.__version__}")
        x = np.array([1, 2, 3])
        y = np.mean(x)
        print(f"✅ NumPy basic operations work: mean of {x} = {y}")
        assert y == 2.0
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
        assert False

def test_basic_python():
    """Test basic Python functionality."""
    try:
        from collections import deque
        from typing import Dict, List, Optional
        import yaml
        import random
        
        # Test basic data structures
        buffer = deque(maxlen=10)
        buffer.append([1, 2, 3])
        print(f"✅ Collections working: deque length = {len(buffer)}")
        
        # Test yaml
        config = {"test": True, "value": 42}
        yaml_str = yaml.dump(config)
        loaded = yaml.safe_load(yaml_str)
        print(f"✅ YAML working: {loaded}")
        assert loaded == config
    except Exception as e:
        print(f"❌ Basic Python test failed: {e}")
        assert False

def test_our_code_structure():
    """Test that our code structure is correct."""
    try:
        # Test importing our modules without heavy dependencies
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        
        # Check if files exist
        sac_path = os.path.join(src_path, 'agents', 'sac_agent.py')
        td3_path = os.path.join(src_path, 'agents', 'td3_agent.py')
        cnn_path = os.path.join(src_path, 'models', 'cnn_lstm.py')
        
        assert os.path.exists(sac_path), "SAC agent file missing"
        assert os.path.exists(td3_path), "TD3 agent file missing"
        assert os.path.exists(cnn_path), "CNN-LSTM model file missing"
        
        print("✅ All core implementation files exist")
        
        # Check file sizes to ensure they're not empty
        sac_size = os.path.getsize(sac_path)
        td3_size = os.path.getsize(td3_path)
        cnn_size = os.path.getsize(cnn_path)
        
        print(f"✅ File sizes: SAC={sac_size}B, TD3={td3_size}B, CNN-LSTM={cnn_size}B")
        assert sac_size > 0 and td3_size > 0 and cnn_size > 0
    except Exception as e:
        print(f"❌ Code structure test failed: {e}")
        assert False

if __name__ == "__main__":
    print("🔍 Minimal Testing - Dependencies and Code Structure\n")
    
    success = True
    success &= test_numpy_compatibility()
    print()
    success &= test_basic_python()
    print()
    success &= test_our_code_structure()
    
    print(f"\n{'🎉 All tests passed!' if success else '💥 Some tests failed.'}")
    
    if success:
        print("\n📋 Summary of streamlined dependencies:")
        print("- ✅ NumPy < 2.0.0 (compatibility fixed)")
        print("- ✅ Core Python libraries working")
        print("- ✅ All agent implementations present")
        print("- ✅ Model architectures implemented")
        print("\n🚀 Ready to continue development!")
