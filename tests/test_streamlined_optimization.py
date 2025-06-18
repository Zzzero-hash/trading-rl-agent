#!/usr/bin/env python3
"""Test script for streamlined optimization."""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_optimization():
    """Test the streamlined optimization function."""
    print("🧪 Testing streamlined optimization...")
    
    try:
        from src.optimization import optimize_cnn_lstm
        
        # Test with small data
        np.random.seed(42)
        features = np.random.randn(100, 10) 
        targets = np.random.randn(100)
        
        print("   • Running optimization with 2 trials...")
        results = optimize_cnn_lstm(
            features=features, 
            targets=targets, 
            num_samples=2, 
            max_epochs_per_trial=5
        )
        
        print("✅ Optimization test successful!")
        print(f"   • Best score: {results['best_score']:.4f}")
        print(f"   • Best config: {results['best_config']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_optimization()
    exit(0 if success else 1)
