#!/usr/bin/env python3
"""
Test script to validate the hyperparameter optimization fixes.

This script tests the new coordinated CNN architectures and error handling.
"""

import os
import sys

import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from trade_agent.training.train_cnn_lstm_enhanced import HyperparameterOptimizer


def test_single_optimization() -> bool:
    """Test a single optimization trial to validate fixes."""
    print("🧪 Testing Fixed Hyperparameter Optimization")
    print("=" * 60)

    # Generate sample data similar to the failing case
    np.random.seed(42)
    n_samples = 100
    seq_length = 77  # Match the input_dim from the error
    n_features = 77

    sequences = np.random.randn(n_samples, seq_length, n_features)
    targets = np.sum(sequences[:, -5:, :3], axis=(1, 2)) + np.random.randn(n_samples) * 0.1
    targets = targets.reshape(-1, 1)  # Ensure proper shape

    print("Generated test data:")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Create optimizer with just 1 trial for testing
    optimizer = HyperparameterOptimizer(
        sequences=sequences,
        targets=targets,
        n_trials=1,
        timeout=120  # 2 minute timeout
    )

    print("\n🚀 Starting single trial test...")

    try:
        # Run optimization
        results = optimizer.optimize()

        print("\n✅ Test Results:")
        print(f"  Best score: {results['best_score']:.4f}")
        print(f"  Trial completed: {'Success' if results['best_score'] != float('inf') else 'Failed'}")
        print(f"  Best params preview: {list(results['best_params'].keys())[:5]}...")

        if results["best_score"] != float("inf"):
            print("\n🎉 SUCCESS: The optimization fixes work correctly!")
            return True
        else:
            print("\n❌ ISSUE: Trial returned infinite loss")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(f"Error details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_single_optimization()
    if success:
        print("=" * 60)
        print("✅ All fixes validated successfully!")
    else:
        print("=" * 60)
        print("❌ Issues still remain - check the error details above")
