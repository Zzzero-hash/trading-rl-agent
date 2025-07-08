#!/usr/bin/env python3
"""Test script for streamlined optimization."""

from pathlib import Path
import sys

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_optimization():
    """Test the streamlined optimization function."""
    print("🧪 Testing streamlined optimization...")

    try:
        from trading_rl_agent.optimization import optimize_cnn_lstm

        # Test with small data
        np.random.seed(42)
        features = np.random.randn(100, 10)
        targets = np.random.randn(100)

        print("   • Running optimization with 2 trials...")
        results = optimize_cnn_lstm(
            features=features, targets=targets, num_samples=2, max_epochs_per_trial=5
        )

        assert results.get("method") == "optuna"
        print("✅ Optimization test successful!")
        print(f"   • Best score: {results['best_score']:.4f}")
        print(f"   • Best config: {results['best_config']}")

        assert isinstance(results["best_score"], float)
        assert isinstance(results["best_config"], dict)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_optimization()
