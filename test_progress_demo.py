#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced Optuna progress indicators.

This script shows how the new progress feedback looks for hyperparameter optimization.
"""

import os
import sys

import numpy as np

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from trade_agent.training.train_cnn_lstm_enhanced import HyperparameterOptimizer


def main() -> None:
    """Run a quick optimization demo with progress indicators."""
    print("🧪 Testing Enhanced Optuna Progress Indicators")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    seq_length = 30
    n_features = 10

    sequences = np.random.randn(n_samples, seq_length, n_features)
    targets = np.sum(sequences[:, -5:, :3], axis=(1, 2)) + np.random.randn(n_samples) * 0.1

    print("Generated sample data:")
    print(f"  Sequences shape: {sequences.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Create optimizer with few trials for demo
    optimizer = HyperparameterOptimizer(
        sequences=sequences,
        targets=targets,
        n_trials=3,  # Small number for quick demo
        timeout=300  # 5 minute timeout
    )

    print("\n🚀 Starting optimization demo...")

    # Run optimization
    results = optimizer.optimize()

    print("\n🎯 Demo Results:")
    print(f"  Best score: {results['best_score']:.4f}")
    print(f"  Best params preview: {list(results['best_params'].keys())[:5]}...")
    print("=" * 60)
    print("✅ Demo completed! The progress indicators are now active.")

if __name__ == "__main__":
    main()
