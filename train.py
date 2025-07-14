#!/usr/bin/env python3
"""
Trading RL Agent Training Script

This is the main entry point for training the Trading RL Agent.
It uses the optimized training pipeline with all major performance improvements.

USAGE:
    # Basic training with optimizations
    python train.py --epochs 100 --gpu

    # With hyperparameter optimization
    python train.py --optimize-hyperparams --n-trials 50 --epochs 30 --gpu

    # Forex-focused training
    python train.py --forex-focused --epochs 150 --gpu

    # Force rebuild dataset
    python train.py --force-rebuild --epochs 100 --gpu
"""

import sys
from pathlib import Path
from typing import Any


def setup_path() -> None:
    """Add the project root to the Python path."""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))


def main() -> dict[str, Any]:
    """Main entry point for the training script."""
    setup_path()
    from train_advanced import main as advanced_main

    return advanced_main()


if __name__ == "__main__":
    main()
