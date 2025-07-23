#!/usr/bin/env python3
"""
Demonstration script for hyperparameter optimization of sequence_length and prediction_horizon.

This script shows how to use the hyperparameter optimization to find the optimal
lookback period and prediction horizon for the CNN+LSTM model.
"""

import argparse
from pathlib import Path

from trade_agent.data.optimized_dataset_builder import OptimizedDatasetBuilder, OptimizedDatasetConfig
from trade_agent.training.train_cnn_lstm_enhanced import (
    EnhancedCNNLSTMTrainer,
    HyperparameterOptimizer,
    create_enhanced_model_config,
    create_enhanced_training_config,
)


def main() -> None:
    """Run hyperparameter optimization for sequence_length and prediction_horizon."""
    parser = argparse.ArgumentParser(description="Optimize sequence_length and prediction_horizon")
    parser.add_argument("--data-path", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--output-dir", type=str, default="models/optimized", help="Output directory")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of optimization trials")
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL", help="Comma-separated symbols")
    parser.add_argument("--start-date", type=str, default="2022-01-01", help="Start date")
    parser.add_argument("--end-date", type=str, default="2023-01-01", help="End date")
    args = parser.parse_args()

    print(f"🚀 Starting hyperparameter optimization with {args.n_trials} trials")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset config
    symbols = args.symbols.split(",")
    dataset_config = OptimizedDatasetConfig(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=str(output_dir / "dataset"),
    )

    # Build dataset with default parameters
    print("📊 Building initial dataset...")
    sequences, targets, _ = OptimizedDatasetBuilder.load_or_build(dataset_config)

    # Run hyperparameter optimization
    print("🔍 Running hyperparameter optimization...")
    optimizer = HyperparameterOptimizer(sequences, targets, n_trials=args.n_trials)
    result = optimizer.optimize()

    # Extract best parameters
    best_params = result["best_params"]
    best_score = result["best_score"]
    print(f"✅ Best parameters found: {best_params}")
    print(f"📈 Best validation loss: {best_score:.6f}")

    # Extract sequence_length and prediction_horizon
    sequence_length = best_params.get("sequence_length", 60)
    prediction_horizon = best_params.get("prediction_horizon", 1)

    # Rebuild dataset with optimized parameters
    print(f"🔄 Rebuilding dataset with sequence_length={sequence_length}, prediction_horizon={prediction_horizon}")
    optimized_sequences, optimized_targets, _ = OptimizedDatasetBuilder.load_or_build(
        dataset_config,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )

    # Train final model with optimized parameters
    print("🏋️ Training final model with optimized parameters")
    model_config = create_enhanced_model_config(input_dim=optimized_sequences.shape[-1])
    training_config = create_enhanced_training_config(
        learning_rate=best_params.get("learning_rate", 0.001),
        batch_size=best_params.get("batch_size", 32),
        epochs=100,
    )

    # Add sequence parameters to training config
    training_config["sequence_length"] = sequence_length
    training_config["prediction_horizon"] = prediction_horizon

    # Create and run trainer
    trainer = EnhancedCNNLSTMTrainer(
        model_config=model_config,
        training_config=training_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create dataset config
    dataset_config = {
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
    }

    # Train model
    result = trainer.train_from_dataset(
        sequences=optimized_sequences,
        targets=optimized_targets,
        save_path=str(output_dir / "best_model.pth"),
        dataset_config=dataset_config,
    )

    print(f"✅ Training complete! Final validation loss: {result['best_val_loss']:.6f}")
    print(f"📊 Model saved to {output_dir / 'best_model.pth'}")
    print(f"📏 Used sequence_length={sequence_length}, prediction_horizon={prediction_horizon}")

    # Save optimization results
    with open(output_dir / "optimization_results.txt", "w") as f:
        f.write(f"Best parameters: {best_params}\n")
        f.write(f"Best validation loss: {best_score:.6f}\n")
        f.write(f"Sequence length: {sequence_length}\n")
        f.write(f"Prediction horizon: {prediction_horizon}\n")
        f.write(f"Final validation loss: {result['best_val_loss']:.6f}\n")
        f.write(f"Training time: {result['training_time']:.2f}s\n")

    print("✅ Done!")


if __name__ == "__main__":
    import torch
    main()
