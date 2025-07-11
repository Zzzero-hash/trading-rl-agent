"""
CLI interface for CNN+LSTM model training.

This module provides a command-line interface for training CNN+LSTM models
with comprehensive configuration and monitoring capabilities.
"""

import argparse
import logging
import sys
from pathlib import Path

from trading_rl_agent.training.cnn_lstm_trainer import CNNLSTMTrainingManager


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train CNN+LSTM models for trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python -m trading_rl_agent.training.cli train --config configs/cnn_lstm_training.yaml

  # Train with custom output directory
  python -m trading_rl_agent.training.cli train --config configs/cnn_lstm_training.yaml --output-dir models/my_model

  # Evaluate a trained model
  python -m trading_rl_agent.training.cli evaluate --model-path models/best_model.pth --config configs/cnn_lstm_training.yaml

  # Predict with a trained model
  python -m trading_rl_agent.training.cli predict --model-path models/best_model.pth --data-path data/test_data.npy
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a CNN+LSTM model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    train_parser.add_argument("--output-dir", type=str, default="models", help="Output directory for trained models")
    train_parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild dataset even if it exists")
    train_parser.add_argument("--checkpoint-path", type=str, help="Path to save model checkpoints")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--model-path", type=str, required=True, help="Path to trained model file")
    eval_parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    eval_parser.add_argument("--output-file", type=str, help="Path to save evaluation results")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions with a trained model")
    predict_parser.add_argument("--model-path", type=str, required=True, help="Path to trained model file")
    predict_parser.add_argument("--data-path", type=str, required=True, help="Path to input data file (.npy)")
    predict_parser.add_argument("--config", type=str, required=True, help="Path to configuration YAML file")
    predict_parser.add_argument("--output-path", type=str, help="Path to save predictions")

    # Common arguments
    for subparser in [train_parser, eval_parser, predict_parser]:
        subparser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
        subparser.add_argument(
            "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level"
        )

    return parser


def setup_logging(log_level: str, verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())

    if verbose:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def train_model(args: argparse.Namespace) -> None:
    """Train a CNN+LSTM model."""
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting CNN+LSTM model training...")

    # Validate inputs
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training manager
    trainer = CNNLSTMTrainingManager(str(config_path))

    # Prepare dataset
    sequences, targets, dataset_info = trainer.prepare_dataset(force_rebuild=args.force_rebuild)

    # Create model
    input_dim = sequences.shape[-1]
    trainer.model = trainer.create_model(input_dim)

    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(sequences, targets)

    # Setup checkpoint path
    checkpoint_path = None
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = output_dir / "best_model_checkpoint.pth"

    # Train model
    training_results = trainer.train(train_loader, val_loader, save_path=str(checkpoint_path))

    # Evaluate on test set
    logger.info("🧪 Evaluating model on test set...")
    test_metrics = trainer.evaluate(test_loader)

    # Save final model
    final_model_path = output_dir / "final_model.pth"
    trainer.save_model(str(final_model_path))

    # Save training results
    results = {
        "training_results": training_results,
        "test_metrics": test_metrics,
        "dataset_info": dataset_info,
        "config": trainer.config,
    }

    results_path = output_dir / "training_results.json"
    import json

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("✅ Training completed successfully!")
    logger.info(f"📁 Model saved to: {final_model_path}")
    logger.info(f"📊 Results saved to: {results_path}")
    logger.info(f"📈 Best validation loss: {training_results['best_val_loss']:.6f}")
    logger.info(f"🧪 Test MAE: {test_metrics['mae']:.6f}")
    logger.info(f"🧪 Test RMSE: {test_metrics['rmse']:.6f}")


def evaluate_model(args: argparse.Namespace) -> None:
    """Evaluate a trained CNN+LSTM model."""
    logger = logging.getLogger(__name__)

    logger.info("🧪 Evaluating CNN+LSTM model...")

    # Validate inputs
    model_path = Path(args.model_path)
    config_path = Path(args.config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Initialize training manager
    trainer = CNNLSTMTrainingManager(str(config_path))

    # Prepare dataset
    sequences, targets, dataset_info = trainer.prepare_dataset()

    # Create data loaders
    train_loader, val_loader, test_loader = trainer.create_data_loaders(sequences, targets)

    # Load model
    input_dim = sequences.shape[-1]
    trainer.load_model(str(model_path), input_dim)

    # Evaluate model
    metrics = trainer.evaluate(test_loader)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path("evaluation_results.json")

    results = {"model_path": str(model_path), "metrics": metrics, "dataset_info": dataset_info}

    import json

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("✅ Evaluation completed!")
    logger.info(f"📊 Results saved to: {output_path}")

    # Print key metrics
    print("\n📊 Evaluation Results:")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  R² Score: {metrics['r2_score']:.6f}")
    print(f"  Correlation: {metrics['correlation']:.6f}")

    if "sharpe_ratio" in metrics:
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.6f}")
    if "max_drawdown" in metrics:
        print(f"  Max Drawdown: {metrics['max_drawdown']:.6f}")


def predict_model(args: argparse.Namespace) -> None:
    """Make predictions with a trained CNN+LSTM model."""
    logger = logging.getLogger(__name__)

    logger.info("🔮 Making predictions with CNN+LSTM model...")

    # Validate inputs
    model_path = Path(args.model_path)
    data_path = Path(args.data_path)
    config_path = Path(args.config)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load data
    import numpy as np

    data = np.load(data_path)
    logger.info(f"📊 Loaded data shape: {data.shape}")

    # Initialize training manager
    trainer = CNNLSTMTrainingManager(str(config_path))

    # Load model
    input_dim = data.shape[-1]
    trainer.load_model(str(model_path), input_dim)

    # Make predictions
    predictions = trainer.predict(data)

    # Save predictions
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = Path("predictions.npy")

    np.save(output_path, predictions)

    logger.info("✅ Predictions completed!")
    logger.info(f"📊 Predictions shape: {predictions.shape}")
    logger.info(f"💾 Predictions saved to: {output_path}")

    # Print summary statistics
    print("\n📊 Prediction Summary:")
    print(f"  Mean: {np.mean(predictions):.6f}")
    print(f"  Std: {np.std(predictions):.6f}")
    print(f"  Min: {np.min(predictions):.6f}")
    print(f"  Max: {np.max(predictions):.6f}")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(args.log_level, args.verbose)

    try:
        if args.command == "train":
            train_model(args)
        elif args.command == "evaluate":
            evaluate_model(args)
        elif args.command == "predict":
            predict_model(args)
        else:
            parser.print_help()

    except Exception as e:
        logging.exception(f"❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
