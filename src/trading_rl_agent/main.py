import argparse


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the main entry point."""

    parser = argparse.ArgumentParser(
        description="Train or evaluate an RL trading agent",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to unified system config YAML",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        required=True,
        help="Path to environment config YAML",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--trainer-config",
        type=str,
        required=True,
        help="Path to trainer config YAML",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="outputs",
        help="Directory to save models and logs",
    )
    parser.add_argument("--train", action="store_true", help="Run training loop")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Ray Tune hyperparameter search",
    )
    return parser
