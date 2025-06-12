import argparse
from .train_cnn_lstm import CNNLSTMTrainer
from .train_rl import train_from_config as rl_train_from_config
from src.agents.tune import run_tune


def main() -> None:
    """Entry point for training utilities."""
    parser = argparse.ArgumentParser(description="Unified training CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    cnn = sub.add_parser("cnn-lstm", help="Train CNN-LSTM model")
    cnn.add_argument("config", help="Path to YAML config file")
    cnn.add_argument("--tune", action="store_true", help="Run Ray Tune search")

    rl = sub.add_parser("rl", help="Train RL agent")
    rl.add_argument("config", help="Path to YAML config file")
    rl.add_argument("--tune", action="store_true", help="Run Ray Tune search")

    args = parser.parse_args()

    if args.command == "cnn-lstm":
        if args.tune:
            run_tune(args.config)
        else:
            trainer = CNNLSTMTrainer()
            trainer.train_from_config(args.config)
    else:
        if args.tune:
            run_tune(args.config)
        else:
            rl_train_from_config(args.config)


if __name__ == "__main__":
    main()

