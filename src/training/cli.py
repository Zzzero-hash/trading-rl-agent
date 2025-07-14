import argparse

from trading_rl_agent.agents.tune import run_tune

from . import cnn_lstm, rl


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Unified training entry point")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rl_parser = subparsers.add_parser("rl", help="Train reinforcement learning agent")
    rl_parser.add_argument("--data", required=True)
    rl_parser.add_argument("--model-path", required=True)
    rl_parser.add_argument("--num-workers", type=int, default=0)
    rl_parser.add_argument("--num-gpus", type=int, default=0)
    rl_parser.add_argument("--cluster-config")
    rl_parser.add_argument("--local-mode", action="store_true")

    cnn_parser = subparsers.add_parser("cnn-lstm", help="Train CNN-LSTM model")
    cnn_parser.add_argument("config", help="Path to training YAML config")

    tune_parser = subparsers.add_parser("tune", help="Run Ray Tune sweep")
    tune_parser.add_argument(
        "configs",
        nargs="+",
        help="One or more YAML files defining search space",
    )

    args = parser.parse_args(argv)

    if args.command == "rl":
        rl_args = [
            "--data",
            args.data,
            "--model-path",
            args.model_path,
            "--num-workers",
            str(args.num_workers),
            "--num-gpus",
            str(args.num_gpus),
        ]
        if args.cluster_config:
            rl_args += ["--cluster-config", args.cluster_config]
        if args.local_mode:
            rl_args.append("--local-mode")
        rl.main(rl_args)  # type: ignore[call-arg]
    elif args.command == "cnn-lstm":
        trainer = cnn_lstm.CNNLSTMTrainer()
        trainer.train_from_config(args.config)
    elif args.command == "tune":
        run_tune(args.configs)


if __name__ == "__main__":
    main()
