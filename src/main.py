import argparse

from trading_rl_agent.agents.trainer import Trainer
from trading_rl_agent.agents.tune import run_tune
from trading_rl_agent.core.config import ConfigManager


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for the main entry point."""

    parser = argparse.ArgumentParser(
        description="Train or evaluate an RL trading agent",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/config.yaml",
        help="Path to unified system config YAML. Defaults to src/configs/config.yaml",
    )
    parser.add_argument(
        "--env-config",
        type=str,
        required=False,
        help="Path to environment config YAML",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=False,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--trainer-config",
        type=str,
        required=False,
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


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.tune:
        run_tune([args.config])
        return

    cfg_mgr = ConfigManager(args.config)
    system_cfg = cfg_mgr.load_config()

    trainer = Trainer(
        system_cfg,
        seed=args.seed,
        save_dir=args.save_dir,
    )
    if args.train:
        trainer.train()
    if args.eval:
        trainer.evaluate()
    if args.test:
        print("Running tests...")
        trainer.test()


if __name__ == "__main__":
    main()
