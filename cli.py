#!/usr/bin/env python3
"""
Unified CLI for Trading RL System.

Subcommands:
  generate-data   Load data via FinRL utilities
  train           Train RL or CNN-LSTM models
  backtest        Run backtests using Backtester
  serve           Serve predictor deployment with Ray Serve
  evaluate        Evaluate a trained RL agent
"""
import argparse
import importlib
import sys

import pandas as pd

# import modules for commands
from finrl_data_loader import main as _data_main
from evaluate_agent import main as _evaluate_agent_main
from src.backtesting import Backtester
from src.main import main as _trainer_main


def _cmd_generate_data(args):
    # Delegate to finrl_data_loader.main
    sys.argv = ["finrl_data_loader", "--config", args.config]
    if args.synthetic:
        sys.argv.append("--synthetic")
    _data_main()


def _cmd_train(args):
    if args.type == "cnn-lstm":
        from src.training.cnn_lstm import CNNLSTMTrainer

        trainer = CNNLSTMTrainer()
        trainer.train_from_config(args.configs[0])
    else:
        # RL training
        # Set sys.argv for rl_main if it expects command-line arguments
        import sys

        from src.training.rl import main as rl_main

        sys.argv = [
            "rl_main",
            "--data",
            args.configs[0],
            "--model-path",
            args.configs[1],
            "--num-workers",
            str(args.num_workers),
            "--num-gpus",
            str(args.num_gpus),
        ]
        rl_main()


def _load_policy(policy_str: str):
    """Load a policy function specified as 'module:func_name'."""
    try:
        module_name, func_name = policy_str.split(":", 1)
    except ValueError:
        raise ValueError("Policy must be specified as 'module:func_name'")
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
    except Exception as err:
        raise ValueError(f"Cannot load policy '{policy_str}': {err}")
    if not callable(func):
        raise ValueError(
            f"Policy '{func_name}' in module '{module_name}' is not callable"
        )
    return func


def _cmd_backtest(args):
    df = pd.read_csv(args.data)
    prices = df[args.price_column].tolist()
    # Load user-provided policy function
    try:
        policy = _load_policy(args.policy)
    except ValueError as err:
        print(f"Invalid policy: {err}")
        sys.exit(1)
    bt = Backtester(slippage_pct=args.slippage_pct, latency_seconds=args.latency)
    results = bt.run_backtest(prices=prices, policy=policy)
    print(results)


def _cmd_serve(args):
    try:
        from ray import serve

        # Use deployment_graph to get predictor deployment
        from src.serve_deployment import deployment_graph
    except ImportError:
        print("Ray Serve is not installed or src.serve_deployment missing.")
        sys.exit(1)
    # Build deployment graph and run the predictor deployment
    graph = deployment_graph(args.predictor_path)
    serve.run(graph["predictor"])


def _cmd_evaluate(args):
    argv = [
        "--data",
        args.data,
        "--checkpoint",
        args.checkpoint,
        "--agent",
        args.agent,
        "--output",
        args.output,
        "--window-size",
        str(args.window_size),
    ]
    _evaluate_agent_main()


def main():
    parser = argparse.ArgumentParser(description="Unified CLI for Trading RL System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Load data via FinRL")
    p_gen.add_argument("--config", required=True, help="Path to data config YAML")
    p_gen.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic data"
    )
    p_gen.set_defaults(func=_cmd_generate_data)

    # train
    p_train = subparsers.add_parser("train", help="Train models")
    p_train.add_argument(
        "--type",
        choices=["rl", "cnn-lstm"],
        required=True,
        help="Type of training: 'rl' or 'cnn-lstm'",
    )
    p_train.add_argument("configs", nargs="+", help="Config files for training")
    p_train.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel workers for RL training",
    )
    p_train.add_argument(
        "--num-gpus", type=int, default=0, help="Number of GPUs for RL training"
    )
    p_train.set_defaults(func=_cmd_train)

    # backtest
    p_bt = subparsers.add_parser("backtest", help="Run backtest using Backtester")
    p_bt.add_argument("--data", required=True, help="CSV file with price data")
    p_bt.add_argument("--price-column", default="close", help="Column for price series")
    p_bt.add_argument(
        "--slippage-pct", type=float, default=0.0, help="Slippage percentage"
    )
    p_bt.add_argument("--latency", type=float, default=0.0, help="Latency in seconds")
    p_bt.add_argument(
        "--policy",
        default="lambda p: 'hold'",
        help="Policy as lambda string, e.g., \"lambda p: 'buy' if p>1 else 'sell'\"",
    )
    p_bt.set_defaults(func=_cmd_backtest)

    # serve
    p_serve = subparsers.add_parser("serve", help="Serve predictor deployment")
    p_serve.add_argument(
        "--predictor-path", default=None, help="Path to predictor model checkpoint"
    )
    p_serve.set_defaults(func=_cmd_serve)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained RL agent")
    p_eval.add_argument("--data", required=True, help="CSV dataset for evaluation")
    p_eval.add_argument("--checkpoint", required=True, help="Agent checkpoint path")
    p_eval.add_argument(
        "--agent",
        choices=["sac", "td3", "ensemble"],
        default="sac",
        help="Agent type to load",
    )
    p_eval.add_argument(
        "--output", default="results/evaluation.json", help="Path to save metrics JSON"
    )
    p_eval.add_argument(
        "--window-size", type=int, default=50, help="Observation window size"
    )
    p_eval.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
