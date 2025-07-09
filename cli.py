#!/usr/bin/env python3
"""Unified CLI for Trading RL System using Typer."""
import ast
import importlib
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import typer

from backtesting.backtester import Backtester
from evaluate_agent import main as eval_main
from finrl_data_loader import load_real_data, load_synthetic_data
from training.cnn_lstm import CNNLSTMTrainer
from training.rl import main as rl_main

# Module-level default options to avoid function calls in defaults
GEN_DATA_CONFIG = typer.Option(..., help="Path to data config YAML")
GEN_DATA_SYNTHETIC = typer.Option(False, help="Generate synthetic data")

TRAIN_TYPE = typer.Option(
    ..., "--type", "-t", help="Type of training: 'rl' or 'cnn-lstm'"
)
TRAIN_CONFIGS = typer.Argument(
    ..., help="Config files for training (1 for cnn-lstm, 2 for rl)"
)
TRAIN_NUM_WORKERS = typer.Option(0, help="Number of parallel workers for RL training")
TRAIN_NUM_GPUS = typer.Option(0, help="Number of GPUs for RL training")

BT_DATA = typer.Option(..., help="CSV file with price data")
BT_PRICE_COLUMN = typer.Option("close", help="Column for price series")
BT_SLIPPAGE_PCT = typer.Option(0.0, help="Slippage percentage")
BT_LATENCY = typer.Option(0.0, help="Latency in seconds")
BT_POLICY = typer.Option(
    "lambda p: 'hold'", help="Policy as lambda string or module:func"
)

SERVE_PREDICTOR_PATH = typer.Option(None, help="Path to predictor model checkpoint")

EVAL_DATA = typer.Option(..., help="CSV dataset for evaluation")
EVAL_CHECKPOINT = typer.Option(..., help="Agent checkpoint path")
EVAL_AGENT = typer.Option("sac", "--agent", "-a", help="Agent type to load")
EVAL_OUTPUT = typer.Option("results/evaluation.json", help="Path to save metrics JSON")
EVAL_WINDOW_SIZE = typer.Option(50, help="Observation window size")
app = typer.Typer()


# generate-data command
@app.command(help="Load data via FinRL")
def generate_data(
    config: str = GEN_DATA_CONFIG,
    synthetic: bool = GEN_DATA_SYNTHETIC,
):
    """Generate or load market data using FinRL utilities."""
    if synthetic:
        df = load_synthetic_data(config)
    else:
        df = load_real_data(config)
    typer.echo(df.head())


# train command
@app.command(help="Train models")
def train(
    type: str = TRAIN_TYPE,
    configs: list[str] = TRAIN_CONFIGS,
    num_workers: int = TRAIN_NUM_WORKERS,
    num_gpus: int = TRAIN_NUM_GPUS,
):
    """Train RL or CNN-LSTM models based on config."""
    if type == "cnn-lstm":
        trainer = CNNLSTMTrainer()
        trainer.train_from_config(configs[0])
    else:
        import sys

        sys.argv = [
            "rl_main",
            "--data",
            configs[0],
            "--model-path",
            configs[1],
            "--num-workers",
            str(num_workers),
            "--num-gpus",
            str(num_gpus),
        ]
        rl_main()


# backtest command
@app.command(help="Run backtests using Backtester")
def backtest(
    data: str = BT_DATA,
    price_column: str = BT_PRICE_COLUMN,
    slippage_pct: float = BT_SLIPPAGE_PCT,
    latency: float = BT_LATENCY,
    policy: str = BT_POLICY,
):
    """Backtest a trading policy on historical data."""
    df = pd.read_csv(data)
    prices = df[price_column].tolist()
    try:
        module_name, func_name = policy.split(":", 1)
        module = importlib.import_module(module_name)
        policy_func = getattr(module, func_name)
    except Exception:
        # Safely evaluate simple literal expressions (e.g., predefined lambdas not supported here)
        try:
            policy_func = ast.literal_eval(policy)
        except Exception as err:
            typer.secho(f"Invalid policy expression: {err}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    bt = Backtester(slippage_pct=slippage_pct, latency_seconds=latency)
    results = bt.run_backtest(prices=prices, policy=policy_func)
    typer.echo(results)


# serve command
@app.command(help="Serve predictor deployment with Ray Serve")
def serve(
    predictor_path: str = SERVE_PREDICTOR_PATH,
):
    """Start a Ray Serve deployment for inference."""
    try:
        from ray import serve as ray_serve

        from serve_deployment import deployment_graph
    except ImportError:
        typer.secho(
            "Ray Serve is not installed or src.serve_deployment missing.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    graph = deployment_graph(predictor_path)
    ray_serve.run(graph["predictor"])


# evaluate command
@app.command(help="Evaluate a trained RL agent")
def evaluate(
    data: str = EVAL_DATA,
    checkpoint: str = EVAL_CHECKPOINT,
    agent: str = EVAL_AGENT,
    output: str = EVAL_OUTPUT,
    window_size: int = EVAL_WINDOW_SIZE,
):
    """Evaluate an RL agent and report performance metrics."""
    import sys

    sys.argv = [
        "evaluate_agent",
        "--data",
        data,
        "--checkpoint",
        checkpoint,
        "--agent",
        agent,
        "--output",
        output,
        "--window-size",
        str(window_size),
    ]
    eval_main()


def main():
    """Entry point for console_scripts"""
    app()


if __name__ == "__main__":
    app()
