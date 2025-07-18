import importlib
import sys
import traceback
from pathlib import Path
from typing import Any

import typer

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_settings, load_settings

from .console import console

app = typer.Typer(help="Training CLI for Trading RL Agent")

# Module-level constants for typer defaults to fix B008 errors
DEFAULT_CONFIG_FILE: Path | None = None
DEFAULT_EPOCHS: int | None = None
DEFAULT_LR: float | None = None
DEFAULT_DEVICES: str | None = None
DEFAULT_CHECKPOINT_OUT: Path | None = None
DEFAULT_LOG_INTERVAL = 10


def get_trainer(algo: str) -> Any:
    """Dynamically import the correct training routine based on algorithm name."""
    algo = algo.lower()
    if algo in {"ppo", "sac", "td3"}:
        # RLlib/Stable Baselines3 RL algorithms
        try:
            mod = importlib.import_module("src.trading_rl_agent.training.rl_trainers")
            return getattr(mod, f"train_{algo}")
        except (ImportError, AttributeError) as err:
            raise ImportError(f"No RL trainer implemented for algorithm: {algo}") from err
    elif algo in {"dqn", "lstm", "cnn_lstm"}:
        try:
            mod = importlib.import_module("src.trading_rl_agent.training.supervised_trainers")
            return getattr(mod, f"train_{algo}")
        except (ImportError, AttributeError) as err:
            raise ImportError(f"No supervised trainer implemented for algorithm: {algo}") from err
    else:
        raise ImportError(f"No trainer found for algorithm: {algo}")


@app.command()
def train(
    config_file: Path | None = typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c", help="Path to config file"),  # noqa: B008
    epochs: int | None = typer.Option(DEFAULT_EPOCHS, "--epochs", "-e", help="Override number of epochs"),
    lr: float | None = typer.Option(DEFAULT_LR, "--lr", help="Override learning rate"),
    devices: str | None = typer.Option(
        DEFAULT_DEVICES,
        "--devices",
        help="Override devices (e.g., 'cpu', 'cuda', 'cuda:0,1')",
    ),
    checkpoint_out: Path | None = typer.Option(  # noqa: B008
        DEFAULT_CHECKPOINT_OUT,
        "--checkpoint-out",
        help="Output checkpoint directory",
    ),
    log_interval: int = typer.Option(DEFAULT_LOG_INTERVAL, "--log-interval", help="Log metrics every N epochs/steps"),
) -> None:
    """
    Train a model using the algorithm specified in Settings.model.algorithm.
    CLI options override config values.
    """
    try:
        settings = load_settings(config_path=config_file) if config_file else get_settings()
        algo = settings.model.algorithm
        trainer = get_trainer(algo)

        # Prepare training kwargs
        train_kwargs = {
            "settings": settings,
            "epochs": epochs or settings.model.epochs,
            "learning_rate": lr or settings.model.learning_rate,
            "devices": devices or settings.model.device,
            "checkpoint_dir": str(checkpoint_out or Path(settings.model.checkpoint_dir)),
            "log_interval": log_interval,
            "console": console,
        }

        console.print(f"[green]Starting training: {algo}[/green]")
        result = trainer(**train_kwargs)

        if result.get("success", False):
            console.print("[bold green]Training completed successfully![/bold green]")
            raise typer.Exit(0)
        console.print(f"[bold red]Training failed: {result.get('error', 'Unknown error')}[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Training error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def resume(
    config_file: Path | None = typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c", help="Path to config file"),  # noqa: B008
    devices: str | None = typer.Option(
        DEFAULT_DEVICES,
        "--devices",
        help="Override devices (e.g., 'cpu', 'cuda', 'cuda:0,1')",
    ),
    log_interval: int = typer.Option(DEFAULT_LOG_INTERVAL, "--log-interval", help="Log metrics every N epochs/steps"),
) -> None:
    """
    Resume training from the last checkpoint in Settings.model.checkpoint_dir.
    """
    try:
        settings = load_settings(config_path=config_file) if config_file else get_settings()
        algo = settings.model.algorithm
        trainer = get_trainer(algo)
        checkpoint_dir = Path(settings.model.checkpoint_dir)

        # Find last checkpoint
        if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            console.print(f"[red]No checkpoint found in {checkpoint_dir}[/red]")
            raise typer.Exit(1)
        latest_ckpt = max(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, default=None)
        if not latest_ckpt:
            console.print(f"[red]No checkpoint file found in {checkpoint_dir}[/red]")
            raise typer.Exit(1)

        console.print(f"[green]Resuming training from checkpoint: {latest_ckpt}[/green]")

        # Prepare training kwargs
        train_kwargs = {
            "settings": settings,
            "resume_from": str(latest_ckpt),
            "devices": devices or settings.model.device,
            "checkpoint_dir": str(checkpoint_dir),
            "log_interval": log_interval,
            "console": console,
        }
        result = trainer(**train_kwargs)
        if result.get("success", False):
            console.print("[bold green]Resume completed successfully![/bold green]")
            raise typer.Exit(0)
        console.print(f"[bold red]Resume failed: {result.get('error', 'Unknown error')}[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Resume error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
