import importlib
import sys
import traceback
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from config import get_settings, load_settings

from .console import console

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
            mod = importlib.import_module("src.trade_agent.training.rl_trainers")
            return getattr(mod, f"train_{algo}")
        except (ImportError, AttributeError) as err:
            raise ImportError(f"No RL trainer implemented for algorithm: {algo}") from err
    elif algo in {"dqn", "lstm", "cnn_lstm"}:
        try:
            mod = importlib.import_module("src.trade_agent.training.supervised_trainers")
            return getattr(mod, f"train_{algo}")
        except (ImportError, AttributeError) as err:
            raise ImportError(f"No supervised trainer implemented for algorithm: {algo}") from err
    else:
        raise ImportError(f"No trainer found for algorithm: {algo}")


@app.command()
def train(
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_FILE,
    epochs: Annotated[int | None, typer.Option("--epochs", "-e", help="Override number of epochs")] = DEFAULT_EPOCHS,
    lr: Annotated[float | None, typer.Option("--lr", help="Override learning rate")] = DEFAULT_LR,
    devices: Annotated[
        str | None, typer.Option("--devices", help="Override devices (e.g., 'cpu', 'cuda', 'cuda:0,1')")
    ] = DEFAULT_DEVICES,
    checkpoint_out: Annotated[
        Path | None, typer.Option("--checkpoint-out", help="Output checkpoint directory")
    ] = DEFAULT_CHECKPOINT_OUT,
    log_interval: Annotated[
        int, typer.Option("--log-interval", help="Log metrics every N epochs/steps")
    ] = DEFAULT_LOG_INTERVAL,
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
    config_file: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to config file")
    ] = DEFAULT_CONFIG_FILE,
    devices: Annotated[
        str | None, typer.Option("--devices", help="Override devices (e.g., 'cpu', 'cuda', 'cuda:0,1')")
    ] = DEFAULT_DEVICES,
    log_interval: Annotated[
        int, typer.Option("--log-interval", help="Log metrics every N epochs/steps")
    ] = DEFAULT_LOG_INTERVAL,
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


@app.command()
def cnn_lstm_visual(
    data_path: Annotated[
        Path | None, typer.Option("--data", "-d", help="Path to CSV data file")
    ] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of epochs")] = 30,
    lr: Annotated[float, typer.Option("--lr", help="Learning rate")] = 0.001,
    batch_size: Annotated[int, typer.Option("--batch", help="Batch size")] = 32,
    sequence_length: Annotated[int, typer.Option("--seq-len", help="Sequence length")] = 60,
    no_visual: Annotated[bool, typer.Option("--no-visual", help="Disable visual monitoring")] = False,
) -> None:
    """
    Train CNN-LSTM model with automatic visual monitoring.

    This command automatically starts real-time visual monitoring when training begins,
    showing live training metrics, loss curves, and model performance.
    """
    try:
        console.print("🎬 [bold green]CNN-LSTM Training with Visual Monitoring[/bold green]")
        console.print("=" * 60)

        # Import here to avoid circular imports
        from trade_agent.training.train_cnn_lstm_enhanced import (
            EnhancedCNNLSTMTrainer,
            create_enhanced_model_config,
            create_enhanced_training_config,
            load_and_preprocess_csv_data,
        )

        # Load or create data
        if data_path and data_path.exists():
            console.print(f"📊 Loading data from: {data_path}")
            sequences, targets = load_and_preprocess_csv_data(
                csv_path=data_path,
                sequence_length=sequence_length,
                prediction_horizon=1
            )
            input_dim = sequences.shape[-1]
        else:
            console.print("🔧 Creating synthetic demo data...")
            # Create synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            n_features = 5

            # Generate synthetic time series data
            data = np.random.randn(n_samples + sequence_length, n_features).astype(np.float32)
            sequences = []
            targets = []

            for i in range(n_samples):
                seq = data[i:i + sequence_length]
                target = np.mean(data[i + sequence_length]) + 0.1 * np.random.randn()
                sequences.append(seq)
                targets.append(target)

            sequences = np.array(sequences, dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)
            input_dim = n_features

        console.print(f"✅ Data loaded: {sequences.shape} sequences, {targets.shape} targets")

        # Create model configuration
        model_config = create_enhanced_model_config(
            input_dim=input_dim,
            cnn_filters=[64, 128, 256],
            cnn_kernel_sizes=[3, 3, 3],
            lstm_units=128,
            lstm_layers=2,
            dropout_rate=0.2,
            use_attention=True
        )

        # Create training configuration
        training_config = create_enhanced_training_config(
            learning_rate=lr,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=10
        )

        console.print(f"🏗️  Model: CNN{model_config['cnn_filters']} -> LSTM({model_config['lstm_units']})")
        console.print(f"⚙️  Training: {epochs} epochs, LR={lr}, Batch={batch_size}")
        console.print(f"📈 Visual Monitor: {'❌ Disabled' if no_visual else '✅ Auto-enabled'}")

        # Create trainer with visual monitoring
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            enable_visual_monitor=not no_visual,  # Visual monitoring by default!
            enable_mlflow=False,
            enable_tensorboard=False
        )

        console.print("\n🚀 [bold yellow]Starting training with automatic visual monitoring...[/bold yellow]")
        if not no_visual:
            console.print("📊 Real-time visualizations will appear automatically!")
            console.print("📁 Check './training_visualizations/' for saved charts")

        # Start training - visual monitor activates automatically!
        results = trainer.train_from_dataset(sequences, targets)

        console.print("\n✅ [bold green]Training completed successfully![/bold green]")
        console.print(f"🎯 Best validation loss: {results['best_val_loss']:.4f}")
        console.print(f"📊 Total epochs: {results['total_epochs']}")
        console.print(f"⏱️  Training time: {results['training_time']:.2f}s")

        if not no_visual:
            console.print("📁 Visual monitoring assets saved to: ./training_visualizations/")

    except Exception as e:
        console.print(f"[red]CNN-LSTM training error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def cnn_lstm_optimize(
    data_path: Annotated[
        Path | None, typer.Option("--data", "-d", help="Path to CSV data file")
    ] = None,
    trials: Annotated[int, typer.Option("--trials", "-t", help="Number of optimization trials")] = 20,
    timeout: Annotated[int | None, typer.Option("--timeout", help="Timeout in seconds")] = None,
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of epochs per trial")] = 50,
    no_visual: Annotated[bool, typer.Option("--no-visual", help="Disable visual monitoring")] = False,
) -> None:
    """
    Optimize CNN-LSTM hyperparameters with visual trial monitoring.

    Uses Optuna to find the best hyperparameters while showing real-time
    trial progress, parameter importance, and optimization convergence.
    """
    try:
        console.print("🔬 [bold green]CNN-LSTM Hyperparameter Optimization with Visual Monitoring[/bold green]")
        console.print("=" * 70)

        # Import here to avoid circular imports
        from trade_agent.training.train_cnn_lstm_enhanced import HyperparameterOptimizer, load_and_preprocess_csv_data

        # Load or create data
        if data_path and data_path.exists():
            console.print(f"📊 Loading data from: {data_path}")
            sequences, targets = load_and_preprocess_csv_data(
                csv_path=data_path,
                sequence_length=60,
                prediction_horizon=1
            )
        else:
            console.print("🔧 Creating synthetic demo data for optimization...")
            # Create synthetic data
            np.random.seed(42)
            n_samples = 800
            n_features = 5
            sequence_length = 40

            data = np.random.randn(n_samples + sequence_length, n_features).astype(np.float32)
            sequences = []
            targets = []

            for i in range(n_samples):
                seq = data[i:i + sequence_length]
                target = np.mean(data[i + sequence_length]) + 0.1 * np.random.randn()
                sequences.append(seq)
                targets.append(target)

            sequences = np.array(sequences, dtype=np.float32)
            targets = np.array(targets, dtype=np.float32)

        console.print(f"✅ Data loaded: {sequences.shape} sequences, {targets.shape} targets")
        console.print(f"🔍 Optimization: {trials} trials, {epochs} epochs per trial{'⏱️ ' + str(timeout) + 's timeout' if timeout else ''}")
        console.print(f"📈 Visual Monitor: {'❌ Disabled' if no_visual else '✅ Auto-enabled'}")

        # Create hyperparameter optimizer with visual monitoring
        optimizer = HyperparameterOptimizer(
            sequences=sequences,
            targets=targets,
            n_trials=trials,
            timeout=timeout,
            enable_visual_monitor=not no_visual,  # Visual monitoring for trials!
            epochs=epochs
        )

        console.print("\n🎯 [bold yellow]Starting Optuna optimization with visual monitoring...[/bold yellow]")
        if not no_visual:
            console.print("📊 Trial progress will be visualized in real-time!")
            console.print("🔬 Hyperparameter relationships will be analyzed")
            console.print("📁 Check './training_visualizations/' for Optuna charts")

        # Run optimization - visual monitor tracks trials automatically!
        results = optimizer.optimize()

        console.print("\n🎯 [bold green]Optimization completed![/bold green]")
        console.print(f"🏆 Best score: {results['best_score']:.4f}")
        console.print(f"📊 Total trials: {len(results['study'].trials)}")

        # Display best parameters
        console.print("\n🏅 [bold yellow]Best parameters found:[/bold yellow]")
        for param, value in results["best_params"].items():
            console.print(f"   {param}: {value}")

        if not no_visual:
            console.print("\n📁 Optuna visualizations saved to: ./training_visualizations/")

    except Exception as e:
        console.print(f"[red]Optimization error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def visual_demo() -> None:
    """
    Run comprehensive demonstration of visual monitoring features.

    Shows basic training, Optuna optimization, and CSV data processing
    with automatic visual monitoring for each scenario.
    """
    try:
        console.print("🎭 [bold green]CNN-LSTM Visual Monitoring Demo Suite[/bold green]")
        console.print("=" * 60)

        # Import and run the demo
        from trade_agent.training.auto_visual_demo import run_all_demos

        console.print("🎬 Running comprehensive visual monitoring demonstrations...")
        console.print("📊 All training sessions will have automatic visual monitoring!")

        success = run_all_demos()

        if success:
            console.print("\n🎉 [bold green]All demonstrations completed successfully![/bold green]")
            console.print("📁 Check './training_visualizations/' for all generated assets")
        else:
            console.print("\n❌ [bold red]Demo encountered some issues[/bold red]")
            raise typer.Exit(1)

    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
