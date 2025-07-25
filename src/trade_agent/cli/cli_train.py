"""
Training CLI commands.

This module contains all training-related CLI commands including:
- CNN+LSTM model training
- Reinforcement learning agent training
- Hybrid model training
- Hyperparameter optimization
"""

from pathlib import Path
from typing import Annotated

import typer

from .cli_main import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CNN_LSTM_OUTPUT,
    DEFAULT_EPOCHS,
    DEFAULT_GPU,
    DEFAULT_HYBRID_OUTPUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_N_TRIALS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OPTIMIZATION_OUTPUT,
    DEFAULT_RL_OUTPUT,
    DEFAULT_TIMESTEPS,
    console,
    logger,
)

# Training operations sub-app
train_app = typer.Typer(
    name="train",
    help="Model training operations: CNN+LSTM, RL agents, hybrid models",
    rich_markup_mode="rich",
)


@train_app.command()
def cnn_lstm(
    data_path: Annotated[Path, typer.Argument(..., help="Path to CSV dataset file or directory containing dataset.csv")],
    output_dir: Path = DEFAULT_CNN_LSTM_OUTPUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    gpu: bool = DEFAULT_GPU,
    sequence_length: int = typer.Option(60, help="Lookback window length for sequences"),
    prediction_horizon: int = typer.Option(1, help="Steps ahead to predict"),
    optimize_hyperparams: bool = typer.Option(False, help="Run hyperparameter optimization"),
    n_trials: int = typer.Option(50, help="Number of hyperparameter optimization trials"),
) -> None:
    """
    Train a CNN+LSTM model for feature extraction.

    Trains a hybrid CNN+LSTM neural network for pattern recognition in market data.
    The model learns to extract features that will be used by RL agents.

    Examples:
        trade-agent train cnn-lstm data/processed/dataset.csv
        trade-agent train cnn-lstm data/dataset.csv --epochs 200 --gpu
        trade-agent train cnn-lstm data/dataset.csv --optimize-hyperparams --n-trials 100
    """
    console.print("[bold blue]Training CNN+LSTM model...[/bold blue]")

    try:
        from trade_agent.models.cnn_lstm import CNNLSTMTrainer

        # Validate data path
        if not data_path.exists():
            console.print(f"[bold red]Error: Data path does not exist: {data_path}[/bold red]")
            raise typer.Exit(1)

        # If directory provided, look for dataset.csv
        if data_path.is_dir():
            dataset_file = data_path / "dataset.csv"
            if not dataset_file.exists():
                console.print(f"[bold red]Error: dataset.csv not found in {data_path}[/bold red]")
                raise typer.Exit(1)
            data_path = dataset_file

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print("[yellow]Training configuration:[/yellow]")
        console.print(f"  Data: {data_path}")
        console.print(f"  Output: {output_dir}")
        console.print(f"  Epochs: {epochs}")
        console.print(f"  Batch size: {batch_size}")
        console.print(f"  Learning rate: {learning_rate}")
        console.print(f"  Sequence length: {sequence_length}")
        console.print(f"  Prediction horizon: {prediction_horizon}")
        console.print(f"  GPU: {gpu}")
        console.print(f"  Mixed precision: {DEFAULT_MIXED_PRECISION}")

        # Initialize trainer
        trainer = CNNLSTMTrainer(
            data_path=data_path,
            output_dir=output_dir,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            use_gpu=gpu,
            mixed_precision=DEFAULT_MIXED_PRECISION,
        )

        if optimize_hyperparams:
            console.print(f"[green]Running hyperparameter optimization with {n_trials} trials...[/green]")

            result = trainer.optimize_hyperparameters(
                n_trials=n_trials,
                epochs=epochs
            )

            if result.get("success", False):
                best_params = result["best_params"]
                console.print("[bold green]✓ Hyperparameter optimization completed![/bold green]")
                console.print("Best parameters:")
                for param, value in best_params.items():
                    console.print(f"  {param}: {value}")

                # Train with best parameters
                console.print("[green]Training model with optimized parameters...[/green]")
                final_result = trainer.train(**best_params, epochs=epochs)
            else:
                console.print("[bold red]✗ Hyperparameter optimization failed![/bold red]")
                raise typer.Exit(1)
        else:
            # Train with provided parameters
            console.print("[green]Starting training...[/green]")

            final_result = trainer.train(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )

        if final_result.get("success", False):
            console.print("[bold green]✓ Training completed successfully![/bold green]")
            console.print(f"Model saved to: {output_dir}")

            # Show training metrics
            if "metrics" in final_result:
                metrics = final_result["metrics"]
                console.print("\nFinal metrics:")
                console.print(f"  Loss: {metrics.get('final_loss', 'N/A'):.6f}")
                console.print(f"  Accuracy: {metrics.get('final_accuracy', 'N/A'):.4f}")
                console.print(f"  Training time: {metrics.get('training_time', 'N/A')}")
        else:
            console.print("[bold red]✗ Training failed![/bold red]")
            if "error" in final_result:
                console.print(f"Error: {final_result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"CNN+LSTM training error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@train_app.command()
def rl(
    env_name: Annotated[str, typer.Argument(..., help="Environment name for RL training")],
    algorithm: str = typer.Option("ppo", help="RL algorithm: ppo, sac, td3"),
    timesteps: int = DEFAULT_TIMESTEPS,
    output_dir: Path = DEFAULT_RL_OUTPUT,
    cnn_lstm_path: Path | None = None,
    gpu: bool = DEFAULT_GPU,
    num_workers: int = DEFAULT_NUM_WORKERS,
    ray_address: str | None = typer.Option(None, help="Ray cluster address for distributed training"),
) -> None:
    """
    Train a reinforcement learning agent.

    Trains an RL agent (PPO, SAC, or TD3) for trading decisions. Can optionally
    use a pre-trained CNN+LSTM model for feature extraction.

    Examples:
        trade-agent train rl TradingEnv-v0 --algorithm ppo --timesteps 1000000
        trade-agent train rl TradingEnv-v0 --algorithm sac --cnn-lstm-path models/cnn_lstm/model.pt
        trade-agent train rl TradingEnv-v0 --num-workers 8 --ray-address ray://localhost:10001
    """
    console.print(f"[bold blue]Training {algorithm.upper()} agent in {env_name}...[/bold blue]")

    try:
        from trade_agent.models.rl_trainer import RLTrainer

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print("[yellow]Training configuration:[/yellow]")
        console.print(f"  Environment: {env_name}")
        console.print(f"  Algorithm: {algorithm.upper()}")
        console.print(f"  Timesteps: {timesteps:,}")
        console.print(f"  Output: {output_dir}")
        console.print(f"  Workers: {num_workers}")
        console.print(f"  GPU: {gpu}")

        if cnn_lstm_path:
            console.print(f"  CNN+LSTM model: {cnn_lstm_path}")
        if ray_address:
            console.print(f"  Ray address: {ray_address}")

        # Initialize trainer
        trainer = RLTrainer(
            env_name=env_name,
            algorithm=algorithm,
            output_dir=output_dir,
            cnn_lstm_path=cnn_lstm_path,
            use_gpu=gpu,
            num_workers=num_workers,
            ray_address=ray_address,
        )

        # Start training
        console.print("[green]Starting RL training...[/green]")

        result = trainer.train(timesteps=timesteps)

        if result.get("success", False):
            console.print("[bold green]✓ RL training completed successfully![/bold green]")
            console.print(f"Model saved to: {output_dir}")

            # Show training metrics
            if "metrics" in result:
                metrics = result["metrics"]
                console.print("\nFinal metrics:")
                console.print(f"  Episode reward: {metrics.get('episode_reward_mean', 'N/A'):.2f}")
                console.print(f"  Episodes: {metrics.get('episodes_total', 'N/A')}")
                console.print(f"  Training time: {metrics.get('training_time', 'N/A')}")
        else:
            console.print("[bold red]✗ RL training failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"RL training error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@train_app.command()
def hybrid(
    data_path: Annotated[Path, typer.Argument(..., help="Path to training dataset")],
    cnn_lstm_path: Annotated[Path, typer.Argument(..., help="Path to pre-trained CNN+LSTM model")],
    output_dir: Path = DEFAULT_HYBRID_OUTPUT,
    algorithm: str = typer.Option("ppo", help="RL algorithm for hybrid training"),
    timesteps: int = DEFAULT_TIMESTEPS,
    gpu: bool = DEFAULT_GPU,
) -> None:
    """
    Train a hybrid CNN+LSTM+RL model.

    Combines a pre-trained CNN+LSTM model with an RL agent for end-to-end
    trading system training.

    Examples:
        trade-agent train hybrid data/dataset.csv models/cnn_lstm/model.pt
        trade-agent train hybrid data/dataset.csv models/cnn_lstm/model.pt --algorithm sac
    """
    console.print("[bold blue]Training hybrid CNN+LSTM+RL model...[/bold blue]")

    try:
        from trade_agent.models.hybrid_trainer import HybridTrainer

        # Validate paths
        if not data_path.exists():
            console.print(f"[bold red]Error: Data path does not exist: {data_path}[/bold red]")
            raise typer.Exit(1)

        if not cnn_lstm_path.exists():
            console.print(f"[bold red]Error: CNN+LSTM model does not exist: {cnn_lstm_path}[/bold red]")
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print("[yellow]Training configuration:[/yellow]")
        console.print(f"  Data: {data_path}")
        console.print(f"  CNN+LSTM model: {cnn_lstm_path}")
        console.print(f"  Algorithm: {algorithm.upper()}")
        console.print(f"  Timesteps: {timesteps:,}")
        console.print(f"  Output: {output_dir}")
        console.print(f"  GPU: {gpu}")

        # Initialize trainer
        trainer = HybridTrainer(
            data_path=data_path,
            cnn_lstm_path=cnn_lstm_path,
            algorithm=algorithm,
            output_dir=output_dir,
            use_gpu=gpu,
        )

        # Start training
        console.print("[green]Starting hybrid training...[/green]")

        result = trainer.train(timesteps=timesteps)

        if result.get("success", False):
            console.print("[bold green]✓ Hybrid training completed successfully![/bold green]")
            console.print(f"Model saved to: {output_dir}")

            # Show training metrics
            if "metrics" in result:
                metrics = result["metrics"]
                console.print("\nFinal metrics:")
                console.print(f"  Episode reward: {metrics.get('episode_reward_mean', 'N/A'):.2f}")
                console.print(f"  Training episodes: {metrics.get('episodes_total', 'N/A')}")
                console.print(f"  Training time: {metrics.get('training_time', 'N/A')}")
        else:
            console.print("[bold red]✗ Hybrid training failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Hybrid training error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@train_app.command()
def optimize(
    config_path: Annotated[Path, typer.Argument(..., help="Path to hyperparameter optimization config")],
    output_dir: Path = DEFAULT_OPTIMIZATION_OUTPUT,
    n_trials: int = DEFAULT_N_TRIALS,
    study_name: str | None = typer.Option(None, help="Optuna study name"),
    storage_url: str | None = typer.Option(None, help="Optuna storage URL"),
) -> None:
    """
    Run hyperparameter optimization for model training.

    Uses Optuna for automated hyperparameter tuning across different model types
    and training configurations.

    Examples:
        trade-agent train optimize configs/optimization.yaml
        trade-agent train optimize configs/optimization.yaml --n-trials 200
    """
    console.print("[bold blue]Running hyperparameter optimization...[/bold blue]")

    try:
        from trade_agent.optimization.hyperopt import HyperparameterOptimizer

        # Validate config path
        if not config_path.exists():
            console.print(f"[bold red]Error: Config file does not exist: {config_path}[/bold red]")
            raise typer.Exit(1)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Display configuration
        console.print("[yellow]Optimization configuration:[/yellow]")
        console.print(f"  Config: {config_path}")
        console.print(f"  Trials: {n_trials}")
        console.print(f"  Output: {output_dir}")
        if study_name:
            console.print(f"  Study name: {study_name}")
        if storage_url:
            console.print(f"  Storage: {storage_url}")

        # Initialize optimizer
        optimizer = HyperparameterOptimizer(
            config_path=config_path,
            output_dir=output_dir,
            study_name=study_name,
            storage_url=storage_url,
        )

        # Start optimization
        console.print("[green]Starting hyperparameter optimization...[/green]")

        result = optimizer.optimize(n_trials=n_trials)

        if result.get("success", False):
            console.print("[bold green]✓ Optimization completed successfully![/bold green]")
            console.print(f"Results saved to: {output_dir}")

            # Show best parameters
            if "best_params" in result:
                best_params = result["best_params"]
                console.print("\nBest parameters:")
                for param, value in best_params.items():
                    console.print(f"  {param}: {value}")

            if "best_value" in result:
                console.print(f"Best objective value: {result['best_value']:.6f}")
        else:
            console.print("[bold red]✗ Optimization failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")
            raise typer.Exit(1)

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e
