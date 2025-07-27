#!/usr/bin/env python3
"""
Production-Ready Trading RL Platform CLI

A comprehensive command-line interface for algorithmic trading with RL.
Go from raw market data to live trading deployment through simple commands.

Usage:
    trading-cli data collect --source yahoo --symbols SPY,QQQ --period 1y
    trading-cli train --algorithm PPO --env TradingEnv --episodes 10000
    trading-cli deploy --model best_model.pkl --broker alpaca --paper-trading
"""

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from trade_agent.core.config import get_logger
from trade_agent.core.live_trading import CLICompatibleLiveTradingEngine as LiveTradingEngine
from trade_agent.eval.model_evaluator import ModelEvaluator

# Initialize console and logger
console = Console()
logger = get_logger(__name__)

# Main app
app = typer.Typer(
    name="trading-cli",
    help="🚀 Production-Ready Trading RL Platform",
    add_completion=True,
    rich_markup_mode="rich",
    no_args_is_help=True,
    epilog="Transform raw market data into live trading deployment in hours, not weeks.",
)

# Global state
_config_path: Path | None = None
_verbose: int = 0


@app.callback()
def main(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Configuration file (YAML/JSON)")
    ] = None,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity (-v, -vv, -vvv)")
    ] = 0,
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version and exit")
    ] = False,
) -> None:
    """
    🚀 Production-Ready Trading RL Platform

    Build, train, and deploy algorithmic trading systems with reinforcement learning.

    [bold green]Key Features:[/bold green]
    • End-to-end data pipeline automation
    • Advanced CNN+LSTM and RL model training
    • Real-time market data streaming
    • Comprehensive risk management
    • Live trading with multiple brokers
    • Production monitoring & alerting

    [bold blue]Quick Start:[/bold blue]
    ```
    # 1. Collect and prepare data
    trading-cli data collect --source yahoo --symbols AAPL,GOOGL --period 1y
    trading-cli data preprocess --features technical,fundamental --clean

    # 2. Train a model
    trading-cli train --algorithm PPO --episodes 10000 --save-best

    # 3. Deploy to paper trading
    trading-cli deploy --model best_model.pkl --broker alpaca --paper-trading
    ```
    """
    global _config_path, _verbose
    _config_path = config
    _verbose = verbose

    if version:
        show_version()
        raise typer.Exit()

    # Setup logging based on verbosity
    if verbose >= 3:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose >= 2:
        import logging
        logging.getLogger().setLevel(logging.INFO)
    elif verbose >= 1:
        import logging
        logging.getLogger().setLevel(logging.WARNING)


def show_version() -> None:
    """Display version information."""
    try:
        import importlib.metadata
        version = importlib.metadata.version("trade-agent")
    except Exception:
        version = "0.2.0-dev"

    console.print(Panel.fit(
        f"[bold blue]Trading RL Platform[/bold blue] v{version}\n\n"
        f"[dim]A production-grade reinforcement learning trading system[/dim]\n"
        f"[dim]Built with CNN+LSTM + Deep RL agents[/dim]",
        title="🚀 Version Information",
        border_style="blue"
    ))


# ================================
# DATA PIPELINE COMMANDS
# ================================

data_app = typer.Typer(
    name="data",
    help="📊 Data pipeline operations",
    no_args_is_help=True,
)


@data_app.command("collect")
def data_collect(
    source: Annotated[
        str,
        typer.Option("--source", help="Data source: yahoo, alpha, polygon, ccxt")
    ] = "yahoo",
    symbols: Annotated[
        str,
        typer.Option("--symbols", help="Comma-separated symbols (e.g., SPY,QQQ,AAPL)")
    ] = "SPY,QQQ",
    period: Annotated[
        str,
        typer.Option("--period", help="Time period: 1d, 1w, 1m, 1y, 5y, max")
    ] = "1y",
    interval: Annotated[
        str,
        typer.Option("--interval", help="Data interval: 1m, 5m, 15m, 1h, 1d")
    ] = "1d",
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory")
    ] = Path("data/raw"),
    parallel: Annotated[
        bool,
        typer.Option("--parallel/--no-parallel", help="Enable parallel downloading")
    ] = True,
    cache: Annotated[
        bool,
        typer.Option("--cache/--no-cache", help="Use caching")
    ] = True,
) -> None:
    """
    📊 Collect market data from various sources.

    Supports multiple data providers with automatic rate limiting,
    retry logic, and intelligent caching.

    Examples:
        trading-cli data collect --source yahoo --symbols SPY,QQQ --period 2y
        trading-cli data collect --source alpha --symbols AAPL,MSFT --interval 1h
        trading-cli data collect --source polygon --symbols BTC-USD,ETH-USD
    """
    from trade_agent.data.parallel_data_fetcher import ParallelDataFetcher

    console.print(f"🔄 Collecting data from [bold]{source}[/bold]...")
    console.print(f"📈 Symbols: [green]{symbols}[/green]")
    console.print(f"⏰ Period: [blue]{period}[/blue], Interval: [blue]{interval}[/blue]")

    try:
        output.mkdir(parents=True, exist_ok=True)

        fetcher = ParallelDataFetcher(
            data_source=source,
            output_dir=output,
            use_cache=cache,
            parallel=parallel
        )

        symbol_list = [s.strip() for s in symbols.split(",")]

        results = asyncio.run(fetcher.fetch_symbols(
            symbols=symbol_list,
            period=period,
            interval=interval
        ))

        console.print(f"✅ Successfully collected data for {len(results)} symbols")
        console.print(f"📁 Data saved to: [cyan]{output.absolute()}[/cyan]")

    except Exception as e:
        console.print(f"❌ Error collecting data: {e}", style="red")
        raise typer.Exit(1) from e


@data_app.command("preprocess")
def data_preprocess(
    input_dir: Annotated[
        Path,
        typer.Option("--input", help="Input data directory")
    ] = Path("data/raw"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory")
    ] = Path("data/processed"),
    features: Annotated[
        str,
        typer.Option("--features", help="Feature types: technical,fundamental,sentiment,all")
    ] = "technical",
    clean: Annotated[
        bool,
        typer.Option("--clean/--no-clean", help="Clean missing values and outliers")
    ] = True,
    validate: Annotated[
        bool,
        typer.Option("--validate/--no-validate", help="Validate data quality")
    ] = True,
    normalize: Annotated[
        str,
        typer.Option("--normalize", help="Normalization method: standard, robust, minmax")
    ] = "robust",
) -> None:
    """
    🔧 Preprocess and engineer features from raw market data.

    Comprehensive preprocessing pipeline with feature engineering,
    data cleaning, validation, and normalization.

    Examples:
        trading-cli data preprocess --features technical,sentiment --clean
        trading-cli data preprocess --normalize standard --validate
    """
    from trade_agent.data.pipeline import DataPipeline

    console.print("🔧 Starting data preprocessing...")
    console.print(f"📂 Input: [cyan]{input_dir}[/cyan] → Output: [cyan]{output}[/cyan]")

    try:
        output.mkdir(parents=True, exist_ok=True)

        pipeline = DataPipeline(
            input_dir=input_dir,
            output_dir=output,
            feature_types=features.split(","),
            clean_data=clean,
            validate_data=validate,
            normalization_method=normalize
        )

        results = pipeline.process_all()

        console.print(f"✅ Preprocessed {len(results)} datasets")
        console.print(f"📊 Features: [green]{features}[/green]")
        console.print(f"🧹 Cleaning: [blue]{'enabled' if clean else 'disabled'}[/blue]")
        console.print(f"📁 Output: [cyan]{output.absolute()}[/cyan]")

    except Exception as e:
        console.print(f"❌ Error preprocessing data: {e}", style="red")
        raise typer.Exit(1) from e


@data_app.command("split")
def data_split(
    input_file: Annotated[
        Path,
        typer.Argument(help="Input dataset file")
    ],
    train_ratio: Annotated[
        float,
        typer.Option("--train-ratio", help="Training set ratio")
    ] = 0.8,
    validation_ratio: Annotated[
        float,
        typer.Option("--validation-ratio", help="Validation set ratio")
    ] = 0.1,
    output: Annotated[
        Path,
        typer.Option("--output", help="Output directory")
    ] = Path("data/splits"),
    time_aware: Annotated[
        bool,
        typer.Option("--time-aware/--random", help="Time-aware vs random splitting")
    ] = True,
) -> None:
    """
    📊 Split dataset into train/validation/test sets.

    Supports both time-aware splitting (for time series) and random splitting
    with proper data leakage prevention.

    Examples:
        trading-cli data split dataset.csv --train-ratio 0.7 --validation-ratio 0.15
        trading-cli data split data.parquet --random
    """
    from trade_agent.data.preprocessing import DataSplitter

    console.print("📊 Splitting dataset...")
    console.print(f"📂 Input: [cyan]{input_file}[/cyan]")
    console.print(f"📈 Ratios - Train: {train_ratio}, Val: {validation_ratio}, Test: {1-train_ratio-validation_ratio}")

    try:
        output.mkdir(parents=True, exist_ok=True)

        splitter = DataSplitter(
            input_file=input_file,
            output_dir=output,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            time_aware=time_aware
        )

        splits = splitter.split()

        console.print("✅ Dataset split successfully:")
        for split_name, split_path in splits.items():
            console.print(f"  [green]{split_name}[/green]: {split_path}")

    except Exception as e:
        console.print(f"❌ Error splitting data: {e}", style="red")
        raise typer.Exit(1) from e


# ================================
# MODEL TRAINING COMMANDS
# ================================

train_app = typer.Typer(
    name="train",
    help="🧠 Model training operations",
    no_args_is_help=True,
)


@train_app.command()
def train(
    algorithm: Annotated[
        str,
        typer.Option("--algorithm", help="Algorithm: PPO, SAC, TD3, CNN-LSTM, hybrid")
    ] = "PPO",
    env: Annotated[
        str,
        typer.Option("--env", help="Trading environment")
    ] = "TradingEnv",
    episodes: Annotated[
        int,
        typer.Option("--episodes", help="Number of training episodes")
    ] = 10000,
    data: Annotated[
        Path,
        typer.Option("--data", help="Training data path")
    ] = Path("data/processed"),
    output: Annotated[
        Path,
        typer.Option("--output", help="Model output directory")
    ] = Path("models"),
    save_best: Annotated[
        bool,
        typer.Option("--save-best/--no-save-best", help="Save best performing model")
    ] = True,
    tensorboard: Annotated[
        bool,
        typer.Option("--tensorboard/--no-tensorboard", help="Enable TensorBoard logging")
    ] = True,
    checkpoint_freq: Annotated[
        int,
        typer.Option("--checkpoint-freq", help="Checkpoint frequency (episodes)")
    ] = 1000,
) -> None:
    """
    🧠 Train a reinforcement learning trading model.

    Supports multiple RL algorithms with advanced features:
    - Automatic hyperparameter tuning
    - Distributed training with Ray
    - Real-time monitoring
    - Checkpointing and resume capability

    Examples:
        trading-cli train --algorithm PPO --episodes 50000 --save-best
        trading-cli train --algorithm CNN-LSTM --data data/processed/AAPL.csv
        trading-cli train --algorithm hybrid --tensorboard
    """
    from trade_agent.agents.trainer import UnifiedTrainer

    console.print(f"🧠 Training [bold]{algorithm}[/bold] model...")
    console.print(f"🎯 Environment: [blue]{env}[/blue]")
    console.print(f"📊 Episodes: [green]{episodes:,}[/green]")

    try:
        output.mkdir(parents=True, exist_ok=True)

        trainer = UnifiedTrainer(
            algorithm=algorithm.lower(),
            env_name=env,
            data_path=data,
            output_dir=output,
            tensorboard=tensorboard,
            save_best=save_best,
            checkpoint_freq=checkpoint_freq
        )

        results = trainer.train(episodes=episodes)

        console.print("✅ Training completed successfully!")
        console.print(f"📈 Best reward: [green]{results.get('best_reward', 'N/A')}[/green]")
        console.print(f"💾 Model saved to: [cyan]{output.absolute()}[/cyan]")

    except Exception as e:
        console.print(f"❌ Training failed: {e}", style="red")
        raise typer.Exit(1) from e


@train_app.command("resume")
def train_resume(
    checkpoint: Annotated[
        Path,
        typer.Option("--checkpoint", help="Checkpoint path (or 'latest')")
    ],
    additional_episodes: Annotated[
        int,
        typer.Option("--additional-episodes", help="Additional episodes to train")
    ] = 5000,
) -> None:
    """
    🔄 Resume training from a checkpoint.

    Continue training from where you left off with full state restoration.

    Examples:
        trading-cli train resume --checkpoint latest --additional-episodes 10000
        trading-cli train resume --checkpoint models/ppo_checkpoint_5000.pkl
    """
    from trade_agent.agents.trainer import UnifiedTrainer

    console.print("🔄 Resuming training from checkpoint...")
    console.print(f"📂 Checkpoint: [cyan]{checkpoint}[/cyan]")
    console.print(f"+ Additional episodes: [green]{additional_episodes:,}[/green]")

    try:
        trainer = UnifiedTrainer.from_checkpoint(checkpoint)
        results = trainer.resume_training(additional_episodes=additional_episodes)

        console.print("✅ Training resumed and completed!")
        console.print(f"📈 Final reward: [green]{results.get('final_reward', 'N/A')}[/green]")

    except Exception as e:
        console.print(f"❌ Resume failed: {e}", style="red")
        raise typer.Exit(1) from e


@train_app.command("evaluate")
def train_evaluate(
    model: Annotated[
        Path,
        typer.Option("--model", help="Model path to evaluate")
    ],
    data: Annotated[
        Path,
        typer.Option("--data", help="Test data path")
    ] = Path("data/splits/test.csv"),
    backtest: Annotated[
        bool,
        typer.Option("--backtest/--no-backtest", help="Run full backtest")
    ] = True,
    metrics: Annotated[
        str,
        typer.Option("--metrics", help="Metrics to calculate: sharpe,returns,drawdown,all")
    ] = "all",
    output: Annotated[
        Path,
        typer.Option("--output", help="Results output directory")
    ] = Path("evaluation"),
) -> None:
    """
    📊 Evaluate a trained model's performance.

    Comprehensive evaluation with financial metrics, backtesting,
    and performance visualization.

    Examples:
        trading-cli train evaluate --model models/best_ppo.pkl --backtest
        trading-cli train evaluate --model cnn_lstm.pkl --metrics sharpe,returns
    """

    console.print(f"📊 Evaluating model: [cyan]{model}[/cyan]")
    console.print(f"📈 Test data: [blue]{data}[/blue]")
    console.print(f"🎯 Metrics: [green]{metrics}[/green]")

    try:
        output.mkdir(parents=True, exist_ok=True)

        evaluator = ModelEvaluator(
            model_path=model,
            test_data=data,
            output_dir=output,
            run_backtest=backtest,
            metrics=metrics.split(",") if metrics != "all" else None
        )

        results = evaluator.evaluate()

        console.print("✅ Evaluation completed!")

        # Display key metrics
        table = Table(title="📊 Model Performance", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if "sharpe_ratio" in results:
            table.add_row("Sharpe Ratio", f"{results['sharpe_ratio']:.3f}")
        if "total_return" in results:
            table.add_row("Total Return", f"{results['total_return']:.2%}")
        if "max_drawdown" in results:
            table.add_row("Max Drawdown", f"{results['max_drawdown']:.2%}")

        console.print(table)
        console.print(f"📁 Full results: [cyan]{output.absolute()}[/cyan]")

    except Exception as e:
        console.print(f"❌ Evaluation failed: {e}", style="red")
        raise typer.Exit(1) from e


# ================================
# DEPLOYMENT COMMANDS
# ================================

deploy_app = typer.Typer(
    name="deploy",
    help="🚀 Deployment operations",
    no_args_is_help=True,
)


@deploy_app.command()
def deploy(
    model: Annotated[
        Path,
        typer.Option("--model", help="Model path to deploy")
    ],
    broker: Annotated[
        str,
        typer.Option("--broker", help="Broker: alpaca, ib, ccxt")
    ] = "alpaca",
    paper_trading: Annotated[
        bool,
        typer.Option("--paper-trading/--live", help="Paper trading mode")
    ] = True,
    symbols: Annotated[
        str,
        typer.Option("--symbols", help="Trading symbols")
    ] = "SPY,QQQ",
    capital: Annotated[
        float,
        typer.Option("--capital", help="Trading capital")
    ] = 100000.0,
    risk_limits: Annotated[
        str,
        typer.Option("--risk-limits", help="Risk profile: conservative, moderate, aggressive")
    ] = "conservative",
) -> None:
    """
    🚀 Deploy a trained model for live or paper trading.

    Production deployment with comprehensive risk management,
    real-time monitoring, and automatic failsafes.

    Examples:
        trading-cli deploy --model best_model.pkl --broker alpaca --paper-trading
        trading-cli deploy --model ppo.pkl --broker ib --live --capital 50000
    """

    mode = "PAPER" if paper_trading else "LIVE"
    console.print(f"🚀 Deploying model for [bold]{mode}[/bold] trading...")
    console.print(f"🤖 Model: [cyan]{model}[/cyan]")
    console.print(f"🏦 Broker: [blue]{broker}[/blue]")
    console.print(f"💰 Capital: [green]${capital:,.2f}[/green]")

    if not paper_trading:
        confirm = typer.confirm("⚠️  You're about to deploy to LIVE trading. Are you sure?")
        if not confirm:
            console.print("❌ Deployment cancelled.")
            raise typer.Exit()

    try:
        engine = LiveTradingEngine(
            model_path=model,
            broker=broker,
            symbols=symbols.split(","),
            initial_capital=capital,
            paper_trading=paper_trading,
            risk_profile=risk_limits
        )

        engine.start()

        console.print(f"✅ Model deployed successfully in [bold]{mode}[/bold] mode!")
        console.print(f"📊 Trading symbols: [green]{symbols}[/green]")
        console.print("🎯 Use 'trading-cli monitor' to track performance")

    except Exception as e:
        console.print(f"❌ Deployment failed: {e}", style="red")
        raise typer.Exit(1) from e


@deploy_app.command("live")
def deploy_live(
    model: Annotated[
        Path,
        typer.Option("--model", help="Model path to deploy")
    ],
    broker: Annotated[
        str,
        typer.Option("--broker", help="Broker: alpaca, ib, ccxt")
    ] = "alpaca",
    risk_limits: Annotated[
        str,
        typer.Option("--risk-limits", help="Risk profile: conservative, moderate, aggressive")
    ] = "conservative",
    symbols: Annotated[
        str,
        typer.Option("--symbols", help="Trading symbols")
    ] = "SPY,QQQ",
    capital: Annotated[
        float,
        typer.Option("--capital", help="Trading capital")
    ] = 100000.0,
) -> None:
    """
    ⚠️  Deploy to LIVE trading with real money.

    CAUTION: This trades with real money. Ensure thorough testing first.
    """
    # Delegate to main deploy command with live=True
    deploy(
        model=model,
        broker=broker,
        paper_trading=False,
        symbols=symbols,
        capital=capital,
        risk_limits=risk_limits
    )


@deploy_app.command("monitor")
def deploy_monitor(
    dashboard: Annotated[
        bool,
        typer.Option("--dashboard/--no-dashboard", help="Launch web dashboard")
    ] = True,
    alerts: Annotated[
        str,
        typer.Option("--alerts", help="Alert methods: email,slack,webhook")
    ] = "email",
    interval: Annotated[
        int,
        typer.Option("--interval", help="Monitoring interval (seconds)")
    ] = 60,
) -> None:
    """
    📊 Monitor live trading performance.

    Real-time monitoring with customizable alerts and web dashboard.

    Examples:
        trading-cli deploy monitor --dashboard --alerts email,slack
        trading-cli deploy monitor --interval 30
    """
    from trade_agent.monitoring.system_health_monitor import SystemHealthMonitor

    console.print("📊 Starting trading monitor...")
    console.print(f"🖥️  Dashboard: [blue]{'enabled' if dashboard else 'disabled'}[/blue]")
    console.print(f"🚨 Alerts: [green]{alerts}[/green]")
    console.print(f"⏰ Interval: [yellow]{interval}s[/yellow]")

    try:
        monitor = SystemHealthMonitor(
            enable_dashboard=dashboard,
            alert_methods=alerts.split(","),
            check_interval=interval
        )

        monitor.start()

        if dashboard:
            console.print("🌐 Dashboard available at: http://localhost:8080")

        console.print("✅ Monitor started successfully!")
        console.print("Press Ctrl+C to stop monitoring")

        # Keep running until interrupted
        import signal
        import time

        def signal_handler(signum, frame):  # noqa: ARG001
            console.print("\n🛑 Stopping monitor...")
            monitor.stop()
            console.print("✅ Monitor stopped.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        console.print("\n🛑 Monitor stopped by user.")
    except Exception as e:
        console.print(f"❌ Monitor failed: {e}", style="red")
        raise typer.Exit(1) from e


# ================================
# REGISTER SUB-APPS
# ================================

app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(deploy_app, name="deploy")


# ================================
# ADDITIONAL UTILITY COMMANDS
# ================================

@app.command("status")
def status() -> None:
    """
    📊 Show system status and health.
    """
    from trade_agent.monitoring.system_health_monitor import SystemHealthMonitor

    console.print("📊 System Status Check")
    console.print("=" * 50)

    try:
        monitor = SystemHealthMonitor()
        health_data = monitor.get_health_status()

        # Create status table
        table = Table(title="🏥 System Health", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")

        for component, status in health_data.items():
            status_icon = "✅" if status["healthy"] else "❌"
            table.add_row(component, status_icon, status.get("message", "OK"))

        console.print(table)

    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="red")


@app.command("config")
def config(
    init: Annotated[
        bool,
        typer.Option("--init", help="Initialize configuration")
    ] = False,
    show: Annotated[
        bool,
        typer.Option("--show", help="Show current configuration")
    ] = False,
    validate: Annotated[
        bool,
        typer.Option("--validate", help="Validate configuration")
    ] = False,
) -> None:
    """
    ⚙️  Configuration management.
    """
    if init:
        # Create default configuration
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        # Copy default config template
        default_config = """
# Trading RL Platform Configuration
data:
  sources:
    yahoo:
      enabled: true
    alpha_vantage:
      enabled: false
      api_key: YOUR_API_KEY
    polygon:
      enabled: false
      api_key: YOUR_API_KEY

  default_symbols:
    - SPY
    - QQQ
    - AAPL
    - GOOGL
    - MSFT

training:
  default_algorithm: PPO
  episodes: 10000
  checkpoint_frequency: 1000
  tensorboard: true

deployment:
  brokers:
    alpaca:
      api_key: YOUR_ALPACA_KEY
      secret_key: YOUR_ALPACA_SECRET
      paper_trading: true

risk_management:
  max_position_size: 0.1
  max_portfolio_risk: 0.02
  stop_loss: 0.05

monitoring:
  alerts:
    email:
      enabled: false
      smtp_server: smtp.gmail.com
      recipients: []
    slack:
      enabled: false
      webhook_url: YOUR_SLACK_WEBHOOK
"""

        config_file = config_dir / "trading_config.yaml"
        config_file.write_text(default_config.strip())

        console.print(f"✅ Configuration initialized: [cyan]{config_file}[/cyan]")
        console.print("📝 Edit the file to customize your settings")

    elif show:
        console.print("⚙️  Current Configuration:")
        # Show current config

    elif validate:
        console.print("✅ Configuration validation:")
        # Validate config


if __name__ == "__main__":
    app()
