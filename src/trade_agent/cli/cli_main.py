"""
Main CLI application and shared utilities.

This module contains the root Typer app and common functionality used across
all CLI modules.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table

from trade_agent.config import get_logger, load_settings
from trade_agent.logging_conf import setup_logging_for_typer
from trade_agent.utils.cache_manager import CacheManager

# Type variable for decorator functions
F = TypeVar("F", bound=Callable[..., Any])

# Module-level constants for typer defaults to fix B008 errors
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_FORCE_REBUILD = False
DEFAULT_PARALLEL = True
DEFAULT_STANDARDIZED_OUTPUT = Path("data/processed")
DEFAULT_STANDARDIZATION_METHOD = "robust"
DEFAULT_PIPELINE_OUTPUT = Path("data/processed")
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_CNN_LSTM_OUTPUT = Path("models/cnn_lstm")
DEFAULT_GPU = False
DEFAULT_MIXED_PRECISION = True
DEFAULT_TIMESTEPS = 1000000
DEFAULT_RL_OUTPUT = Path("models/rl")
DEFAULT_NUM_WORKERS = 4
DEFAULT_HYBRID_OUTPUT = Path("models/hybrid")
DEFAULT_N_TRIALS = 100
DEFAULT_OPTIMIZATION_OUTPUT = Path("optimization_results")
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION = 0.001
DEFAULT_SLIPPAGE = 0.0001
DEFAULT_BACKTEST_OUTPUT = Path("backtest_results")
DEFAULT_EVALUATION_OUTPUT = Path("evaluation_results")
DEFAULT_COMPARISON_OUTPUT = Path("comparison_results")
DEFAULT_REPORT_FORMAT = "html"
DEFAULT_REPORTS_OUTPUT = Path("reports")
DEFAULT_PAPER_TRADING = True
DEFAULT_TRADING_CAPITAL = 100000.0
DEFAULT_PAPER_SYMBOLS = "AAPL,GOOGL,MSFT"
DEFAULT_PAPER_DURATION = "1d"
DEFAULT_VERBOSE = 0
DEFAULT_DATA_OUTPUT = Path("data")
DEFAULT_DATA_SOURCE = "yfinance"
DEFAULT_REFRESH_DAYS = 1
DEFAULT_MONITOR_METRICS = "all"
DEFAULT_MONITOR_INTERVAL = 60

# Additional module-level constants for remaining B008 errors
DEFAULT_CONFIG_FILE: Path | None = None
DEFAULT_ENV_FILE: Path | None = None
DEFAULT_START_DATE: str | None = None
DEFAULT_END_DATE: str | None = None
DEFAULT_OUTPUT_DIR_NONE: Path | None = None
DEFAULT_SOURCE: str | None = None
DEFAULT_TIMEFRAME: str | None = None
DEFAULT_FORCE = False
DEFAULT_SYMBOLS_STR: str | None = None
DEFAULT_DAYS = 1
DEFAULT_SYMBOLS_NONE: str | None = None
DEFAULT_SESSION_ID_NONE: str | None = None
DEFAULT_ALL_SESSIONS_FALSE = False
DEFAULT_DETAILED_FALSE = False
DEFAULT_METRICS_ALL = "all"
DEFAULT_INTERVAL_60 = 60
DEFAULT_AGENT_TYPE: str | None = None
DEFAULT_RAY_ADDRESS: str | None = None
DEFAULT_CNN_LSTM_PATH: Path | None = None
DEFAULT_RL_PATH: Path | None = None
DEFAULT_DATA_PATH: Path | None = None
DEFAULT_MODEL_PATH: Path | None = None
DEFAULT_POLICY: str | None = None
DEFAULT_MODELS: str | None = None
DEFAULT_RESULTS_PATH: Path | None = None

# Initialize console for rich output
console = Console()
logger = get_logger(__name__)

# Global state
verbose_count: int = 0


def get_config_manager() -> Any:
    """Get configuration manager with minimal fallback."""
    try:
        # Try production config first, for proper typing
        settings = load_settings()
        return settings
    except Exception:
        # Fallback to simple object for basic functionality
        logger.warning("Using minimal config fallback")

        class MinimalSettings:
            def __init__(self) -> None:
                data = type("Data", (), {"symbols": ["AAPL", "GOOGL", "MSFT"]})()
                model = type("Model", (), {"algorithm": "ppo"})()
                live = type("Live", (), {"max_position_size": 0.1, "exchange": "alpaca", "paper_trading": True})()
                infrastructure = type("Infrastructure", (), {"max_workers": 4})()

        _settings = MinimalSettings()
    return _settings


# Root app
app = typer.Typer(
    name="trade-agent",
    help="Production-grade trading RL agent with CNN+LSTM integration",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)


@app.callback()
def main(
    config: Annotated[Path | None, typer.Option("--config", help="Configuration file path")] = DEFAULT_CONFIG_FILE,
    env_file: Annotated[Path | None, typer.Option("--env", help="Environment variables file")] = DEFAULT_ENV_FILE,
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True, help="Increase verbosity")] = DEFAULT_VERBOSE,
) -> None:
    """
    Trading RL Agent - Production-grade reinforcement learning for trading.

    A comprehensive trading system that combines CNN+LSTM models with deep RL agents
    for sophisticated algorithmic trading strategies.

    Features:
    - CNN+LSTM for pattern recognition
    - SAC, TD3, PPO RL agents
    - Real-time data processing
    - Risk management
    - Live trading integration

    Examples:
        train-agent data pipeline --symbols AAPL,GOOGL,MSFT
        train-agent train cnn-lstm data/processed/dataset.csv
        train-agent backtest evaluate --model-path models/hybrid_model.zip
        train-agent trade start --paper --symbols AAPL,GOOGL
    """
    global verbose_count
    verbose_count = verbose

    # Configure logging based on verbosity
    setup_logging_for_typer(verbose)

    if verbose >= 1:
        console.print(f"[dim]Verbosity level: {verbose}[/dim]")

    if config:
        console.print(f"[dim]Using config file: {config}[/dim]")

    if env_file:
        console.print(f"[dim]Using env file: {env_file}[/dim]")


@app.command()
def version() -> None:
    """Show version information."""
    try:
        import importlib.metadata
        version = importlib.metadata.version("trade-agent")
    except Exception:
        version = "0.2.0-dev"

    console.print(f"[bold blue]Trading RL Agent[/bold blue] version [green]{version}[/green]")
    console.print("A production-grade reinforcement learning trading system")


@app.command()
def info() -> None:
    """Show system information and configuration."""
    console.print("[bold blue]Trading RL Agent - System Information[/bold blue]")

    # Create information table
    table = Table(title="System Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    table.add_column("Details", style="white")

    # Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    table.add_row("Python", "✓", python_version)

    # Configuration
    try:
        get_config_manager()
        table.add_row("Configuration", "✓", "Loaded successfully")
    except Exception as e:
        table.add_row("Configuration", "✗", f"Error: {e}")

    # Cache manager
    try:
        cache_manager = CacheManager()
        cache_stats = cache_manager.get_cache_stats()
        table.add_row("Cache", "✓", f"Size: {cache_stats.get('size', 'Unknown')}")
    except Exception as e:
        table.add_row("Cache", "✗", f"Error: {e}")

    console.print(table)


# Import and register sub-apps
def register_sub_apps() -> None:
    """Register all sub-application modules."""
    from . import cli_backtest, cli_data, cli_trade, cli_train

    # Add sub-apps to root app
    app.add_typer(cli_data.data_app, name="data", help="Data pipeline operations")
    app.add_typer(cli_train.train_app, name="train", help="Model training operations")
    app.add_typer(cli_backtest.backtest_app, name="backtest", help="Backtesting operations")
    app.add_typer(cli_trade.trade_app, name="trade", help="Live trading operations")


# Register sub-apps on module import
register_sub_apps()

# Export sub-apps for backward compatibility with tests
def _get_sub_apps():
    """Get sub-apps for backward compatibility."""
    from . import cli_backtest, cli_data, cli_trade, cli_train
    return {
        "data_app": cli_data.data_app,
        "train_app": cli_train.train_app,
        "backtest_app": cli_backtest.backtest_app,
        "trade_app": cli_trade.trade_app,
    }

# Make sub-apps available at module level for backward compatibility
_sub_apps = _get_sub_apps()
data_app = _sub_apps["data_app"]
train_app = _sub_apps["train_app"]
backtest_app = _sub_apps["backtest_app"]
trade_app = _sub_apps["trade_app"]
