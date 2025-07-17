"""
Unified Typer CLI for Trading RL Agent.

This module provides a comprehensive command-line interface for all trading RL agent operations:
- Dataset management and download
- Model training (CNN+LSTM and RL agents)
- Backtesting and evaluation
- Live trading execution
"""

import sys
from pathlib import Path
from typing import Any, Callable, TypeVar

import typer
from rich.console import Console
from rich.table import Table

# Type variable for decorator functions
F = TypeVar("F", bound=Callable[..., Any])

from trading_rl_agent.logging_conf import get_logger, setup_logging_for_typer

# Module-level constants for typer defaults to fix B008 errors
DEFAULT_OUTPUT_DIR = Path("outputs/datasets")
DEFAULT_FORCE_REBUILD = False
DEFAULT_PARALLEL = True
DEFAULT_STANDARDIZED_OUTPUT = Path("data/standardized")
DEFAULT_STANDARDIZATION_METHOD = "robust"
DEFAULT_PIPELINE_OUTPUT = Path("data/pipeline")
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

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_settings, load_settings

# Initialize console for rich output
console = Console()
logger = get_logger(__name__)

# Global state
verbose_count: int = 0
_settings = None


def get_config_manager() -> Any:
    """Get or create config manager instance."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


# Root app
app = typer.Typer(
    name="trading-rl-agent",
    help="Production-grade trading RL agent with CNN+LSTM integration",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# Sub-apps
data_app = typer.Typer(
    name="data",
    help="Data pipeline operations: download, process, standardize",
    rich_markup_mode="rich",
)

train_app = typer.Typer(
    name="train",
    help="Model training operations: CNN+LSTM, RL agents, hybrid models",
    rich_markup_mode="rich",
)

backtest_app = typer.Typer(
    name="backtest",
    help="Backtesting operations: strategy evaluation, performance analysis",
    rich_markup_mode="rich",
)

trade_app = typer.Typer(
    name="trade",
    help="Live trading operations: start, stop, monitor trading sessions",
    rich_markup_mode="rich",
)

# Add sub-apps to root app
app.add_typer(data_app, help="Data pipeline operations")
app.add_typer(train_app, help="Model training operations")
app.add_typer(backtest_app, help="Backtesting operations")
app.add_typer(trade_app, help="Live trading operations")

# New sub-apps for additional features
features_app = typer.Typer(
    name="features",
    help="Feature engineering operations: technical indicators, candlestick patterns, market regimes",
    rich_markup_mode="rich",
)

advanced_data_app = typer.Typer(
    name="advanced-data",
    help="Advanced data operations: optimized datasets, synthetic data, alternative data",
    rich_markup_mode="rich",
)

advanced_train_app = typer.Typer(
    name="advanced-train",
    help="Advanced training operations: optimized training, mixed precision, data augmentation",
    rich_markup_mode="rich",
)

nlp_app = typer.Typer(
    name="nlp",
    help="Natural language processing operations: news analysis, sentiment analysis",
    rich_markup_mode="rich",
)

monitor_app = typer.Typer(
    name="monitor",
    help="Monitoring operations: dashboards, alerts, system health",
    rich_markup_mode="rich",
)

# Add new sub-apps to root app
app.add_typer(features_app, help="Feature engineering operations")
app.add_typer(advanced_data_app, help="Advanced data operations")
app.add_typer(advanced_train_app, help="Advanced training operations")
app.add_typer(nlp_app, help="Natural language processing operations")
app.add_typer(monitor_app, help="Monitoring operations")


@app.callback()
def main(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    verbose: int = DEFAULT_VERBOSE,
    env_file: Path | None = DEFAULT_ENV_FILE,
) -> None:
    """
    Trading RL Agent - Production-grade live trading system.

    A hybrid reinforcement learning trading system that combines CNN+LSTM supervised learning
    with deep RL optimization for algorithmic trading.
    """
    config_file = (
        config_file
        if config_file is not None
        else typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c", help="Path to configuration file")
    )
    verbose = (
        verbose
        if verbose is not None
        else typer.Option(
            DEFAULT_VERBOSE,
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (use multiple times for more detail)",
        )
    )
    env_file = (
        env_file
        if env_file is not None
        else typer.Option(DEFAULT_ENV_FILE, "--env-file", help="Path to environment file (.env)")
    )

    global verbose_count, _settings

    # Set global verbose count
    verbose_count = verbose

    # Setup logging with verbose level
    setup_logging_for_typer(verbose)

    # Load environment file if provided
    if env_file:
        if not env_file.exists():
            console.print(f"[red]Environment file not found: {env_file}[/red]")
            raise typer.Exit(1)

        try:
            from dotenv import load_dotenv

            load_dotenv(env_file)
            console.print(f"[green]Loaded environment from: {env_file}[/green]")
        except ImportError as err:
            console.print("[red]python-dotenv not installed. Install with: pip install python-dotenv[/red]")
            raise typer.Exit(1) from err

    # Load configuration if provided
    if config_file:
        if not config_file.exists():
            console.print(f"[red]Configuration file not found: {config_file}[/red]")
            raise typer.Exit(1)

        try:
            _settings = load_settings(config_file, env_file)
            console.print(f"[green]Loaded configuration from: {config_file}[/green]")
        except Exception as e:
            console.print(f"[red]Error loading configuration: {e}[/red]")
            raise typer.Exit(1) from e
    else:
        _settings = get_settings()

    # Set up logging based on configuration
    if _settings:
        setup_logging_for_typer(verbose, _settings.monitoring.log_level)


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__

    table = Table(title="Trading RL Agent")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")

    table.add_row("Trading RL Agent", __version__)
    table.add_row("Python", sys.version.split()[0])

    console.print(table)


@app.command()
def info() -> None:
    """Show system information and configuration."""
    settings = get_config_manager()

    table = Table(title="System Information")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Environment", settings.environment)
    table.add_row("Debug Mode", str(settings.debug))
    table.add_row("Data Source", settings.data.primary_source)
    table.add_row("Symbols", ", ".join(settings.data.symbols))
    table.add_row("Agent Type", settings.agent.agent_type)
    table.add_row("Risk Management", "Enabled" if settings.risk.max_position_size > 0 else "Disabled")
    table.add_row("Execution Broker", settings.execution.broker)
    table.add_row("Paper Trading", str(settings.execution.paper_trading))

    console.print(table)


# ============================================================================
# DATA SUB-APP COMMANDS
# ============================================================================


@data_app.command()
def download_all(
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR_NONE,
    source: str | None = DEFAULT_SOURCE,
    timeframe: str | None = DEFAULT_TIMEFRAME,
    parallel: bool = DEFAULT_PARALLEL,
    force: bool = DEFAULT_FORCE,
) -> None:
    """
    Download all configured datasets.

    Downloads data for all symbols configured in Settings.data.symbols.
    Uses Settings.data values with CLI option overrides.
    """
    try:
        # Load configuration
        settings = get_config_manager()

        # Use config values with CLI overrides
        symbols = settings.data.symbols
        output_dir = output_dir or Path(settings.data.data_path)
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        console.print("[green]Starting download of all configured datasets[/green]")
        console.print(f"[cyan]Symbols: {', '.join(symbols)}[/cyan]")
        console.print(f"[cyan]Date range: {start_date} to {end_date}[/cyan]")
        console.print(f"[cyan]Source: {source}[/cyan]")
        console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]")

        # Import and use actual download functions
        from .data.parallel_data_fetcher import fetch_data_parallel, fetch_data_with_retry

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Choose download method based on parallel flag
        if parallel:
            console.print("[yellow]Using parallel data fetching...[/yellow]")
            data_dict = fetch_data_parallel(
                symbols=symbols,
                start_date=start_date or settings.data.start_date,
                end_date=end_date or settings.data.end_date,
                interval=timeframe,
                cache_dir=str(output_dir / "cache"),
                max_workers=settings.infrastructure.max_workers,
            )
        else:
            console.print("[yellow]Using sequential data fetching...[/yellow]")
            data_dict = fetch_data_with_retry(
                symbols=symbols,
                start_date=start_date or settings.data.start_date,
                end_date=end_date or settings.data.end_date,
                interval=timeframe,
                cache_dir=str(output_dir / "cache"),
                max_retries=3,
            )

        # Log results
        total_rows = 0
        successful_downloads = 0

        for symbol, data in data_dict.items():
            if not data.empty:
                successful_downloads += 1
                rows = len(data)
                total_rows += rows
                console.print(f"[green]✓ {symbol}: {rows:,} rows[/green]")

                # Save to output directory
                output_file = output_dir / f"{symbol}_{start_date}_{end_date}_{timeframe}.parquet"
                data.to_parquet(output_file)
                console.print(f"[blue]  Saved to: {output_file}[/blue]")
            else:
                console.print(f"[red]✗ {symbol}: No data available[/red]")

        console.print("\n[green]Download completed![/green]")
        console.print(f"[cyan]Successful downloads: {successful_downloads}/{len(symbols)}[/cyan]")
        console.print(f"[cyan]Total rows downloaded: {total_rows:,}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during download: {e}[/red]")
        logger.error(f"Download failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command()
def symbols(
    symbols: str | None = DEFAULT_SYMBOLS_STR,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR_NONE,
    source: str | None = DEFAULT_SOURCE,
    timeframe: str | None = DEFAULT_TIMEFRAME,
    parallel: bool = DEFAULT_PARALLEL,
) -> None:
    """
    Download data for specific comma-separated instruments.

    Downloads data for the specified symbols using Settings.data values
    with CLI option overrides.
    """
    try:
        # Load configuration
        settings = get_config_manager()

        # Parse symbols
        if symbols is None:
            symbol_list = settings.data.symbols
        else:
            symbol_list = [s.strip() for s in symbols.split(",")]

        # Use config values with CLI overrides
        start_date = start_date or settings.data.start_date
        end_date = end_date or settings.data.end_date
        output_dir = output_dir or Path(settings.data.data_path)
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        console.print("[green]Starting download for specific symbols[/green]")
        console.print(f"[cyan]Symbols: {', '.join(symbol_list)}[/cyan]")
        console.print(f"[cyan]Date range: {start_date} to {end_date}[/cyan]")
        console.print(f"[cyan]Source: {source}[/cyan]")
        console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]")

        # Import and use actual download functions
        from .data.parallel_data_fetcher import fetch_data_parallel, fetch_data_with_retry

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Choose download method based on parallel flag
        if parallel:
            console.print("[yellow]Using parallel data fetching...[/yellow]")
            data_dict = fetch_data_parallel(
                symbols=symbol_list,
                start_date=start_date or settings.data.start_date,
                end_date=end_date or settings.data.end_date,
                interval=timeframe or settings.data.timeframe,
                cache_dir=str(output_dir / "cache"),
                max_workers=settings.infrastructure.max_workers,
            )
        else:
            console.print("[yellow]Using sequential data fetching...[/yellow]")
            data_dict = fetch_data_with_retry(
                symbols=symbol_list,
                start_date=start_date or settings.data.start_date,
                end_date=end_date or settings.data.end_date,
                interval=timeframe or settings.data.timeframe,
                cache_dir=str(output_dir / "cache"),
                max_retries=3,
            )

        # Log results
        total_rows = 0
        successful_downloads = 0

        for symbol, data in data_dict.items():
            if not data.empty:
                successful_downloads += 1
                rows = len(data)
                total_rows += rows
                console.print(f"[green]✓ {symbol}: {rows:,} rows[/green]")

                # Save to output directory
                output_file = output_dir / f"{symbol}_{start_date}_{end_date}_{timeframe}.parquet"
                data.to_parquet(output_file)
                console.print(f"[blue]  Saved to: {output_file}[/blue]")
            else:
                console.print(f"[red]✗ {symbol}: No data available[/red]")

        console.print("\n[green]Download completed![/green]")
        console.print(f"[cyan]Successful downloads: {successful_downloads}/{len(symbol_list)}[/cyan]")
        console.print(f"[cyan]Total rows downloaded: {total_rows:,}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during download: {e}[/red]")
        logger.error(f"Download failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command()
def refresh(
    days: int = DEFAULT_DAYS,
    symbols: str | None = DEFAULT_SYMBOLS_NONE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR_NONE,
    source: str | None = DEFAULT_SOURCE,
    timeframe: str | None = DEFAULT_TIMEFRAME,
    parallel: bool = DEFAULT_PARALLEL,
) -> None:
    """
    Re-download data if older than N days.

    Checks cache age and re-downloads data that is older than the specified number of days.
    Uses Settings.data values with CLI option overrides.
    """
    try:
        # Load configuration
        settings = get_config_manager()

        # Determine symbols to refresh
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
        else:
            symbol_list = settings.data.symbols

        # Use config values with CLI overrides
        start_date = settings.data.start_date
        end_date = settings.data.end_date
        output_dir = output_dir or Path(settings.data.data_path)
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        console.print(f"[green]Checking data freshness and refreshing if older than {days} days[/green]")
        console.print(f"[cyan]Symbols: {', '.join(symbol_list)}[/cyan]")
        console.print(f"[cyan]Cache TTL: {days} days[/cyan]")
        console.print(f"[cyan]Source: {source}[/cyan]")
        console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]")

        # Import and use actual download functions with cache checking
        import time

        from .data.parallel_data_fetcher import ParallelDataManager

        # Create data manager with custom TTL
        cache_ttl_hours = days * 24
        data_manager = ParallelDataManager(
            cache_dir=str(output_dir / "cache"),
            ttl_hours=cache_ttl_hours,
            max_workers=settings.infrastructure.max_workers if parallel else 1,
        )

        # Check cache and download fresh data
        console.print("[yellow]Checking cache and downloading fresh data...[/yellow]")
        data_dict = data_manager.fetch_multiple_symbols(
            symbols=symbol_list, start_date=start_date, end_date=end_date, interval=timeframe, show_progress=True
        )

        # Log results
        total_rows = 0
        successful_downloads = 0
        refreshed_count = 0

        for symbol, data in data_dict.items():
            if not data.empty:
                successful_downloads += 1
                rows = len(data)
                total_rows += rows

                # Check if this was a refresh (cache miss)
                cache_file = output_dir / "cache" / f"{symbol}_{start_date}_{end_date}_{timeframe}.parquet"
                if cache_file.exists():
                    file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                    if file_age_hours > cache_ttl_hours:
                        refreshed_count += 1
                        console.print(f"[yellow]🔄 {symbol}: {rows:,} rows (refreshed)[/yellow]")
                    else:
                        console.print(f"[green]✓ {symbol}: {rows:,} rows (cached)[/green]")
                else:
                    refreshed_count += 1
                    console.print(f"[yellow]🔄 {symbol}: {rows:,} rows (new)[/yellow]")

                # Save to output directory
                output_file = output_dir / f"{symbol}_{start_date}_{end_date}_{timeframe}.parquet"
                data.to_parquet(output_file)
                console.print(f"[blue]  Saved to: {output_file}[/blue]")
            else:
                console.print(f"[red]✗ {symbol}: No data available[/red]")

        # Get cache statistics
        cache_stats = data_manager.get_cache_stats()

        console.print("\n[green]Refresh completed![/green]")
        console.print(f"[cyan]Successful downloads: {successful_downloads}/{len(symbol_list)}[/cyan]")
        console.print(f"[cyan]Refreshed datasets: {refreshed_count}[/cyan]")
        console.print(f"[cyan]Total rows: {total_rows:,}[/cyan]")
        console.print(f"[cyan]Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during refresh: {e}[/red]")
        logger.error(f"Refresh failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command()
def download(
    symbols: str | None = DEFAULT_SYMBOLS_STR,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR_NONE,
    source: str | None = DEFAULT_SOURCE,
) -> None:
    """
    Download market data for specified symbols.

    Uses the DataPipeline.download_data() function from src/trading_rl_agent/data/pipeline.py
    to fetch historical market data from various sources.
    """
    console.print(f"[blue]PLACEHOLDER: Would download data for {symbols} from {start_date} to {end_date}[/blue]")
    console.print("[blue]Target module: src/trading_rl_agent/data/pipeline.py - DataPipeline.download_data()[/blue]")


@data_app.command()
def process(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force_rebuild: bool = DEFAULT_FORCE_REBUILD,
    parallel: bool = DEFAULT_PARALLEL,
) -> None:
    """Process and standardize downloaded data."""
    if config_file is None:
        config_file = Path("config.yaml")  # Provide a default config file
    # ... rest of function implementation


@data_app.command()
def standardize(
    input_path: Path | None = DEFAULT_CONFIG_FILE,
    output_path: Path = DEFAULT_STANDARDIZED_OUTPUT,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
) -> None:
    """Standardize data format."""
    if input_path is None:
        input_path = Path("data/raw")  # Provide a default input path
    # ... rest of function implementation


@data_app.command()
def pipeline(
    config_path: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = DEFAULT_PIPELINE_OUTPUT,
) -> None:
    """Run complete data pipeline."""
    if config_path is None:
        config_path = Path("config.yaml")  # Provide a default config path
    # ... rest of function implementation


# ============================================================================
# TRAIN SUB-APP COMMANDS
# ============================================================================


@train_app.command()
def cnn_lstm(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    output_dir: Path = DEFAULT_CNN_LSTM_OUTPUT,
    gpu: bool = DEFAULT_GPU,
    mixed_precision: bool = DEFAULT_MIXED_PRECISION,
) -> None:
    """
    Train CNN+LSTM models for pattern recognition.

    Uses the OptimizedTrainingManager.train() function from
    src/trading_rl_agent/training/optimized_trainer.py to train
    CNN+LSTM models with advanced optimizations.
    """
    console.print(f"[blue]PLACEHOLDER: Would train CNN+LSTM model for {epochs} epochs[/blue]")
    console.print(
        "[blue]Target module: src/trading_rl_agent/training/optimized_trainer.py - OptimizedTrainingManager.train()[/blue]"
    )


@train_app.command()
def rl(
    agent_type: str | None = DEFAULT_AGENT_TYPE,
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    timesteps: int = DEFAULT_TIMESTEPS,
    output_dir: Path = DEFAULT_RL_OUTPUT,
    ray_address: str | None = DEFAULT_RAY_ADDRESS,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Train reinforcement learning agents.

    Uses the Trainer.train() function from src/trading_rl_agent/agents/trainer.py
    to train RL agents (PPO, SAC, TD3) using Ray RLlib.
    """
    console.print(f"[blue]PLACEHOLDER: Would train {agent_type} agent for {timesteps} timesteps[/blue]")
    console.print("[blue]Target module: src/trading_rl_agent/agents/trainer.py - Trainer.train()[/blue]")


@train_app.command()
def hybrid(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    cnn_lstm_path: Path | None = DEFAULT_CNN_LSTM_PATH,
    rl_path: Path | None = DEFAULT_RL_PATH,
    output_dir: Path = DEFAULT_HYBRID_OUTPUT,
) -> None:
    """
    Train hybrid models combining CNN+LSTM with RL agents.

    Uses the HybridAgent class from src/trading_rl_agent/agents/hybrid.py
    to create and train hybrid models that combine supervised and RL components.
    """
    console.print("[blue]PLACEHOLDER: Would train hybrid model combining CNN+LSTM and RL[/blue]")
    console.print("[blue]Target module: src/trading_rl_agent/agents/hybrid.py - HybridAgent[/blue]")


@train_app.command()
def hyperopt(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    n_trials: int = DEFAULT_N_TRIALS,
    output_dir: Path = DEFAULT_OPTIMIZATION_OUTPUT,
) -> None:
    """
    Perform hyperparameter optimization.

    Uses Optuna-based optimization from train.py to find optimal
    hyperparameters for models and training configurations.
    """
    console.print(f"[blue]PLACEHOLDER: Would run hyperparameter optimization with {n_trials} trials[/blue]")
    console.print("[blue]Target module: train.py - hyperparameter optimization functions[/blue]")


# ============================================================================
# BACKTEST SUB-APP COMMANDS
# ============================================================================


@backtest_app.command()
def strategy(
    data_path: Path | None = DEFAULT_DATA_PATH,
    model_path: Path | None = DEFAULT_MODEL_PATH,
    policy: str | None = DEFAULT_POLICY,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    commission: float = DEFAULT_COMMISSION,
    slippage: float = DEFAULT_SLIPPAGE,
    output_dir: Path = DEFAULT_BACKTEST_OUTPUT,
) -> None:
    """
    Run backtesting on historical data.

    Uses the TradingSession class from src/trading_rl_agent/core/live_trading.py
    adapted for backtesting to evaluate trading strategies on historical data.
    """
    console.print(f"[blue]PLACEHOLDER: Would backtest strategy on {data_path} with ${initial_capital} capital[/blue]")
    console.print("[blue]Target module: src/trading_rl_agent/core/live_trading.py - TradingSession (adapted)[/blue]")


@backtest_app.command()
def evaluate(
    model_path: Path | None = DEFAULT_MODEL_PATH,
    data_path: Path | None = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_EVALUATION_OUTPUT,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> None:
    """
    Evaluate trained models on test data.

    Uses the OptimizedTrainingManager.evaluate() function from
    src/trading_rl_agent/training/optimized_trainer.py to evaluate
    model performance on test datasets.
    """
    console.print(f"[blue]PLACEHOLDER: Would evaluate model {model_path} on test data[/blue]")
    console.print(
        "[blue]Target module: src/trading_rl_agent/training/optimized_trainer.py - OptimizedTrainingManager.evaluate()[/blue]"
    )


@backtest_app.command()
def walk_forward(
    data_path: Path | None = DEFAULT_DATA_PATH,
    model_type: str = "cnn_lstm",
    train_window_size: int = 252,
    validation_window_size: int = 63,
    test_window_size: int = 63,
    step_size: int = 21,
    output_dir: Path = DEFAULT_EVALUATION_OUTPUT,
    confidence_level: float = 0.95,
    generate_plots: bool = True,
    save_results: bool = True,
) -> None:
    """
    Perform walk-forward analysis for robust model evaluation.

    Uses the WalkForwardAnalyzer class from src/trading_rl_agent/eval/walk_forward_analyzer.py
    to evaluate model performance across multiple time windows.
    """
    console.print(f"[blue]PLACEHOLDER: Would perform walk-forward analysis on {data_path}[/blue]")
    console.print(
        "[blue]Target module: src/trading_rl_agent/eval/walk_forward_analyzer.py - WalkForwardAnalyzer[/blue]"
    )

    # TODO: Implement actual walk-forward analysis
    # from trading_rl_agent.eval import WalkForwardAnalyzer, WalkForwardConfig
    #
    # config = WalkForwardConfig(
    #     train_window_size=train_window_size,
    #     validation_window_size=validation_window_size,
    #     test_window_size=test_window_size,
    #     step_size=step_size,
    #     model_type=model_type,
    #     confidence_level=confidence_level,
    #     output_dir=str(output_dir),
    #     generate_plots=generate_plots,
    #     save_results=save_results,
    # )
    #
    # analyzer = WalkForwardAnalyzer(config)
    # results = analyzer.analyze(data)
    # analyzer.print_summary()


@backtest_app.command()
def compare(
    models: str | None = DEFAULT_MODELS,
    data_path: Path | None = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_COMPARISON_OUTPUT,
) -> None:
    """
    Compare multiple models on the same dataset.

    Evaluates multiple models using the same evaluation framework
    and generates comparative performance reports.
    """
    console.print(f"[blue]PLACEHOLDER: Would compare models: {models}[/blue]")
    console.print("[blue]Target module: Multiple evaluation functions[/blue]")


@backtest_app.command()
def report(
    results_path: Path | None = DEFAULT_RESULTS_PATH,
    output_format: str = DEFAULT_REPORT_FORMAT,
    output_dir: Path = DEFAULT_REPORTS_OUTPUT,
) -> None:
    """
    Generate performance reports from backtest results.

    Creates comprehensive performance reports including metrics,
    charts, and analysis from backtesting results.
    """
    console.print(f"[blue]PLACEHOLDER: Would generate {output_format} report from {results_path}[/blue]")
    console.print("[blue]Target module: Report generation utilities[/blue]")


# ============================================================================
# TRADE SUB-APP COMMANDS
# ============================================================================


@trade_app.command()
def start(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    symbols: str | None = DEFAULT_SYMBOLS_NONE,
    model_path: Path | None = DEFAULT_MODEL_PATH,
    paper_trading: bool = DEFAULT_PAPER_TRADING,
    initial_capital: float = DEFAULT_TRADING_CAPITAL,
) -> None:
    """
    Start live trading session.

    Uses the LiveTradingEngine.create_session() and TradingSession.start() functions
    from src/trading_rl_agent/core/live_trading.py to initiate live trading.
    """
    console.print(f"[blue]PLACEHOLDER: Would start live trading with ${initial_capital} capital[/blue]")
    console.print(
        "[blue]Target module: src/trading_rl_agent/core/live_trading.py - LiveTradingEngine, TradingSession[/blue]"
    )


@trade_app.command()
def stop(
    session_id: str | None = DEFAULT_SESSION_ID_NONE,
    all_sessions: bool = DEFAULT_ALL_SESSIONS_FALSE,
) -> None:
    """
    Stop live trading session(s).

    Uses the LiveTradingEngine.stop_all_sessions() function from
    src/trading_rl_agent/core/live_trading.py to stop trading sessions.
    """
    console.print("[blue]PLACEHOLDER: Would stop trading session(s)[/blue]")
    console.print(
        "[blue]Target module: src/trading_rl_agent/core/live_trading.py - LiveTradingEngine.stop_all_sessions()[/blue]"
    )


@trade_app.command()
def status(
    session_id: str | None = DEFAULT_SESSION_ID_NONE,
    detailed: bool = DEFAULT_DETAILED_FALSE,
) -> None:
    """
    Show trading session status.

    Displays current status of trading sessions including portfolio value,
    positions, and performance metrics.
    """
    console.print("[blue]PLACEHOLDER: Would show trading session status[/blue]")
    console.print("[blue]Target module: Trading session monitoring functions[/blue]")


@trade_app.command()
def monitor(
    session_id: str | None = DEFAULT_SESSION_ID_NONE,
    metrics: str = DEFAULT_METRICS_ALL,
    interval: int = DEFAULT_INTERVAL_60,
) -> None:
    """
    Monitor live trading session in real-time.

    Provides real-time monitoring of trading sessions with live updates
    on portfolio performance, risk metrics, and trading activity.
    """
    console.print(f"[blue]PLACEHOLDER: Would monitor trading session with {interval}s interval[/blue]")
    console.print("[blue]Target module: Real-time monitoring functions[/blue]")


@monitor_app.command()
def paper(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    symbols: str = DEFAULT_PAPER_SYMBOLS,
    duration: str = DEFAULT_PAPER_DURATION,
) -> None:
    """Start paper trading session for testing strategies."""
    console.print(f"[blue]PLACEHOLDER: Would start paper trading for {symbols} for {duration}[/blue]")


# Feature Engineering Commands
@features_app.command()
def technical_indicators(
    input_path: Path = typer.Argument(..., help="Path to input CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "features" / "technical.csv", "--output", "-o", help="Output path"),
    ma_windows: str = typer.Option("5,10,20,50", "--ma-windows", help="Moving average windows (comma-separated)"),
    rsi_window: int = typer.Option(14, "--rsi-window", help="RSI window"),
    vol_window: int = typer.Option(20, "--vol-window", help="Volume window"),
) -> None:
    """Add technical indicators to market data."""
    try:
        import pandas as pd
        from .data.features import generate_features
        
        # Load data
        df = pd.read_csv(input_path)
        console.print(f"[green]Loaded data: {len(df)} rows[/green]")
        
        # Parse MA windows
        ma_windows_list = [int(x.strip()) for x in ma_windows.split(",")]
        
        # Generate features
        featured_df = generate_features(
            df, 
            ma_windows=ma_windows_list,
            rsi_window=rsi_window,
            vol_window=vol_window,
            advanced_candles=False
        )
        
        # Save output
        featured_df.to_csv(output_path, index=False)
        console.print(f"[green]Technical indicators added and saved to {output_path}[/green]")
        console.print(f"[blue]Added {len(featured_df.columns) - len(df.columns)} new features[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error adding technical indicators: {e}[/red]")
        raise typer.Exit(1) from e


@features_app.command()
def candlestick_patterns(
    input_path: Path = typer.Argument(..., help="Path to input CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "features" / "candlestick.csv", "--output", "-o", help="Output path"),
    advanced: bool = typer.Option(True, "--advanced/--basic", help="Include advanced patterns"),
) -> None:
    """Add candlestick pattern detection to market data."""
    try:
        import pandas as pd
        from .data.features import compute_candle_features, add_missing_candlestick_patterns
        
        # Load data
        df = pd.read_csv(input_path)
        console.print(f"[green]Loaded data: {len(df)} rows[/green]")
        
        # Add basic candlestick features
        df = compute_candle_features(df, advanced=advanced)
        
        # Add missing patterns if advanced
        if advanced:
            df = add_missing_candlestick_patterns(df)
        
        # Save output
        df.to_csv(output_path, index=False)
        console.print(f"[green]Candlestick patterns added and saved to {output_path}[/green]")
        
        # Count pattern columns
        pattern_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['doji', 'hammer', 'engulfing', 'star', 'harami', 'tweezer'])]
        console.print(f"[blue]Added {len(pattern_cols)} candlestick pattern features[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error adding candlestick patterns: {e}[/red]")
        raise typer.Exit(1) from e


@features_app.command()
def market_regime(
    input_path: Path = typer.Argument(..., help="Path to input CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "features" / "market_regime.csv", "--output", "-o", help="Output path"),
    volatility_window: int = typer.Option(20, "--volatility-window", help="Volatility calculation window"),
    trend_window: int = typer.Option(50, "--trend-window", help="Trend calculation window"),
) -> None:
    """Add market regime features to market data."""
    try:
        import pandas as pd
        from .data.features import add_market_regime_features
        
        # Load data
        df = pd.read_csv(input_path)
        console.print(f"[green]Loaded data: {len(df)} rows[/green]")
        
        # Add market regime features
        df = add_market_regime_features(df)
        
        # Save output
        df.to_csv(output_path, index=False)
        console.print(f"[green]Market regime features added and saved to {output_path}[/green]")
        
        # Count regime columns
        regime_cols = [col for col in df.columns if 'regime' in col.lower()]
        console.print(f"[blue]Added {len(regime_cols)} market regime features[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error adding market regime features: {e}[/red]")
        raise typer.Exit(1) from e


@features_app.command()
def all_features(
    input_path: Path = typer.Argument(..., help="Path to input CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "features" / "all_features.csv", "--output", "-o", help="Output path"),
    ma_windows: str = typer.Option("5,10,20,50", "--ma-windows", help="Moving average windows (comma-separated)"),
    rsi_window: int = typer.Option(14, "--rsi-window", help="RSI window"),
    vol_window: int = typer.Option(20, "--vol-window", help="Volume window"),
    advanced_candles: bool = typer.Option(True, "--advanced-candles/--basic-candles", help="Include advanced candlestick patterns"),
) -> None:
    """Add all feature types to market data (technical indicators, candlestick patterns, market regimes)."""
    try:
        import pandas as pd
        from .data.features import generate_features, add_market_regime_features
        
        # Load data
        df = pd.read_csv(input_path)
        console.print(f"[green]Loaded data: {len(df)} rows[/green]")
        
        # Parse MA windows
        ma_windows_list = [int(x.strip()) for x in ma_windows.split(",")]
        
        # Generate all features
        featured_df = generate_features(
            df, 
            ma_windows=ma_windows_list,
            rsi_window=rsi_window,
            vol_window=vol_window,
            advanced_candles=advanced_candles
        )
        
        # Add market regime features
        featured_df = add_market_regime_features(featured_df)
        
        # Save output
        featured_df.to_csv(output_path, index=False)
        console.print(f"[green]All features added and saved to {output_path}[/green]")
        console.print(f"[blue]Total features: {len(featured_df.columns)} (original: {len(df.columns)})[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error adding all features: {e}[/red]")
        raise typer.Exit(1) from e


# Advanced Data Commands
@advanced_data_app.command()
def build_optimized_dataset(
    symbols: str = typer.Argument(..., help="Comma-separated list of symbols"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR / "optimized", "--output", "-o", help="Output directory"),
    sequence_length: int = typer.Option(60, "--sequence-length", help="Sequence length for CNN+LSTM"),
    prediction_horizon: int = typer.Option(1, "--prediction-horizon", help="Prediction horizon"),
    real_data_ratio: float = typer.Option(0.95, "--real-data-ratio", help="Ratio of real data (0.0-1.0)"),
    technical_indicators: bool = typer.Option(True, "--technical/--no-technical", help="Include technical indicators"),
    sentiment_features: bool = typer.Option(True, "--sentiment/--no-sentiment", help="Include sentiment features"),
    market_regime_features: bool = typer.Option(True, "--market-regime/--no-market-regime", help="Include market regime features"),
    max_workers: int = typer.Option(None, "--max-workers", help="Maximum parallel workers"),
) -> None:
    """Build optimized dataset with parallel processing and advanced features."""
    try:
        from .data.optimized_dataset_builder import OptimizedDatasetBuilder, OptimizedDatasetConfig
        
        # Parse symbols
        symbols_list = [s.strip() for s in symbols.split(",")]
        
        # Create config
        config = OptimizedDatasetConfig(
            symbols=symbols_list,
            start_date=start_date,
            end_date=end_date,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            real_data_ratio=real_data_ratio,
            technical_indicators=technical_indicators,
            sentiment_features=sentiment_features,
            market_regime_features=market_regime_features,
            max_workers=max_workers,
            output_dir=str(output_dir)
        )
        
        # Build dataset
        console.print(f"[green]Building optimized dataset for {len(symbols_list)} symbols...[/green]")
        builder = OptimizedDatasetBuilder(config)
        sequences, targets, dataset_info = builder.build_dataset()
        
        console.print(f"[green]✅ Optimized dataset built successfully![/green]")
        console.print(f"[blue]Sequences shape: {sequences.shape}[/blue]")
        console.print(f"[blue]Targets shape: {targets.shape}[/blue]")
        console.print(f"[blue]Dataset info: {dataset_info}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error building optimized dataset: {e}[/red]")
        raise typer.Exit(1) from e


@advanced_data_app.command()
def standardize_data(
    input_path: Path = typer.Argument(..., help="Path to input CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "standardized" / "standardized.csv", "--output", "-o", help="Output path"),
    method: str = typer.Option("robust", "--method", help="Standardization method (robust, standard, minmax)"),
    outlier_threshold: float = typer.Option(5.0, "--outlier-threshold", help="Outlier detection threshold"),
    missing_value_threshold: float = typer.Option(0.25, "--missing-threshold", help="Missing value threshold"),
) -> None:
    """Standardize and clean market data."""
    try:
        import pandas as pd
        from .data.data_standardizer import DataStandardizer
        
        # Load data
        df = pd.read_csv(input_path)
        console.print(f"[green]Loaded data: {len(df)} rows, {len(df.columns)} columns[/green]")
        
        # Create standardizer
        standardizer = DataStandardizer(
            outlier_threshold=outlier_threshold,
            missing_value_threshold=missing_value_threshold
        )
        
        # Standardize data
        standardized_df = standardizer.standardize(df, method=method)
        
        # Save output
        standardized_df.to_csv(output_path, index=False)
        console.print(f"[green]Data standardized and saved to {output_path}[/green]")
        console.print(f"[blue]Final shape: {standardized_df.shape}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error standardizing data: {e}[/red]")
        raise typer.Exit(1) from e


@advanced_data_app.command()
def generate_synthetic_data(
    symbols: str = typer.Argument(..., help="Comma-separated list of symbols"),
    n_samples: int = typer.Option(1000, "--n-samples", help="Number of samples per symbol"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "synthetic" / "synthetic_data.csv", "--output", "-o", help="Output path"),
    volatility: float = typer.Option(0.02, "--volatility", help="Volatility parameter"),
    trend: float = typer.Option(0.001, "--trend", help="Trend parameter"),
    noise_level: float = typer.Option(0.01, "--noise", help="Noise level"),
) -> None:
    """Generate synthetic market data for testing and augmentation."""
    try:
        from .data.synthetic import fetch_synthetic_data
        import pandas as pd
        
        # Parse symbols
        symbols_list = [s.strip() for s in symbols.split(",")]
        
        # Generate synthetic data
        console.print(f"[green]Generating synthetic data for {len(symbols_list)} symbols...[/green]")
        
        all_data = []
        for symbol in symbols_list:
            synthetic_df = fetch_synthetic_data(
                symbol=symbol,
                n_samples=n_samples,
                volatility=volatility,
                trend=trend,
                noise_level=noise_level
            )
            synthetic_df['symbol'] = symbol
            synthetic_df['data_source'] = 'synthetic'
            all_data.append(synthetic_df)
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save output
        combined_df.to_csv(output_path, index=False)
        console.print(f"[green]Synthetic data generated and saved to {output_path}[/green]")
        console.print(f"[blue]Total samples: {len(combined_df)}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error generating synthetic data: {e}[/red]")
        raise typer.Exit(1) from e


@advanced_data_app.command()
def alternative_data(
    symbols: str = typer.Argument(..., help="Comma-separated list of symbols"),
    start_date: str = typer.Argument(..., help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Argument(..., help="End date (YYYY-MM-DD)"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "alternative" / "alternative_data.csv", "--output", "-o", help="Output path"),
    include_sentiment: bool = typer.Option(True, "--sentiment/--no-sentiment", help="Include sentiment data"),
    include_news: bool = typer.Option(True, "--news/--no-news", help="Include news data"),
    include_social: bool = typer.Option(True, "--social/--no-social", help="Include social media data"),
) -> None:
    """Fetch and process alternative data sources."""
    try:
        from .features.alternative_data import AlternativeDataFeatures, AlternativeDataConfig
        import pandas as pd
        
        # Parse symbols
        symbols_list = [s.strip() for s in symbols.split(",")]
        
        # Create config
        config = AlternativeDataConfig()
        alt_data = AlternativeDataFeatures(config)
        
        # Fetch alternative data
        console.print(f"[green]Fetching alternative data for {len(symbols_list)} symbols...[/green]")
        
        all_data = []
        for symbol in symbols_list:
            symbol_data = alt_data.get_alternative_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                include_sentiment=include_sentiment,
                include_news=include_news,
                include_social=include_social
            )
            if symbol_data is not None and not symbol_data.empty:
                symbol_data['symbol'] = symbol
                all_data.append(symbol_data)
        
        if all_data:
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save output
            combined_df.to_csv(output_path, index=False)
            console.print(f"[green]Alternative data saved to {output_path}[/green]")
            console.print(f"[blue]Total samples: {len(combined_df)}[/blue]")
        else:
            console.print("[yellow]No alternative data found for the specified symbols and date range[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error fetching alternative data: {e}[/red]")
        raise typer.Exit(1) from e


# Advanced Training Commands
@advanced_train_app.command()
def optimized_cnn_lstm(
    data_path: Path = typer.Argument(..., help="Path to training data"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR / "advanced_models" / "cnn_lstm", "--output", "-o", help="Output directory"),
    epochs: int = typer.Option(100, "--epochs", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size"),
    learning_rate: float = typer.Option(0.001, "--lr", help="Learning rate"),
    enable_amp: bool = typer.Option(True, "--amp/--no-amp", help="Enable automatic mixed precision"),
    enable_checkpointing: bool = typer.Option(True, "--checkpointing/--no-checkpointing", help="Enable gradient checkpointing"),
    memory_efficient: bool = typer.Option(True, "--memory-efficient/--no-memory-efficient", help="Enable memory optimizations"),
    scheduler_type: str = typer.Option("cosine", "--scheduler", help="Learning rate scheduler (cosine, onecycle, plateau)"),
    early_stopping_patience: int = typer.Option(15, "--patience", help="Early stopping patience"),
    mixup_alpha: float = typer.Option(0.2, "--mixup", help="MixUp augmentation alpha"),
    cutmix_prob: float = typer.Option(0.3, "--cutmix", help="CutMix augmentation probability"),
) -> None:
    """Train CNN+LSTM model with advanced optimizations."""
    try:
        from .training.optimized_trainer import create_optimized_trainer, create_advanced_scheduler
        from .training.train_cnn_lstm_enhanced import create_cnn_lstm_model
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        
        # Load data
        console.print(f"[green]Loading training data from {data_path}...[/green]")
        data = np.load(data_path)
        sequences = data['sequences']
        targets = data['targets']
        
        # Create model
        model = create_cnn_lstm_model(
            input_shape=sequences.shape[1:],
            num_classes=targets.shape[-1] if len(targets.shape) > 1 else 1
        )
        
        # Create optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Create scheduler
        scheduler = create_advanced_scheduler(
            scheduler_type=scheduler_type,
            optimizer=optimizer,
            epochs=epochs,
            steps_per_epoch=len(sequences) // batch_size
        )
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Create trainer
        trainer = create_optimized_trainer(
            model=model,
            enable_amp=enable_amp,
            enable_checkpointing=enable_checkpointing,
            memory_efficient=memory_efficient
        )
        
        # Create data loaders
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(
            torch.FloatTensor(sequences),
            torch.FloatTensor(targets)
        )
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Train model
        console.print(f"[green]Starting optimized training for {epochs} epochs...[/green]")
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            save_path=str(output_dir / "best_model.pth")
        )
        
        console.print(f"[green]✅ Optimized training completed![/green]")
        console.print(f"[blue]Best validation loss: {results['best_val_loss']:.6f}[/blue]")
        console.print(f"[blue]Training history: {results['training_history']}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error in optimized training: {e}[/red]")
        raise typer.Exit(1) from e


@advanced_train_app.command()
def hyperparameter_optimization(
    data_path: Path = typer.Argument(..., help="Path to training data"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR / "hyperopt", "--output", "-o", help="Output directory"),
    n_trials: int = typer.Option(100, "--n-trials", help="Number of optimization trials"),
    study_name: str = typer.Option("cnn_lstm_optimization", "--study-name", help="Optuna study name"),
    storage: str = typer.Option(None, "--storage", help="Optuna storage URL"),
) -> None:
    """Run hyperparameter optimization for CNN+LSTM model."""
    try:
        import optuna
        from .training.optimized_trainer import create_optimized_trainer
        from .training.train_cnn_lstm_enhanced import create_cnn_lstm_model
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import numpy as np
        
        # Load data
        console.print(f"[green]Loading training data from {data_path}...[/green]")
        data = np.load(data_path)
        sequences = data['sequences']
        targets = data['targets']
        
        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
            num_layers = trial.suggest_int("num_layers", 1, 4)
            dropout = trial.suggest_float("dropout", 0.1, 0.5)
            
            # Create model
            model = create_cnn_lstm_model(
                input_shape=sequences.shape[1:],
                num_classes=targets.shape[-1] if len(targets.shape) > 1 else 1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Create optimizer and criterion
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            criterion = nn.MSELoss()
            
            # Create trainer
            trainer = create_optimized_trainer(model=model, enable_amp=True)
            
            # Create data loaders
            from torch.utils.data import DataLoader, TensorDataset
            dataset = TensorDataset(
                torch.FloatTensor(sequences),
                torch.FloatTensor(targets)
            )
            
            # Split data
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Train for a few epochs
            results = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                epochs=10,  # Short training for optimization
                early_stopping_patience=5
            )
            
            return results['best_val_loss']
        
        # Create study
        study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            storage=storage
        )
        
        # Run optimization
        console.print(f"[green]Starting hyperparameter optimization with {n_trials} trials...[/green]")
        study.optimize(objective, n_trials=n_trials)
        
        # Save results
        import json
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            "best_params": best_params,
            "best_value": best_value,
            "n_trials": n_trials
        }
        
        with open(output_dir / "optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        console.print(f"[green]✅ Hyperparameter optimization completed![/green]")
        console.print(f"[blue]Best validation loss: {best_value:.6f}[/blue]")
        console.print(f"[blue]Best parameters: {best_params}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error in hyperparameter optimization: {e}[/red]")
        raise typer.Exit(1) from e


@advanced_train_app.command()
def data_augmentation_test(
    input_path: Path = typer.Argument(..., help="Path to input data"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "augmented" / "augmented_data.npz", "--output", "-o", help="Output path"),
    mixup_alpha: float = typer.Option(0.2, "--mixup", help="MixUp augmentation alpha"),
    cutmix_prob: float = typer.Option(0.3, "--cutmix", help="CutMix augmentation probability"),
    noise_factor: float = typer.Option(0.01, "--noise", help="Noise factor"),
    sequence_shift: bool = typer.Option(True, "--sequence-shift/--no-sequence-shift", help="Enable sequence shifting"),
    n_augmented_samples: int = typer.Option(1000, "--n-samples", help="Number of augmented samples to generate"),
) -> None:
    """Test and apply data augmentation techniques."""
    try:
        from .training.optimized_trainer import AdvancedDataAugmentation
        import torch
        import numpy as np
        
        # Load data
        console.print(f"[green]Loading data from {input_path}...[/green]")
        data = np.load(input_path)
        sequences = data['sequences']
        targets = data['targets']
        
        # Create augmentation
        augmentation = AdvancedDataAugmentation(
            mixup_alpha=mixup_alpha,
            cutmix_prob=cutmix_prob,
            noise_factor=noise_factor,
            sequence_shift=sequence_shift
        )
        
        # Convert to tensors
        data_tensor = torch.FloatTensor(sequences)
        target_tensor = torch.FloatTensor(targets)
        
        # Generate augmented samples
        console.print(f"[green]Generating {n_augmented_samples} augmented samples...[/green]")
        
        augmented_sequences = []
        augmented_targets = []
        
        for i in range(n_augmented_samples):
            # Randomly select a sample
            idx = np.random.randint(0, len(data_tensor))
            sample_data = data_tensor[idx:idx+1]
            sample_target = target_tensor[idx:idx+1]
            
            # Apply augmentation
            aug_data, aug_target = augmentation.augment_batch(sample_data, sample_target)
            
            augmented_sequences.append(aug_data.numpy())
            augmented_targets.append(aug_target.numpy())
        
        # Combine augmented data
        aug_sequences = np.concatenate(augmented_sequences, axis=0)
        aug_targets = np.concatenate(augmented_targets, axis=0)
        
        # Save augmented data
        np.savez(output_path, sequences=aug_sequences, targets=aug_targets)
        
        console.print(f"[green]✅ Data augmentation completed![/green]")
        console.print(f"[blue]Augmented sequences shape: {aug_sequences.shape}[/blue]")
        console.print(f"[blue]Augmented targets shape: {aug_targets.shape}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error in data augmentation: {e}[/red]")
        raise typer.Exit(1) from e


# NLP Commands
@nlp_app.command()
def analyze_news(
    input_path: Path = typer.Argument(..., help="Path to news articles CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "nlp" / "news_analysis.csv", "--output", "-o", help="Output path"),
    min_relevance: float = typer.Option(0.5, "--min-relevance", help="Minimum relevance score"),
    min_impact: float = typer.Option(0.7, "--min-impact", help="Minimum market impact score"),
) -> None:
    """Analyze news articles for sentiment and market impact."""
    try:
        from .nlp.news_analyzer import NewsAnalyzer, NewsArticle
        from datetime import datetime
        import pandas as pd
        
        # Load news data
        console.print(f"[green]Loading news data from {input_path}...[/green]")
        df = pd.read_csv(input_path)
        
        # Create news analyzer
        analyzer = NewsAnalyzer()
        
        # Convert to NewsArticle objects
        articles = []
        for _, row in df.iterrows():
            article = NewsArticle(
                title=row.get('title', ''),
                content=row.get('content', ''),
                source=row.get('source', 'unknown'),
                published_at=datetime.fromisoformat(row.get('published_at', datetime.now().isoformat())),
                url=row.get('url'),
                author=row.get('author'),
                category=row.get('category')
            )
            articles.append(article)
        
        # Analyze articles
        console.print(f"[green]Analyzing {len(articles)} news articles...[/green]")
        analyses = analyzer.analyze_batch(articles)
        
        # Filter by relevance and impact
        relevant_analyses = analyzer.filter_relevant_articles(analyses, min_relevance=min_relevance)
        high_impact_analyses = analyzer.get_high_impact_articles(analyses, min_impact=min_impact)
        
        # Get market sentiment
        market_sentiment = analyzer.get_market_sentiment(analyses)
        
        # Convert to DataFrame
        results = []
        for analysis in analyses:
            results.append({
                'title': analysis.article.title,
                'source': analysis.article.source,
                'published_at': analysis.article.published_at.isoformat(),
                'sentiment_score': analysis.sentiment.sentiment_score,
                'sentiment_label': analysis.sentiment.sentiment_label,
                'market_impact': analysis.market_impact,
                'relevance_score': analysis.relevance_score,
                'entities': ', '.join(analysis.entities),
                'is_relevant': analysis.relevance_score >= min_relevance,
                'is_high_impact': analysis.market_impact >= min_impact
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        
        console.print(f"[green]✅ News analysis completed![/green]")
        console.print(f"[blue]Total articles analyzed: {len(analyses)}[/blue]")
        console.print(f"[blue]Relevant articles: {len(relevant_analyses)}[/blue]")
        console.print(f"[blue]High impact articles: {len(high_impact_analyses)}[/blue]")
        console.print(f"[blue]Market sentiment: {market_sentiment}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error analyzing news: {e}[/red]")
        raise typer.Exit(1) from e


@nlp_app.command()
def sentiment_analysis(
    input_path: Path = typer.Argument(..., help="Path to text data CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "nlp" / "sentiment_analysis.csv", "--output", "-o", help="Output path"),
    text_column: str = typer.Option("text", "--text-column", help="Column name containing text to analyze"),
    batch_size: int = typer.Option(100, "--batch-size", help="Batch size for processing"),
) -> None:
    """Perform sentiment analysis on text data."""
    try:
        from .nlp.sentiment_analyzer import SentimentAnalyzer
        import pandas as pd
        
        # Load text data
        console.print(f"[green]Loading text data from {input_path}...[/green]")
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        
        # Create sentiment analyzer
        analyzer = SentimentAnalyzer()
        
        # Analyze sentiment
        console.print(f"[green]Analyzing sentiment for {len(df)} texts...[/green]")
        
        results = []
        for i in range(0, len(df), batch_size):
            batch = df[text_column].iloc[i:i+batch_size]
            batch_results = [analyzer.analyze_sentiment(text) for text in batch]
            results.extend(batch_results)
            
            if (i + batch_size) % 1000 == 0:
                console.print(f"[blue]Processed {i + batch_size} texts...[/blue]")
        
        # Add sentiment results to DataFrame
        df['sentiment_score'] = [r.sentiment_score for r in results]
        df['sentiment_label'] = [r.sentiment_label for r in results]
        df['sentiment_confidence'] = [r.confidence for r in results]
        
        # Save results
        df.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        sentiment_counts = df['sentiment_label'].value_counts()
        avg_confidence = df['sentiment_confidence'].mean()
        
        console.print(f"[green]✅ Sentiment analysis completed![/green]")
        console.print(f"[blue]Total texts analyzed: {len(df)}[/blue]")
        console.print(f"[blue]Sentiment distribution: {sentiment_counts.to_dict()}[/blue]")
        console.print(f"[blue]Average confidence: {avg_confidence:.3f}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error in sentiment analysis: {e}[/red]")
        raise typer.Exit(1) from e


@nlp_app.command()
def process_text(
    input_path: Path = typer.Argument(..., help="Path to text data CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "nlp" / "processed_text.csv", "--output", "-o", help="Output path"),
    text_column: str = typer.Option("text", "--text-column", help="Column name containing text to process"),
    remove_stopwords: bool = typer.Option(True, "--remove-stopwords/--keep-stopwords", help="Remove stopwords"),
    lemmatize: bool = typer.Option(True, "--lemmatize/--no-lemmatize", help="Apply lemmatization"),
    remove_punctuation: bool = typer.Option(True, "--remove-punctuation/--keep-punctuation", help="Remove punctuation"),
    lowercase: bool = typer.Option(True, "--lowercase/--no-lowercase", help="Convert to lowercase"),
) -> None:
    """Process and clean text data."""
    try:
        from .nlp.text_processor import TextProcessor
        import pandas as pd
        
        # Load text data
        console.print(f"[green]Loading text data from {input_path}...[/green]")
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        
        # Create text processor
        processor = TextProcessor()
        
        # Process text
        console.print(f"[green]Processing {len(df)} texts...[/green]")
        
        processed_texts = []
        for text in df[text_column]:
            processed = processor.process_text(
                text,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
                remove_punctuation=remove_punctuation,
                lowercase=lowercase
            )
            processed_texts.append(processed)
        
        # Add processed text to DataFrame
        df['processed_text'] = [pt.processed_text for pt in processed_texts]
        df['tokens'] = [pt.tokens for pt in processed_texts]
        df['token_count'] = [len(pt.tokens) for pt in processed_texts]
        df['entities'] = [pt.entities for pt in processed_texts]
        
        # Save results
        df.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        avg_tokens = df['token_count'].mean()
        total_entities = sum(len(entities) for entities in df['entities'])
        
        console.print(f"[green]✅ Text processing completed![/green]")
        console.print(f"[blue]Total texts processed: {len(df)}[/blue]")
        console.print(f"[blue]Average tokens per text: {avg_tokens:.1f}[/blue]")
        console.print(f"[blue]Total entities found: {total_entities}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error in text processing: {e}[/red]")
        raise typer.Exit(1) from e


@nlp_app.command()
def extract_entities(
    input_path: Path = typer.Argument(..., help="Path to text data CSV file"),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "nlp" / "entities.csv", "--output", "-o", help="Output path"),
    text_column: str = typer.Option("text", "--text-column", help="Column name containing text to analyze"),
    entity_types: str = typer.Option("PERSON,ORG,GPE", "--entity-types", help="Comma-separated list of entity types to extract"),
) -> None:
    """Extract named entities from text data."""
    try:
        from .nlp.text_processor import TextProcessor
        import pandas as pd
        
        # Load text data
        console.print(f"[green]Loading text data from {input_path}...[/green]")
        df = pd.read_csv(input_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in data")
        
        # Parse entity types
        entity_types_list = [et.strip() for et in entity_types.split(",")]
        
        # Create text processor
        processor = TextProcessor()
        
        # Extract entities
        console.print(f"[green]Extracting entities from {len(df)} texts...[/green]")
        
        all_entities = []
        for text in df[text_column]:
            entities = processor.extract_entities(text, entity_types=entity_types_list)
            all_entities.append(entities)
        
        # Add entities to DataFrame
        df['entities'] = all_entities
        df['entity_count'] = [len(entities) for entities in all_entities]
        
        # Create entity summary
        entity_summary = {}
        for entities in all_entities:
            for entity in entities:
                entity_type = entity.get('type', 'UNKNOWN')
                entity_text = entity.get('text', '')
                if entity_type not in entity_summary:
                    entity_summary[entity_type] = {}
                if entity_text not in entity_summary[entity_type]:
                    entity_summary[entity_type][entity_text] = 0
                entity_summary[entity_type][entity_text] += 1
        
        # Save results
        df.to_csv(output_path, index=False)
        
        # Save entity summary
        summary_path = output_path.parent / "entity_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(entity_summary, f, indent=2)
        
        console.print(f"[green]✅ Entity extraction completed![/green]")
        console.print(f"[blue]Total texts processed: {len(df)}[/blue]")
        console.print(f"[blue]Total entities found: {sum(df['entity_count'])}[/blue]")
        console.print(f"[blue]Entity types found: {list(entity_summary.keys())}[/blue]")
        console.print(f"[blue]Entity summary saved to: {summary_path}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error in entity extraction: {e}[/red]")
        raise typer.Exit(1) from e


# Monitoring Commands
@monitor_app.command()
def dashboard(
    port: int = typer.Option(8080, "--port", help="Dashboard port"),
    host: str = typer.Option("localhost", "--host", help="Dashboard host"),
    refresh_interval: int = typer.Option(5, "--refresh", help="Refresh interval in seconds"),
    metrics_collector: str = typer.Option(None, "--metrics", help="Path to metrics collector config"),
) -> None:
    """Start the monitoring dashboard."""
    try:
        from .monitoring.dashboard import Dashboard
        from .monitoring.alert_manager import AlertManager
        import time
        import threading
        
        # Create dashboard components
        alert_manager = AlertManager()
        dashboard = Dashboard(alert_manager=alert_manager)
        
        console.print(f"[green]Starting monitoring dashboard on {host}:{port}...[/green]")
        console.print(f"[blue]Refresh interval: {refresh_interval} seconds[/blue]")
        
        # Start dashboard in a separate thread
        def run_dashboard():
            try:
                # This would typically start a web server
                # For now, we'll simulate dashboard updates
                while True:
                    overview = dashboard.get_system_overview()
                    trading_metrics = dashboard.get_trading_metrics()
                    risk_metrics = dashboard.get_risk_metrics()
                    
                    console.print(f"[blue]System Status: {overview['system_status']}[/blue]")
                    console.print(f"[blue]P&L: ${trading_metrics['pnl']:.2f}[/blue]")
                    console.print(f"[blue]Risk (VaR): ${risk_metrics['var_95']:.2f}[/blue]")
                    
                    time.sleep(refresh_interval)
                    
            except KeyboardInterrupt:
                console.print("[yellow]Dashboard stopped by user[/yellow]")
        
        # Start dashboard thread
        dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
        dashboard_thread.start()
        
        console.print("[green]✅ Dashboard started! Press Ctrl+C to stop.[/green]")
        
        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("[yellow]Stopping dashboard...[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error starting dashboard: {e}[/red]")
        raise typer.Exit(1) from e


@monitor_app.command()
def system_health(
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "monitoring" / "system_health.json", "--output", "-o", help="Output path"),
    detailed: bool = typer.Option(False, "--detailed/--summary", help="Include detailed metrics"),
) -> None:
    """Get system health metrics."""
    try:
        from .monitoring.dashboard import Dashboard
        import json
        import psutil
        
        # Create dashboard
        dashboard = Dashboard()
        
        # Get system health
        health_metrics = dashboard.get_system_health()
        
        # Add additional system metrics if detailed
        if detailed:
            # CPU info
            health_metrics['cpu_count'] = psutil.cpu_count()
            health_metrics['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            
            # Memory info
            memory = psutil.virtual_memory()
            health_metrics['memory_total'] = memory.total
            health_metrics['memory_available'] = memory.available
            health_metrics['memory_percent'] = memory.percent
            
            # Disk info
            disk = psutil.disk_usage('/')
            health_metrics['disk_total'] = disk.total
            health_metrics['disk_free'] = disk.free
            health_metrics['disk_percent'] = disk.percent
            
            # Network info
            network = psutil.net_io_counters()
            health_metrics['network_bytes_sent'] = network.bytes_sent
            health_metrics['network_bytes_recv'] = network.bytes_recv
        
        # Save metrics
        with open(output_path, 'w') as f:
            json.dump(health_metrics, f, indent=2)
        
        console.print(f"[green]✅ System health metrics saved to {output_path}[/green]")
        console.print(f"[blue]CPU Usage: {health_metrics.get('cpu_usage', 0):.1f}%[/blue]")
        console.print(f"[blue]Memory Usage: {health_metrics.get('memory_usage', 0):.1f}%[/blue]")
        console.print(f"[blue]Disk Usage: {health_metrics.get('disk_usage', 0):.1f}%[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error getting system health: {e}[/red]")
        raise typer.Exit(1) from e


@monitor_app.command()
def trading_metrics(
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "monitoring" / "trading_metrics.json", "--output", "-o", help="Output path"),
    session_id: str = typer.Option(None, "--session-id", help="Trading session ID"),
) -> None:
    """Get trading performance metrics."""
    try:
        from .monitoring.dashboard import Dashboard
        import json
        
        # Create dashboard
        dashboard = Dashboard()
        
        # Get trading metrics
        trading_metrics = dashboard.get_trading_metrics()
        
        # Save metrics
        with open(output_path, 'w') as f:
            json.dump(trading_metrics, f, indent=2)
        
        console.print(f"[green]✅ Trading metrics saved to {output_path}[/green]")
        console.print(f"[blue]P&L: ${trading_metrics.get('pnl', 0):.2f}[/blue]")
        console.print(f"[blue]Total Return: {trading_metrics.get('total_return', 0):.2%}[/blue]")
        console.print(f"[blue]Sharpe Ratio: {trading_metrics.get('sharpe_ratio', 0):.3f}[/blue]")
        console.print(f"[blue]Win Rate: {trading_metrics.get('win_rate', 0):.1%}[/blue]")
        console.print(f"[blue]Total Trades: {trading_metrics.get('total_trades', 0)}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error getting trading metrics: {e}[/red]")
        raise typer.Exit(1) from e


@monitor_app.command()
def risk_metrics(
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "monitoring" / "risk_metrics.json", "--output", "-o", help="Output path"),
    confidence_level: float = typer.Option(0.95, "--confidence", help="Confidence level for VaR/CVaR"),
) -> None:
    """Get risk management metrics."""
    try:
        from .monitoring.dashboard import Dashboard
        import json
        
        # Create dashboard
        dashboard = Dashboard()
        
        # Get risk metrics
        risk_metrics = dashboard.get_risk_metrics()
        
        # Add confidence level
        risk_metrics['confidence_level'] = confidence_level
        
        # Save metrics
        with open(output_path, 'w') as f:
            json.dump(risk_metrics, f, indent=2)
        
        console.print(f"[green]✅ Risk metrics saved to {output_path}[/green]")
        console.print(f"[blue]VaR ({confidence_level:.0%}): ${risk_metrics.get('var_95', 0):.2f}[/blue]")
        console.print(f"[blue]CVaR ({confidence_level:.0%}): ${risk_metrics.get('cvar_95', 0):.2f}[/blue]")
        console.print(f"[blue]Current Exposure: ${risk_metrics.get('current_exposure', 0):.2f}[/blue]")
        console.print(f"[blue]Volatility: {risk_metrics.get('volatility', 0):.2%}[/blue]")
        console.print(f"[blue]Beta: {risk_metrics.get('beta', 0):.3f}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error getting risk metrics: {e}[/red]")
        raise typer.Exit(1) from e


@monitor_app.command()
def alerts(
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "monitoring" / "alerts.json", "--output", "-o", help="Output path"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of alerts to return"),
    active_only: bool = typer.Option(True, "--active-only/--all", help="Show only active alerts"),
) -> None:
    """Get system alerts."""
    try:
        from .monitoring.dashboard import Dashboard
        from .monitoring.alert_manager import AlertManager
        import json
        
        # Create components
        alert_manager = AlertManager()
        dashboard = Dashboard(alert_manager=alert_manager)
        
        # Get alerts
        alerts = dashboard.get_recent_alerts(limit=limit)
        
        # Filter active alerts if requested
        if active_only:
            alerts = [alert for alert in alerts if alert.get('status') == 'active']
        
        # Save alerts
        with open(output_path, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        console.print(f"[green]✅ Alerts saved to {output_path}[/green]")
        console.print(f"[blue]Total alerts: {len(alerts)}[/blue]")
        
        # Show alert summary
        if alerts:
            for alert in alerts[:5]:  # Show first 5 alerts
                status_color = "red" if alert.get('severity') == 'high' else "yellow" if alert.get('severity') == 'medium' else "blue"
                console.print(f"[{status_color}]{alert.get('title', 'Unknown')}: {alert.get('message', 'No message')}[/{status_color}]")
        
    except Exception as e:
        console.print(f"[red]Error getting alerts: {e}[/red]")
        raise typer.Exit(1) from e


@monitor_app.command()
def create_alert(
    title: str = typer.Argument(..., help="Alert title"),
    message: str = typer.Argument(..., help="Alert message"),
    severity: str = typer.Option("medium", "--severity", help="Alert severity (low, medium, high)"),
    category: str = typer.Option("general", "--category", help="Alert category"),
    threshold: float = typer.Option(None, "--threshold", help="Threshold value for the alert"),
) -> None:
    """Create a new alert."""
    try:
        from .monitoring.alert_manager import AlertManager
        
        # Create alert manager
        alert_manager = AlertManager()
        
        # Create alert
        alert = alert_manager.create_alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            threshold=threshold
        )
        
        console.print(f"[green]✅ Alert created successfully![/green]")
        console.print(f"[blue]Alert ID: {alert.id}[/blue]")
        console.print(f"[blue]Title: {alert.title}[/blue]")
        console.print(f"[blue]Severity: {alert.severity}[/blue]")
        console.print(f"[blue]Category: {alert.category}[/blue]")
        
    except Exception as e:
        console.print(f"[red]Error creating alert: {e}[/red]")
        raise typer.Exit(1) from e


@monitor_app.command()
def clear_alerts(
    category: str = typer.Option(None, "--category", help="Clear alerts by category"),
    severity: str = typer.Option(None, "--severity", help="Clear alerts by severity"),
    all_alerts: bool = typer.Option(False, "--all", help="Clear all alerts"),
) -> None:
    """Clear system alerts."""
    try:
        from .monitoring.alert_manager import AlertManager
        
        # Create alert manager
        alert_manager = AlertManager()
        
        # Clear alerts based on criteria
        if all_alerts:
            alert_manager.clear_all_alerts()
            console.print("[green]✅ All alerts cleared![/green]")
        elif category:
            alert_manager.clear_alerts_by_category(category)
            console.print(f"[green]✅ Alerts in category '{category}' cleared![/green]")
        elif severity:
            alert_manager.clear_alerts_by_severity(severity)
            console.print(f"[green]✅ Alerts with severity '{severity}' cleared![/green]")
        else:
            console.print("[yellow]Please specify --category, --severity, or --all to clear alerts[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error clearing alerts: {e}[/red]")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
