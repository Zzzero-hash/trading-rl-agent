"""
Unified Typer CLI for Trading RL Agent.

This module provides a comprehensive command-line interface for all trading RL agent operations:
- Dataset management and download
- Model training (CNN+LSTM and RL agents)
- Backtesting and evaluation
- Live trading execution
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from config import get_settings, load_settings
    from trading_rl_agent.logging_conf import get_logger, setup_logging_for_typer
except ImportError:
    # Fallback for when running as module
    # Import config functions differently to avoid redefinition
    import importlib.util

    from .logging_conf import get_logger, setup_logging_for_typer

    spec = importlib.util.spec_from_file_location("config", Path(__file__).parent.parent / "config.py")
    if spec and spec.loader:
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        get_settings = config_module.get_settings
        load_settings = config_module.load_settings
        Settings = config_module.Settings
    else:
        # Final fallback
        def get_settings() -> Any:  # type: ignore[misc]
            return None

        def load_settings(config_path: "Path | None" = None, env_file: "Path | None" = None) -> Any:  # type: ignore[misc]
            return None


# Type variable for decorator functions
F = TypeVar("F", bound=Callable[..., Any])

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
        try:
            _settings = get_settings()
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load configuration: {e}[/yellow]")
            console.print("[yellow]Using default configuration for basic commands.[/yellow]")
            # Create a minimal settings object for basic functionality
            class MinimalSettings:
                environment = "development"
                debug = False
                data = type('Data', (), {
                    'primary_source': 'yfinance',
                    'symbols': ['AAPL', 'GOOGL', 'MSFT'],
                    'start_date': '2024-01-01',
                    'end_date': '2024-12-31',
                    'timeframe': '1d',
                    'data_path': 'data/'
                })()
                agent = type('Agent', (), {'agent_type': 'ppo'})()
                risk = type('Risk', (), {'max_position_size': 0.1})()
                execution = type('Execution', (), {
                    'broker': 'alpaca',
                    'paper_trading': True
                })()
                infrastructure = type('Infrastructure', (), {'max_workers': 4})()
            
            _settings = MinimalSettings()
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

scenario_app = typer.Typer(
    name="scenario",
    help="Agent scenario evaluation with synthetic data",
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
app.add_typer(scenario_app, help="Agent scenario evaluation")
app.add_typer(trade_app, help="Live trading operations")


@app.callback()
def main(
    config_file: Annotated[Path | None, typer.Option("--config", "-c", help="Path to configuration file")] = None,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="Increase verbosity (use multiple times for more detail)"),
    ] = 0,
    env_file: Annotated[Path | None, typer.Option("--env-file", help="Path to environment file (.env)")] = None,
) -> None:
    """
    Trading RL Agent - Production-grade live trading system.

    A hybrid reinforcement learning trading system that combines CNN+LSTM supervised learning
    with deep RL optimization for algorithmic trading.
    """

    global verbose_count, _settings

    # Set defaults if not provided
    if config_file is None:
        config_file = Path("config.yaml")
    if env_file is None:
        env_file = Path(".env")

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
        # Note: setup_logging_for_typer already called above, no need to call again
        pass


@app.command()
def version() -> None:
    """Show version information."""
    try:
        from . import __version__
        version_str = __version__
    except ImportError:
        version_str = "2.0.0"  # Fallback version

    table = Table(title="Trading RL Agent")
    table.add_column("Component", style="cyan")
    table.add_column("Version", style="green")

    table.add_row("Trading RL Agent", version_str)
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

        # Resolve date values before printing
        resolved_start_date = start_date or settings.data.start_date
        resolved_end_date = end_date or settings.data.end_date

        console.print("[green]Starting download of all configured datasets[/green]")
        console.print(f"[cyan]Symbols: {', '.join(symbols)}[/cyan]")
        console.print(f"[cyan]Date range: {resolved_start_date} to {resolved_end_date}[/cyan]")
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
                start_date=resolved_start_date,
                end_date=resolved_end_date,
                interval=timeframe or "1d",
                cache_dir=str(output_dir / "cache"),
                max_workers=settings.infrastructure.max_workers,
            )
        else:
            console.print("[yellow]Using sequential data fetching...[/yellow]")
            data_dict = fetch_data_with_retry(
                symbols=symbols,
                start_date=resolved_start_date,
                end_date=resolved_end_date,
                interval=timeframe or "1d",
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
                output_file = output_dir / f"{symbol}_{resolved_start_date}_{resolved_end_date}_{timeframe}.parquet"
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
        resolved_start_date = start_date or settings.data.start_date
        resolved_end_date = end_date or settings.data.end_date
        output_dir = output_dir or Path(settings.data.data_path)
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        console.print("[green]Starting download for specific symbols[/green]")
        console.print(f"[cyan]Symbols: {', '.join(symbol_list)}[/cyan]")
        console.print(f"[cyan]Date range: {resolved_start_date} to {resolved_end_date}[/cyan]")
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
                interval=timeframe or settings.data.timeframe or "1d",
                cache_dir=str(output_dir / "cache"),
                max_workers=settings.infrastructure.max_workers,
            )
        else:
            console.print("[yellow]Using sequential data fetching...[/yellow]")
            data_dict = fetch_data_with_retry(
                symbols=symbol_list,
                start_date=start_date or settings.data.start_date,
                end_date=end_date or settings.data.end_date,
                interval=timeframe or settings.data.timeframe or "1d",
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
                output_file = output_dir / f"{symbol}_{resolved_start_date}_{resolved_end_date}_{timeframe}.parquet"
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
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            interval=timeframe or "1d",
            show_progress=True,
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
    try:
        if config_file is None:
            config_file = Path("config.yaml")  # Provide a default config file

        console.print(f"[green]Processing data with config: {config_file}[/green]")
        console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
        console.print(f"[cyan]Force rebuild: {force_rebuild}[/cyan]")
        console.print(f"[cyan]Parallel processing: {parallel}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print("[blue]PLACEHOLDER: Data processing pipeline would run here[/blue]")
        console.print("[blue]Target module: src/trading_rl_agent/data/pipeline.py[/blue]")

    except Exception as e:
        console.print(f"[red]Error during processing: {e}[/red]")
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command()
def standardize(
    input_path: Path | None = DEFAULT_CONFIG_FILE,
    output_path: Path = DEFAULT_STANDARDIZED_OUTPUT,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
) -> None:
    """Standardize data format."""
    try:
        if input_path is None:
            input_path = Path("data/raw")  # Provide a default input path

        console.print(f"[green]Standardizing data from: {input_path}[/green]")
        console.print(f"[cyan]Output path: {output_path}[/cyan]")
        console.print(f"[cyan]Method: {method}[/cyan]")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        console.print("[blue]PLACEHOLDER: Data standardization would run here[/blue]")
        console.print("[blue]Target module: src/trading_rl_agent/data/data_standardizer.py[/blue]")

    except Exception as e:
        console.print(f"[red]Error during standardization: {e}[/red]")
        logger.error(f"Standardization failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command()
def pipeline(
    config_path: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = DEFAULT_PIPELINE_OUTPUT,
) -> None:
    """Run complete data pipeline."""
    try:
        if config_path is None:
            config_path = Path("config.yaml")  # Provide a default config path

        console.print(f"[green]Running data pipeline with config: {config_path}[/green]")
        console.print(f"[cyan]Output directory: {output_dir}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print("[blue]PLACEHOLDER: Complete data pipeline would run here[/blue]")
        console.print("[blue]Target module: src/trading_rl_agent/data/pipeline.py[/blue]")

    except Exception as e:
        console.print(f"[red]Error during pipeline execution: {e}[/red]")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


# ============================================================================
# TRAIN SUB-APP COMMANDS
# ============================================================================


@train_app.command(name="cnn_lstm")
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
        "[blue]Target module: src/trading_rl_agent/training/optimized_trainer.py - OptimizedTrainingManager.train()[/blue]",
    )


@train_app.command(name="rl")
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


@train_app.command(name="hybrid")
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


@train_app.command(name="hyperopt")
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
        "[blue]Target module: src/trading_rl_agent/training/optimized_trainer.py - OptimizedTrainingManager.evaluate()[/blue]",
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
        "[blue]Target module: src/trading_rl_agent/eval/walk_forward_analyzer.py - WalkForwardAnalyzer[/blue]",
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
        "[blue]Target module: src/trading_rl_agent/core/live_trading.py - LiveTradingEngine, TradingSession[/blue]",
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
        "[blue]Target module: src/trading_rl_agent/core/live_trading.py - LiveTradingEngine.stop_all_sessions()[/blue]",
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


@trade_app.command()
def paper(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    symbols: str = DEFAULT_PAPER_SYMBOLS,
    duration: str = DEFAULT_PAPER_DURATION,
) -> None:
    """
    Start a paper trading session with simulated trades.
    """
    console.print(f"[blue]PLACEHOLDER: Would start paper trading for {symbols} for {duration}[/blue]")
    console.print("[blue]Target module: Paper trading utilities[/blue]")


# ============================================================================
# SCENARIO SUB-APP COMMANDS
# ============================================================================


@scenario_app.command()
def scenario_evaluate(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    agent_type: str = "moving_average",
    output_dir: Path = Path("outputs/scenario_evaluation"),
    seed: int = 42,
    save_reports: bool = True,
    save_visualizations: bool = True,
) -> None:
    """
    Evaluate agent performance across synthetic market scenarios.

    Tests agent robustness and adaptation to different market regimes
    including trend following, mean reversion, volatility breakouts,
    market crises, and regime changes.
    """
    console.print("[bold blue]Evaluating agent across market scenarios...[/bold blue]")

    try:
        from examples.scenario_evaluation_example import (
            create_mean_reversion_agent,
            create_momentum_agent,
            create_simple_moving_average_agent,
            create_volatility_breakout_agent,
        )
        from trading_rl_agent.eval import AgentScenarioEvaluator

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scenario evaluator
        evaluator = AgentScenarioEvaluator(seed=seed)

        # Create agents based on type
        agents = {
            "moving_average": create_simple_moving_average_agent(window=20),
            "momentum": create_momentum_agent(lookback=10),
            "mean_reversion": create_mean_reversion_agent(lookback=20),
            "volatility_breakout": create_volatility_breakout_agent(vol_window=20),
        }

        if agent_type not in agents:
            console.print(f"[red]Unknown agent type: {agent_type}[/red]")
            console.print(f"Available types: {list(agents.keys())}")
            raise typer.Exit(1)

        agent = agents[agent_type]

        # Run evaluation
        results = evaluator.evaluate_agent(
            agent=agent,
            agent_name=agent_type.replace("_", " ").title(),
        )

        # Print summary
        evaluator.print_evaluation_summary(results)

        # Save reports and visualizations
        if save_reports:
            report_path = output_dir / f"{agent_type}_evaluation_report.md"
            evaluator.generate_evaluation_report(results, report_path)
            console.print(f"📄 Report saved: {report_path}")

        if save_visualizations:
            viz_path = output_dir / f"{agent_type}_evaluation.png"
            evaluator.create_visualization(results, viz_path)
            console.print(f"📊 Visualization saved: {viz_path}")

        console.print("[bold green]✅ Scenario evaluation complete![/bold green]")
        console.print(f"📁 Results saved to: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error during scenario evaluation: {e}[/red]")
        # Note: verbose_count is not available in this scope, so we'll just exit
        raise typer.Exit(1) from None


@scenario_app.command()
def scenario_compare(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = Path("outputs/scenario_evaluation"),
    seed: int = 42,
    save_reports: bool = True,
    save_visualizations: bool = True,
) -> None:
    """
    Compare multiple agents across synthetic market scenarios.

    Evaluates and compares different agent strategies to identify
    the most robust and adaptive trading approaches.
    """
    console.print("[bold blue]Comparing agents across market scenarios...[/bold blue]")

    try:
        from examples.scenario_evaluation_example import (
            create_mean_reversion_agent,
            create_momentum_agent,
            create_simple_moving_average_agent,
            create_volatility_breakout_agent,
        )
        from trading_rl_agent.eval import AgentScenarioEvaluator

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scenario evaluator
        evaluator = AgentScenarioEvaluator(seed=seed)

        # Create multiple agents
        agents = {
            "Moving Average": create_simple_moving_average_agent(window=20),
            "Momentum": create_momentum_agent(lookback=10),
            "Mean Reversion": create_mean_reversion_agent(lookback=20),
            "Volatility Breakout": create_volatility_breakout_agent(vol_window=20),
        }

        # Evaluate all agents
        all_results = {}
        for agent_name, agent in agents.items():
            console.print(f"📊 Evaluating {agent_name}...")

            results = evaluator.evaluate_agent(
                agent=agent,
                agent_name=agent_name,
            )
            all_results[agent_name] = results

        # Create comparison report
        comparison_report = f"""
# Agent Scenario Comparison Report

## Summary
- **Evaluation Date**: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Total Agents Tested**: {len(agents)}
- **Scenarios Per Agent**: {len(evaluator.scenarios)}

## Agent Rankings

| Agent | Overall Score | Robustness | Adaptation | Pass Rate |
|-------|---------------|------------|------------|-----------|
"""

        for agent_name, results in all_results.items():
            comparison_report += f"| {agent_name} | {results['overall_score']:.3f} | {results['robustness_score']:.3f} | {results['adaptation_score']:.3f} | {results['aggregate_metrics']['pass_rate']:.1%} |\n"

        comparison_report += f"""

## Detailed Results

Each agent has been evaluated across multiple market scenarios including:
- Trend Following Markets
- Mean Reversion Markets
- Volatility Breakout Markets
- Market Crisis Scenarios
- Regime Change Scenarios

Detailed reports and visualizations have been saved to: {output_dir}

## Key Insights

1. **Best Overall Performance**: {max(all_results.keys(), key=lambda k: all_results[k]["overall_score"])} achieved the highest overall score
2. **Most Robust**: {max(all_results.keys(), key=lambda k: all_results[k]["robustness_score"])} showed the most consistent performance
3. **Best Adaptation**: {max(all_results.keys(), key=lambda k: all_results[k]["adaptation_score"])} adapted best to challenging scenarios

## Recommendations

- Use {max(all_results.keys(), key=lambda k: all_results[k]["overall_score"])} for general market conditions
- Consider scenario-specific agent selection for specialized strategies
- Monitor performance during regime changes and market crises
- Regular re-evaluation recommended as market conditions evolve
"""

        # Save comparison report
        if save_reports:
            report_path = output_dir / "agent_comparison_report.md"
            with open(report_path, "w") as f:
                f.write(comparison_report)
            console.print(f"📄 Comparison report saved: {report_path}")

        # Save individual agent reports
        if save_reports:
            for agent_name, results in all_results.items():
                agent_report_path = output_dir / f"{agent_name.lower().replace(' ', '_')}_report.md"
                evaluator.generate_evaluation_report(results, agent_report_path)

        # Save visualizations
        if save_visualizations:
            for agent_name, results in all_results.items():
                viz_path = output_dir / f"{agent_name.lower().replace(' ', '_')}_evaluation.png"
                evaluator.create_visualization(results, viz_path)

        console.print("[bold green]✅ Agent comparison complete![/bold green]")
        console.print(f"📁 Results saved to: {output_dir}")

    except Exception as e:
        console.print(f"[red]Error during agent comparison: {e}[/red]")
        if verbose_count > 0:
            raise
        raise typer.Exit(1) from None


@scenario_app.command()
def custom(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    agent_type: str = "moving_average",
    scenario_name: str = "strong_uptrend",
    output_dir: Path = Path("outputs/scenario_evaluation"),
    seed: int = 42,
    save_reports: bool = True,
) -> None:
    """
    Evaluate agent on custom market scenarios.

    Tests agent performance on specific market conditions like
    strong trends, high volatility crises, or sideways markets.
    """
    console.print(f"[bold blue]Evaluating {agent_type} on {scenario_name} scenario...[/bold blue]")

    try:
        from examples.scenario_evaluation_example import (
            create_custom_scenarios,
            create_mean_reversion_agent,
            create_momentum_agent,
            create_simple_moving_average_agent,
            create_volatility_breakout_agent,
        )
        from trading_rl_agent.eval import AgentScenarioEvaluator

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scenario evaluator
        evaluator = AgentScenarioEvaluator(seed=seed)

        # Create agents
        agents = {
            "moving_average": create_simple_moving_average_agent(window=20),
            "momentum": create_momentum_agent(lookback=10),
            "mean_reversion": create_mean_reversion_agent(lookback=20),
            "volatility_breakout": create_volatility_breakout_agent(vol_window=20),
        }

        if agent_type not in agents:
            console.print(f"[red]Unknown agent type: {agent_type}[/red]")
            console.print(f"Available types: {list(agents.keys())}")
            raise typer.Exit(1)

        agent = agents[agent_type]

        # Get custom scenarios
        custom_scenarios = create_custom_scenarios()
        scenario_map = {s.name.lower().replace(" ", "_"): s for s in custom_scenarios}

        if scenario_name not in scenario_map:
            console.print(f"[red]Unknown scenario: {scenario_name}[/red]")
            console.print(f"Available scenarios: {list(scenario_map.keys())}")
            raise typer.Exit(1)

        selected_scenario = scenario_map[scenario_name]

        # Run evaluation
        results = evaluator.evaluate_agent(
            agent=agent,
            agent_name=f"{agent_type.replace('_', ' ').title()} ({selected_scenario.name})",
            custom_scenarios=[selected_scenario],
        )

        # Print summary
        evaluator.print_evaluation_summary(results)

        # Save report
        if save_reports:
            report_path = output_dir / f"{agent_type}_{scenario_name}_report.md"
            evaluator.generate_evaluation_report(results, report_path)
            console.print(f"📄 Report saved: {report_path}")

        console.print("[bold green]✅ Custom scenario evaluation complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error during custom scenario evaluation: {e}[/red]")
        if verbose_count > 0:
            raise
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
