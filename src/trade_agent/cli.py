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
    from trade_agent.logging_conf import get_logger, setup_logging_for_typer
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

        def load_settings(_config_path: "Path | None" = None, _env_file: "Path | None" = None) -> Any:  # type: ignore[misc]
            return None


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
_settings: Any = None


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
                data = type(
                    "Data",
                    (),
                    {
                        "primary_source": "yfinance",
                        "symbols": ["AAPL", "GOOGL", "MSFT"],  # Using default symbols
                        "start_date": "2024-01-01",
                        "end_date": "2024-12-31",
                        "timeframe": "1d",
                        "data_path": "data/",
                    },
                )()
                agent = type("Agent", (), {"agent_type": "ppo"})()
                risk = type("Risk", (), {"max_position_size": 0.1})()
                execution = type("Execution", (), {"broker": "alpaca", "paper_trading": True})()
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

# Sub-apps
data_app = typer.Typer(
    name="data",
    help="Data pipeline operations: download, prepare, pipeline",
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
        typer.Option(
            "--verbose",
            "-v",
            count=True,
            help="Increase verbosity (use multiple times for more detail)",
        ),
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
    table.add_row(
        "Risk Management",
        "Enabled" if settings.risk.max_position_size > 0 else "Disabled",
    )
    table.add_row("Execution Broker", settings.execution.broker)
    table.add_row("Paper Trading", str(settings.execution.paper_trading))

    console.print(table)


# ============================================================================
# DATA SUB-APP COMMANDS
# ============================================================================


@data_app.command()
def download_all(
    start_date: str | None = None,
    end_date: str | None = None,
    source: str | None = DEFAULT_SOURCE,
    timeframe: str | None = "1d",
    parallel: bool = DEFAULT_PARALLEL,
    _force: bool = DEFAULT_FORCE,
) -> None:
    """
    Download all available market data from yfinance with intelligent auto-switching.

    Downloads comprehensive market data including:
    - Major US stocks (S&P 500 top components)
    - Popular ETFs
    - Major forex pairs
    - Cryptocurrencies
    - Market indices
    - Commodities

    Auto-switching behavior based on Yahoo Finance limitations:
    - 1m data: last 7 days
    - 5m/15m/30m data: last 60 days
    - 1h data: last 730 days (2 years)
    - 1d data: unlimited (default for long periods)

    The system automatically adjusts timeframe and date range to work within
    Yahoo Finance's free tier limitations while maximizing data quality.
    Data is automatically organized in the standard data directory structure.
    """
    try:
        # Load configuration
        settings = get_config_manager()

        # Use config values with CLI overrides
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        # Set default date range to last 10 years
        from datetime import datetime, timedelta

        if not start_date:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=3650)  # 10 years
            start_date = start_dt.strftime("%Y-%m-%d")

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Auto-adjust timeframe and date range based on Yahoo Finance limitations
        def get_optimal_timeframe_and_dates(requested_timeframe: str, start_date: str, end_date: str) -> tuple[str, str, str]:
            """
            Auto-adjust timeframe and date range based on Yahoo Finance limitations.

            Yahoo Finance limits:
            - 1m: last 7 days
            - 5m: last 60 days
            - 15m: last 60 days
            - 30m: last 60 days
            - 1h: last 730 days (2 years)
            - 1d: unlimited
            """
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            date_range_days = (end_dt - start_dt).days

            # Define timeframe limitations
            timeframe_limits = {
                "1m": 7,
                "2m": 60,
                "5m": 60,
                "15m": 60,
                "30m": 60,
                "60m": 730,
                "1h": 730,
                "90m": 730,
                "1d": float("inf"),  # No limit
                "5d": float("inf"),
                "1wk": float("inf"),
                "1mo": float("inf"),
                "3mo": float("inf")
            }

            # Check if requested timeframe is supported for the date range
            if requested_timeframe in timeframe_limits:
                max_days = timeframe_limits[requested_timeframe]

                if date_range_days <= max_days:
                    # Requested timeframe works for the date range
                    return requested_timeframe, start_date, end_date
                else:
                    # Need to adjust - try to find the best alternative
                    console.print(f"[yellow]⚠️  Requested {requested_timeframe} data for {date_range_days} days exceeds Yahoo Finance limit of {max_days} days[/yellow]")

                    # For long periods, default to daily data
                    if date_range_days > 730:
                        console.print("[blue]🔄 Auto-switching to daily (1d) data for long-term analysis[/blue]")
                        return "1d", start_date, end_date

                    # For medium periods, try hourly
                    elif date_range_days > 60:
                        console.print("[blue]🔄 Auto-switching to hourly (1h) data for medium-term analysis[/blue]")
                        # Adjust start date to fit within 730 days
                        new_start_dt = end_dt - timedelta(days=730)
                        new_start_date = new_start_dt.strftime("%Y-%m-%d")
                        return "1h", new_start_date, end_date

                    # For short periods, try 30m
                    elif date_range_days > 7:
                        console.print("[blue]🔄 Auto-switching to 30-minute data for short-term analysis[/blue]")
                        # Adjust start date to fit within 60 days
                        new_start_dt = end_dt - timedelta(days=60)
                        new_start_date = new_start_dt.strftime("%Y-%m-%d")
                        return "30m", new_start_date, end_date

                    # For very short periods, try 5m
                    else:
                        console.print("[blue]🔄 Auto-switching to 5-minute data for intraday analysis[/blue]")
                        # Adjust start date to fit within 60 days
                        new_start_dt = end_dt - timedelta(days=60)
                        new_start_date = new_start_dt.strftime("%Y-%m-%d")
                        return "5m", new_start_date, end_date
            else:
                # Unknown timeframe, default to daily
                console.print(f"[yellow]⚠️  Unknown timeframe '{requested_timeframe}', defaulting to daily (1d)[/yellow]")
                return "1d", start_date, end_date

        # Apply auto-adjustment
        original_timeframe = timeframe
        original_start_date = start_date
        original_end_date = end_date

        timeframe, start_date, end_date = get_optimal_timeframe_and_dates(timeframe, start_date, end_date)

        # Show what we're actually downloading
        if timeframe != original_timeframe or start_date != original_start_date:
            console.print("[cyan]📊 Download Strategy:[/cyan]")
            console.print(f"[cyan]  Requested: {original_timeframe} from {original_start_date} to {original_end_date}[/cyan]")
            console.print(f"[cyan]  Adjusted:  {timeframe} from {start_date} to {end_date}[/cyan]")
            console.print("[cyan]  Reason:    Yahoo Finance limitations[/cyan]")
            console.print()

        # Create organized directory structure in the standard data location
        base_data_dir = Path(settings.data.data_path)
        base_output_dir = base_data_dir / "raw" / "comprehensive"

        # Create organized subdirectories by asset type
        asset_dirs = {
            "stocks": base_output_dir / "stocks",
            "etfs": base_output_dir / "etfs",
            "forex": base_output_dir / "forex",
            "crypto": base_output_dir / "crypto",
            "indices": base_output_dir / "indices",
            "commodities": base_output_dir / "commodities"
        }

        # Create all directories
        for dir_path in asset_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Get comprehensive symbols with validation
        try:
            from .data.market_symbols import COMPREHENSIVE_SYMBOLS
            from .data.symbol_validator import get_popular_symbols, get_symbol_info, validate_symbols

            # Use popular symbols as the base list
            all_symbols = get_popular_symbols()

            # First validate symbols for data availability
            valid_symbols, invalid_symbols = validate_symbols(all_symbols, min_data_points=5)

            # Get symbol info for filtering
            symbol_info = get_symbol_info(valid_symbols)

            # Apply market cap filter only to stocks (not indices, commodities, forex, crypto)
            filtered_symbols = []
            for symbol in valid_symbols:
                info = symbol_info.get(symbol, {})
                quote_type = info.get("quote_type", "Unknown")

                # Skip market cap filtering for non-equity assets
                if quote_type in ["INDEX", "FUTURE", "CRYPTOCURRENCY"] or "=X" in symbol or "-USD" in symbol:
                    filtered_symbols.append(symbol)
                else:
                    # Apply market cap filter only to stocks/equities
                    market_cap = info.get("market_cap", 0)
                    if market_cap >= 1e9:  # Minimum $1B market cap for stocks
                        filtered_symbols.append(symbol)

            valid_symbols = filtered_symbols

            # Group symbols by type for organization
            symbols_by_type = {
                "stocks": [s for s in valid_symbols if not any(x in s for x in ["=X", "-USD", "^", "=F"])],
                "forex": [s for s in valid_symbols if "=X" in s],
                "crypto": [s for s in valid_symbols if "-USD" in s],
                "indices": [s for s in valid_symbols if s.startswith("^")],
                "commodities": [s for s in valid_symbols if "=F" in s],
                "etfs": [s for s in valid_symbols if s in COMPREHENSIVE_SYMBOLS["etfs"]]
            }
        except ImportError:
            # Fallback to symbols from market_symbols if symbol validator is not available
            console.print("[yellow]Symbol validator not available, using symbols from market_symbols[/yellow]")
            from .data.market_symbols import COMPREHENSIVE_SYMBOLS
            valid_symbols = []
            for symbols in COMPREHENSIVE_SYMBOLS.values():
                valid_symbols.extend(symbols)
            symbols_by_type = COMPREHENSIVE_SYMBOLS.copy()

        console.print("[green]Starting comprehensive market data download[/green]")
        console.print(f"[cyan]Total symbols: {len(valid_symbols)}[/cyan]")
        console.print(f"[cyan]Date range: {start_date} to {end_date}[/cyan]")
        console.print(f"[cyan]Source: {source}[/cyan]")
        console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
        console.print(f"[cyan]Output: {base_output_dir}[/cyan]")
        console.print("[cyan]Organized by asset type in subdirectories[/cyan]")

        # Import and use actual download functions
        from .data.parallel_data_fetcher import (
            fetch_data_parallel,
            fetch_data_with_retry,
        )

        # Choose download method based on parallel flag
        if parallel:
            console.print("[yellow]Using parallel data fetching...[/yellow]")
            data_dict = fetch_data_parallel(
                symbols=valid_symbols,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe or "1d",
                cache_dir=str(base_data_dir / "cache"),
                max_workers=settings.infrastructure.max_workers,
            )
        else:
            console.print("[yellow]Using sequential data fetching...[/yellow]")
            data_dict = fetch_data_with_retry(
                symbols=valid_symbols,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe or "1d",
                cache_dir=str(base_data_dir / "cache"),
                max_retries=3,
            )

        # Organize and save results by asset type
        total_rows = 0
        successful_downloads = 0
        failed_downloads = 0
        stats_by_type = {cat_type: {"success": 0, "failed": 0, "rows": 0} for cat_type in symbols_by_type}

        for symbol, data in data_dict.items():
            # Determine asset type for this symbol
            asset_type: str = "stocks"  # Default to stocks
            for type_name, type_symbols in symbols_by_type.items():
                if symbol in type_symbols:
                    asset_type = type_name
                    break

            if not data.empty:
                successful_downloads += 1
                rows = len(data)
                total_rows += rows
                stats_by_type[asset_type]["success"] += 1
                stats_by_type[asset_type]["rows"] += rows

                console.print(f"[green]✓ {symbol}: {rows:,} rows ({asset_type})[/green]")

                # Save to appropriate subdirectory
                output_file = asset_dirs[asset_type] / f"{symbol}_{start_date}_{end_date}_{timeframe}.csv"
                data.to_csv(output_file, index=False)
                console.print(f"[blue]  Saved to: {output_file}[/blue]")
            else:
                failed_downloads += 1
                stats_by_type[asset_type]["failed"] += 1
                console.print(f"[red]✗ {symbol}: No data available ({asset_type})[/red]")

        # Print summary statistics
        console.print("\n[green]Download completed![/green]")
        console.print(f"[cyan]Overall: {successful_downloads}/{len(valid_symbols)} successful downloads[/cyan]")
        console.print(f"[cyan]Total rows downloaded: {total_rows:,}[/cyan]")
        console.print(f"[cyan]Data organized in: {base_output_dir}[/cyan]")

        console.print("\n[cyan]Breakdown by asset type:[/cyan]")
        for asset_type, stats in stats_by_type.items():
            total_for_type = stats["success"] + stats["failed"]
            if total_for_type > 0:
                success_rate = (stats["success"] / total_for_type) * 100
                console.print(f"[blue]  {asset_type.capitalize()}: {stats['success']}/{total_for_type} ({success_rate:.1f}%) - {stats['rows']:,} rows[/blue]")

        # Create a metadata file for easy access
        metadata = {
            "download_date": datetime.now().isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
            "source": source,
            "total_symbols": len(valid_symbols),
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "total_rows": total_rows,
            "asset_types": list(symbols_by_type.keys()),
            "directory_structure": {
                asset_type: str(dir_path) for asset_type, dir_path in asset_dirs.items()
            }
        }

        metadata_file = base_output_dir / "download_metadata.json"
        import json
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[cyan]Metadata saved to: {metadata_file}[/cyan]")

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
    _force: bool = DEFAULT_FORCE,
) -> None:
    """
    Download data for specific comma-separated instruments.

    Downloads data for the specified symbols using Settings.data values
    with CLI option overrides. Supports all features from download-all including
    symbol validation, organized directory structure, and comprehensive reporting.
    """
    try:
        # Load configuration
        settings = get_config_manager()

        # Parse symbols
        symbol_list = settings.data.symbols if symbols is None else [s.strip() for s in symbols.split(",")]

        # Use config values with CLI overrides
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        # Set default date range if not specified
        from datetime import datetime, timedelta

        if not start_date:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=365)  # 1 year default for symbols
            start_date = start_dt.strftime("%Y-%m-%d")

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Create organized directory structure
        base_data_dir = Path(settings.data.data_path)
        base_output_dir = output_dir or base_data_dir / "raw" / "symbols"

        # Create organized subdirectories by asset type
        asset_dirs = {
            "stocks": base_output_dir / "stocks",
            "etfs": base_output_dir / "etfs",
            "forex": base_output_dir / "forex",
            "crypto": base_output_dir / "crypto",
            "indices": base_output_dir / "indices",
            "commodities": base_output_dir / "commodities"
        }

        # Create all directories
        for dir_path in asset_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        # Use symbol categories from market_symbols
        from .data.market_symbols import COMPREHENSIVE_SYMBOLS
        symbol_categories = COMPREHENSIVE_SYMBOLS.copy()

        # Validate symbols before downloading
        console.print("[yellow]Validating symbols...[/yellow]")

        try:
            from .data.symbol_validator import validate_symbols
            valid_symbols, invalid_symbols = validate_symbols(symbol_list, min_data_points=5)
        except ImportError:
            console.print("[yellow]Symbol validator not available, using basic validation[/yellow]")
            try:
                import yfinance as yf
                valid_symbols = []
                invalid_symbols = []
                for symbol in symbol_list:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                            valid_symbols.append(symbol)
                        else:
                            invalid_symbols.append(symbol)
                    except Exception:
                        invalid_symbols.append(symbol)
            except ImportError:
                console.print("[yellow]yfinance not available for validation, proceeding with all symbols[/yellow]")
                valid_symbols = symbol_list
                invalid_symbols = []

        if invalid_symbols:
            console.print(f"[yellow]Removed {len(invalid_symbols)} invalid symbols: {', '.join(invalid_symbols[:10])}{'...' if len(invalid_symbols) > 10 else ''}[/yellow]")

        console.print(f"[green]Proceeding with {len(valid_symbols)} valid symbols[/green]")

        console.print("[green]Starting download for specific symbols[/green]")
        console.print(f"[cyan]Symbols: {', '.join(valid_symbols)}[/cyan]")
        console.print(f"[cyan]Date range: {start_date} to {end_date}[/cyan]")
        console.print(f"[cyan]Source: {source}[/cyan]")
        console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
        console.print(f"[cyan]Output: {base_output_dir}[/cyan]")
        console.print("[cyan]Organized by asset type in subdirectories[/cyan]")

        # Import and use actual download functions
        from .data.parallel_data_fetcher import (
            fetch_data_parallel,
            fetch_data_with_retry,
        )

        # Choose download method based on parallel flag
        if parallel:
            console.print("[yellow]Using parallel data fetching...[/yellow]")
            data_dict = fetch_data_parallel(
                symbols=valid_symbols,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe or "1d",
                cache_dir=str(base_data_dir / "cache"),
                max_workers=settings.infrastructure.max_workers,
            )
        else:
            console.print("[yellow]Using sequential data fetching...[/yellow]")
            data_dict = fetch_data_with_retry(
                symbols=valid_symbols,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe or "1d",
                cache_dir=str(base_data_dir / "cache"),
                max_retries=3,
            )

        # Organize and save results by asset type
        total_rows = 0
        successful_downloads = 0
        failed_downloads = 0
        stats_by_type = {asset_type: {"success": 0, "failed": 0, "rows": 0} for asset_type in symbol_categories}

        for symbol, data in data_dict.items():
            # Determine asset type for this symbol
            asset_type: str = "stocks"  # Default to stocks
            for type_name, type_symbols in symbol_categories.items():
                if symbol in type_symbols:
                    asset_type = type_name
                    break

            if not data.empty:
                successful_downloads += 1
                rows = len(data)
                total_rows += rows
                stats_by_type[asset_type]["success"] += 1
                stats_by_type[asset_type]["rows"] += rows

                console.print(f"[green]✓ {symbol}: {rows:,} rows ({asset_type})[/green]")

                # Save to appropriate subdirectory
                output_file = asset_dirs[asset_type] / f"{symbol}_{start_date}_{end_date}_{timeframe}.csv"
                data.to_csv(output_file, index=False)
                console.print(f"[blue]  Saved to: {output_file}[/blue]")
            else:
                failed_downloads += 1
                stats_by_type[asset_type]["failed"] += 1
                console.print(f"[red]✗ {symbol}: No data available ({asset_type})[/red]")

        # Print summary statistics
        console.print("\n[green]Download completed![/green]")
        console.print(f"[cyan]Overall: {successful_downloads}/{len(valid_symbols)} successful downloads[/cyan]")
        console.print(f"[cyan]Total rows downloaded: {total_rows:,}[/cyan]")
        console.print(f"[cyan]Data organized in: {base_output_dir}[/cyan]")

        console.print("\n[cyan]Breakdown by asset type:[/cyan]")
        for asset_type, stats in stats_by_type.items():
            total_for_type = stats["success"] + stats["failed"]
            if total_for_type > 0:
                success_rate = (stats["success"] / total_for_type) * 100
                console.print(f"[blue]  {asset_type.capitalize()}: {stats['success']}/{total_for_type} ({success_rate:.1f}%) - {stats['rows']:,} rows[/blue]")

        # Create a metadata file for easy access
        metadata = {
            "download_date": datetime.now().isoformat(),
            "start_date": start_date,
            "end_date": end_date,
            "timeframe": timeframe,
            "source": source,
            "total_symbols": len(valid_symbols),
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "total_rows": total_rows,
            "asset_types": list(symbol_categories.keys()),
            "directory_structure": {
                asset_type: str(dir_path) for asset_type, dir_path in asset_dirs.items()
            }
        }

        metadata_file = base_output_dir / "download_metadata.json"
        import json
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[cyan]Metadata saved to: {metadata_file}[/cyan]")

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
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else settings.data.symbols

        # Use config values with CLI overrides
        start_date = settings.data.start_date
        end_date = settings.data.end_date
        output_dir = output_dir or Path(settings.data.data_path)
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        # NEW: Validate timeframe consistency
        console.print(f"[green]Checking data freshness and refreshing if older than {days} days[/green]")
        console.print(f"[cyan]Symbols: {', '.join(symbol_list)}[/cyan]")
        console.print(f"[cyan]Cache TTL: {days} days[/cyan]")
        console.print(f"[cyan]Source: {source}[/cyan]")
        console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]")

        # NEW: Check for existing files with different timeframes
        existing_files = []
        for symbol in symbol_list:
            # Look for any existing files for this symbol
            pattern = f"{symbol}_*_{timeframe}.parquet"
            list(output_dir.glob(pattern))

            # Also check for files with different timeframes
            all_pattern = f"{symbol}_*.parquet"
            all_files = list(output_dir.glob(all_pattern))

            different_timeframes = [f for f in all_files if f"{timeframe}.parquet" not in str(f)]

            if different_timeframes:
                console.print(f"[yellow]⚠️  Warning: Found existing files for {symbol} with different timeframes:[/yellow]")
                for file in different_timeframes:
                    console.print(f"[yellow]    {file.name}[/yellow]")
                console.print(f"[yellow]  This refresh will create: {symbol}_{start_date}_{end_date}_{timeframe}.parquet[/yellow]")
                existing_files.extend(different_timeframes)

        if existing_files:
            console.print("[yellow]Note: Different timeframe files will be preserved separately[/yellow]")

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
def prepare(
    input_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    force_rebuild: bool = DEFAULT_FORCE_REBUILD,
    parallel: bool = DEFAULT_PARALLEL,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
    save_standardizer: bool = True,
) -> None:
    """
    Process and standardize downloaded data in one command.

    This command combines data processing and standardization into a single step,
    making it easier to prepare data for training and inference.
    """
    try:
        # Set default input path if not provided
        if input_path is None:
            input_path = Path("data/raw")

        # Set default config file if not provided
        if config_file is None:
            config_file = Path("config.yaml")

        console.print(f"[green]Preparing data from: {input_path}[/green]")
        console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
        console.print(f"[cyan]Config file: {config_file}[/cyan]")
        console.print(f"[cyan]Force rebuild: {force_rebuild}[/cyan]")
        console.print(f"[cyan]Parallel processing: {parallel}[/cyan]")
        console.print(f"[cyan]Standardization method: {method}[/cyan]")
        console.print(f"[cyan]Save standardizer: {save_standardizer}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Import required modules
        from .data.data_standardizer import create_standardized_dataset
        from .data.pipeline import DataPipeline

        # Step 1: Process the data
        console.print("\n[blue]Step 1: Processing data...[/blue]")
        DataPipeline()

        # Load and process data
        if input_path.is_file():
            # Single file
            import pandas as pd
            df = pd.read_csv(input_path)
            console.print(f"[green]Loaded data from: {input_path}[/green]")
            console.print(f"[cyan]Data shape: {df.shape}[/cyan]")
        elif input_path.is_dir():
            # Directory with multiple files
            csv_files = list(input_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in {input_path}")

            import pandas as pd
            dfs = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                df["source_file"] = csv_file.name
                dfs.append(df)

            df = pd.concat(dfs, ignore_index=True)
            console.print(f"[green]Loaded {len(csv_files)} files from: {input_path}[/green]")
            console.print(f"[cyan]Combined data shape: {df.shape}[/cyan]")
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Step 2: Standardize the data
        console.print("\n[blue]Step 2: Standardizing data...[/blue]")

        # Create standardized dataset
        standardized_df, standardizer = create_standardized_dataset(
            df=df,
            save_path=str(output_dir / "data_standardizer.pkl") if save_standardizer else None
        )

        console.print("[green]Standardization complete![/green]")
        console.print(f"[cyan]Original features: {len(df.columns)}[/cyan]")
        console.print(f"[cyan]Standardized features: {standardizer.get_feature_count()}[/cyan]")
        console.print(f"[cyan]Output shape: {standardized_df.shape}[/cyan]")

        # Step 3: Save processed data
        console.print("\n[blue]Step 3: Saving processed data...[/blue]")

        # Save standardized data
        output_file = output_dir / "standardized_data.csv"
        standardized_df.to_csv(output_file, index=False)
        console.print(f"[green]Saved standardized data to: {output_file}[/green]")

        # Save feature summary
        feature_summary = {
            "total_features": standardizer.get_feature_count(),
            "feature_names": standardizer.get_feature_names(),
            "data_shape": standardized_df.shape,
            "standardization_method": method,
            "missing_value_strategies": standardizer.missing_value_strategies
        }

        import json
        summary_file = output_dir / "feature_summary.json"
        with open(summary_file, "w") as f:
            json.dump(feature_summary, f, indent=2)
        console.print(f"[green]Saved feature summary to: {summary_file}[/green]")

        console.print("\n[green]✅ Data preparation complete![/green]")
        console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
        console.print("[cyan]Files created:[/cyan]")
        console.print("  - standardized_data.csv")
        if save_standardizer:
            console.print("  - data_standardizer.pkl")
            console.print("  - data_standardizer.json")
        console.print("  - feature_summary.json")

        # Step 4: Cleanup raw data for organization
        console.print("\n[blue]Step 4: Cleaning up raw data...[/blue]")
        raw_data_path = Path("data/raw")
        if raw_data_path.exists() and raw_data_path.is_dir():
            try:
                # Count files before deletion for reporting
                files_to_delete = list(raw_data_path.glob("*"))
                file_count = len([f for f in files_to_delete if f.is_file()])
                dir_count = len([f for f in files_to_delete if f.is_dir()])

                # Remove all contents of data/raw
                import shutil
                shutil.rmtree(raw_data_path)
                raw_data_path.mkdir(parents=True, exist_ok=True)  # Recreate empty directory

                console.print(f"[green]✓ Cleaned up {file_count} files and {dir_count} directories from {raw_data_path}[/green]")
                console.print(f"[cyan]Raw data directory {raw_data_path} is now empty and ready for new downloads[/cyan]")
            except Exception as cleanup_error:
                console.print(f"[yellow]⚠️  Warning: Could not clean up raw data directory: {cleanup_error}[/yellow]")
                console.print("[yellow]You may need to manually clean up the data/raw directory[/yellow]")
        else:
            console.print(f"[cyan]Raw data directory {raw_data_path} does not exist or is not a directory[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during data preparation: {e}[/red]")
        logger.error(f"Data preparation failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command()
def pipeline(
    # Pipeline step options
    download: bool = typer.Option(False, "--download", "-d", help="Download market data"),
    process: bool = typer.Option(False, "--process", "-p", help="Process and standardize data"),
    run: bool = typer.Option(False, "--run", "-r", help="Run complete pipeline end-to-end"),

    # Common parameters
    config_path: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = DEFAULT_PIPELINE_OUTPUT,
    symbols: str | None = DEFAULT_SYMBOLS_STR,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    source: str | None = DEFAULT_SOURCE,
    timeframe: str | None = DEFAULT_TIMEFRAME,
    parallel: bool = DEFAULT_PARALLEL,
    force: bool = DEFAULT_FORCE,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
    save_standardizer: bool = True,

    # Process-specific parameters
    input_path: Path | None = None,
    force_rebuild: bool = DEFAULT_FORCE_REBUILD,
) -> None:
    """
    Data pipeline operations with flexible step selection.

    Use options to specify which pipeline steps to run:
    - --download: Download market data from specified sources
    - --process: Process and standardize downloaded data
    - --run: Execute complete pipeline (download → process)

    Note: Dataset building and splitting is handled by individual training commands
    to ensure proper separation between CNN+LSTM (with targets) and RL (without targets).

    If no --symbols are provided with --download, comprehensive market coverage is used.

    Examples:
        # Download only (comprehensive coverage if no symbols)
        python main.py data pipeline --download

        # Download specific symbols
        python main.py data pipeline --download --symbols "AAPL,GOOGL"

        # Process only
        python main.py data pipeline --process --input-path data/raw

        # Run complete pipeline
        python main.py data pipeline --run --symbols "AAPL"

        # Custom combination
        python main.py data pipeline --download --process --symbols "AAPL"
    """
    try:
        # Set defaults
        if config_path is None:
            config_path = Path("config.yaml")

        if start_date is None:
            from datetime import datetime, timedelta
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y-%m-%d")

        if source is None:
            source = "yfinance"

        if timeframe is None:
            timeframe = "1d"

        # Determine which steps to run
        if run:
            # Run complete pipeline
            download = True
            process = True

        if not any([download, process]):
            # No steps specified, show help
            console.print("[yellow]No pipeline steps specified. Use --download, --process, or --run[/yellow]")
            console.print("[yellow]Example: python main.py data pipeline --download --symbols 'AAPL'[/yellow]")
            raise typer.Exit(1)

        console.print("[green]🚀 Data Pipeline Operations[/green]")
        console.print(f"[cyan]Steps: {', '.join(['download' if download else '', 'process' if process else '']).strip(', ')}[/cyan]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]")
        console.print(f"[cyan]Parallel: {parallel}[/cyan]")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Download (if requested)
        if download:
            console.print("\n[blue]Step 1: Downloading data...[/blue]")

                    # Handle symbols - if no symbols provided, use comprehensive market coverage
        if symbols is None:
            # Get comprehensive symbols without validation to avoid API calls
            from .data.market_symbols import get_all_symbols
            symbol_list = get_all_symbols()
            symbols = ",".join(symbol_list)
            console.print("[green]Pipeline Download: Comprehensive Market Coverage[/green]")
            console.print(f"[cyan]Total symbols: {len(symbol_list)}[/cyan]")
            console.print("[yellow]Note: Using comprehensive symbol list without validation for speed[/yellow]")
        else:
            console.print(f"[green]Pipeline Download: {symbols}[/green]")

            console.print(f"[cyan]Source: {source}[/cyan]")
            console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
            console.print(f"[cyan]Date range: {start_date} to {end_date}[/cyan]")
            console.print(f"[cyan]Output: {output_dir / 'raw'}[/cyan]")
            console.print(f"[cyan]Force: {force}[/cyan]")

            # Create raw data directory
            raw_dir = output_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            # Import and run download functionality
            from .data.pipeline import DataPipeline

            pipeline = DataPipeline()

            # Parse symbols
            symbol_list = [s.strip() for s in symbols.split(",")]

            # Download data
            downloaded_files = pipeline.download_data(
                symbols=symbol_list,
                start_date=start_date,
                end_date=end_date,
                output_dir=raw_dir
            )

            console.print(f"[green]✅ Downloaded {len(downloaded_files)} files[/green]")
            for file_path in downloaded_files:
                console.print(f"[cyan]  - {file_path}[/cyan]")

        # Step 2: Process (if requested)
        if process:
            console.print("\n[blue]Step 2: Processing data...[/blue]")

            # Determine input path for processing
            if input_path is None:
                input_path = output_dir / "raw" if download else Path("data/raw")

            console.print(f"[cyan]Input: {input_path}[/cyan]")
            console.print(f"[cyan]Output: {output_dir / 'processed'}[/cyan]")
            console.print(f"[cyan]Method: {method}[/cyan]")
            console.print(f"[cyan]Force rebuild: {force_rebuild}[/cyan]")

            # Create processed data directory
            processed_dir = output_dir / "processed"
            processed_dir.mkdir(parents=True, exist_ok=True)

            # Import and run process functionality
            from .data.prepare import prepare_data

            prepare_data(
                input_path=input_path,
                output_dir=processed_dir,
                config_path=config_path,
                method=method,
                save_standardizer=save_standardizer
            )

            console.print("[green]✅ Data processing completed[/green]")

        console.print("\n[green]✅ Pipeline operations completed successfully![/green]")
        console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
        console.print("[cyan]Pipeline structure:[/cyan]")
        if download:
            console.print("  - raw/ (downloaded data)")
        if process:
            console.print("  - processed/ (standardized data)")
        console.print("[cyan]Note: Dataset building and splitting is handled by individual training commands[/cyan]")

    except Exception as e:
        console.print(f"[red]Error during pipeline operations: {e}[/red]")
        logger.error(f"Pipeline operations failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@scenario_app.command()
def scenario_compare(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    _output_dir: Path = Path("outputs/scenario_evaluation"),
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
        from trade_agent.eval import AgentScenarioEvaluator

        # Create output directory
        _output_dir.mkdir(parents=True, exist_ok=True)

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
            comparison_report += (
                f"| {agent_name} | {results['overall_score']:.3f} | "
                f"{results['robustness_score']:.3f} | {results['adaptation_score']:.3f} | "
                f"{results['aggregate_metrics']['pass_rate']:.1%} |\n"
            )

        comparison_report += f"""

## Detailed Results

Each agent has been evaluated across multiple market scenarios including:
- Trend Following Markets
- Mean Reversion Markets
- Volatility Breakout Markets
- Market Crisis Scenarios
- Regime Change Scenarios

Detailed reports and visualizations have been saved to: {_output_dir}
"""

        # Find best performers
        best_overall = max(all_results.keys(), key=lambda k: all_results[k]["overall_score"])
        most_robust = max(all_results.keys(), key=lambda k: all_results[k]["robustness_score"])
        best_adaptation = max(all_results.keys(), key=lambda k: all_results[k]["adaptation_score"])

        comparison_report += f"""
## Key Insights

1. **Best Overall Performance**: {best_overall} achieved the highest overall score
2. **Most Robust**: {most_robust} showed the most consistent performance
3. **Best Adaptation**: {best_adaptation} adapted best to challenging scenarios

## Recommendations

- Use {best_overall} for general market conditions
- Consider scenario-specific agent selection for specialized strategies
- Monitor performance during regime changes and market crises
- Regular re-evaluation recommended as market conditions evolve
"""

        # Save comparison report
        if save_reports:
            report_path = _output_dir / "agent_comparison_report.md"
            with open(report_path, "w") as f:
                f.write(comparison_report)
            console.print(f"📄 Comparison report saved: {report_path}")

        # Save individual agent reports
        if save_reports:
            for agent_name, results in all_results.items():
                agent_report_path = _output_dir / f"{agent_name.lower().replace(' ', '_')}_report.md"
                evaluator.generate_evaluation_report(results, agent_report_path)

        # Save visualizations
        if save_visualizations:
            for agent_name, results in all_results.items():
                viz_path = _output_dir / f"{agent_name.lower().replace(' ', '_')}_evaluation.png"
                evaluator.create_visualization(results, viz_path)

        console.print("[bold green]✅ Agent comparison complete![/bold green]")
        console.print(f"📁 Results saved to: {_output_dir}")

    except Exception as e:
        console.print(f"[red]Error during agent comparison: {e}[/red]")
        # Note: verbose_count is not available in this scope, so we'll just exit
        raise typer.Exit(1) from None


@scenario_app.command()
def custom(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    agent_type: str = "moving_average",
    scenario_name: str = "strong_uptrend",
    _output_dir: Path = Path("outputs/scenario_evaluation"),
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
        from trade_agent.eval import AgentScenarioEvaluator

        # Create output directory
        _output_dir.mkdir(parents=True, exist_ok=True)

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
            report_path = _output_dir / f"{agent_type}_{scenario_name}_report.md"
            evaluator.generate_evaluation_report(results, report_path)
            console.print(f"📄 Report saved: {report_path}")

        console.print("[bold green]✅ Custom scenario evaluation complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Error during custom scenario evaluation: {e}[/red]")
        # Note: verbose_count is not available in this scope, so we'll just exit
        raise typer.Exit(1) from None


def _get_comprehensive_market_symbols() -> str:
    """
    Get a comprehensive list of market symbols for data download.

    Returns:
        str: Comma-separated string of valid market symbols
    """
    from .data.market_symbols import get_comprehensive_symbols
    return get_comprehensive_symbols()


if __name__ == "__main__":
    app()
