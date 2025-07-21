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
                        "symbols": ["AAPL", "GOOGL", "MSFT"],
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
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR_NONE,
    source: str | None = DEFAULT_SOURCE,
    timeframe: str | None = DEFAULT_TIMEFRAME,
    parallel: bool = DEFAULT_PARALLEL,
    _force: bool = DEFAULT_FORCE,
) -> None:
    """
    Download all available market data from yfinance.

    Downloads comprehensive market data including:
    - Major US stocks (S&P 500 top components)
    - Popular ETFs
    - Major forex pairs
    - Cryptocurrencies
    - Market indices
    - Commodities

    Defaults to last 10 years of daily data unless specified otherwise.
    Data is automatically organized in the standard data directory structure.
    """
    try:
        # Load configuration
        settings = get_config_manager()

        # Use config values with CLI overrides
        source = source or settings.data.primary_source
        timeframe = timeframe or settings.data.timeframe

        # Set default date range to last 10 years if not specified
        from datetime import datetime, timedelta

        if not start_date:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=3650)  # 10 years
            start_date = start_dt.strftime("%Y-%m-%d")

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Create organized directory structure in the standard data location
        base_data_dir = Path(settings.data.data_path)
        base_output_dir = output_dir or base_data_dir / "raw" / "comprehensive"

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

        # Comprehensive symbol list organized by asset type
        symbols_by_type = {
            "stocks": [
                # Major US Stocks (S&P 500 top components) - Current as of 2024
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
                "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE", "CRM",
                "NFLX", "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "DHR", "ACN", "LLY",
                "NKE", "TXN", "QCOM", "HON", "ORCL", "LOW", "UPS", "INTU", "SPGI", "GILD",
                "AMD", "ISRG", "TGT", "ADI", "PLD", "REGN", "MDLZ", "VRTX", "PANW", "KLAC",
                # Tech Giants (current symbols)
                "GOOG", "INTC", "CSCO", "IBM", "MU", "LRCX", "AMAT", "ASML",
                # Financial Sector
                "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC", "COF",
                "TFC", "KEY", "HBAN", "RF", "ZION", "CMA", "FITB", "MTB",
                # Healthcare
                "PFE", "ABBV", "BMY", "ABT", "MRK", "AMGN", "BIIB", "DXCM", "ALGN", "IDXX", "ILMN",
                # Consumer
                "WMT", "SBUX", "MCD", "YUM", "CMCSA", "UA", "LULU", "ROST", "TJX", "MAR", "HLT", "BKNG",
                # Energy (current symbols)
                "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "PSX", "VLO", "MPC",
                "OXY", "DVN", "FANG", "HES", "APA",
                # Additional Popular Stocks (current)
                "UBER", "LYFT", "SNAP", "PINS", "ZM", "ROKU", "SPOT", "BYND", "PLTR", "SNOW",
                "CRWD", "ZS", "OKTA", "TEAM", "DOCU", "TDOC", "RBLX", "HOOD", "COIN", "RIVN",
                "LCID", "NIO", "XPEV", "LI", "BIDU", "JD", "BABA", "TCEHY", "PDD", "NTES",
                "BILI", "XNET", "ZTO", "TME", "VIPS"
            ],
            "etfs": [
                # ETFs (Major categories) - Current symbols
                "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT",
                "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY",
                "XLB", "XLU", "VNQ", "IEMG", "EFA", "EEM", "ACWI", "VT", "BNDX", "EMB"
            ],
            "indices": [
                # Market Indices - Current symbols
                "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
                "^BSESN", "^AXJO", "^TNX", "^TYX", "^IRX"
            ],
            "forex": [
                # Forex (Major pairs) - Current symbols
                "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
                "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X",
                "EURCHF=X", "GBPCHF=X", "AUDCHF=X", "CADCHF=X", "NZDCHF=X"
            ],
            "crypto": [
                # Cryptocurrencies - Current symbols
                "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD",
                "XRP-USD", "BNB-USD", "SOL-USD", "AVAX-USD", "MATIC-USD", "UNI-USD", "ATOM-USD",
                "NEAR-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD"
            ],
            "commodities": [
                # Commodities - Current symbols
                "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F",
                "LBS=F", "HE=F", "LE=F", "GF=F"
            ]
        }

        # Deduplicate symbols within each category first
        for category_type in symbols_by_type:
            seen_in_category = set()
            deduplicated = []
            for x in symbols_by_type[category_type]:
                if x not in seen_in_category:
                    seen_in_category.add(x)
                    deduplicated.append(x)
            symbols_by_type[category_type] = deduplicated

        # Flatten all symbols for downloading
        all_symbols = []
        for category_type, symbols in symbols_by_type.items():
            all_symbols.extend(symbols)

        # Remove duplicates across all categories while preserving order
        seen = set()
        unique_symbols = []
        for x in all_symbols:
            if x not in seen:
                seen.add(x)
                unique_symbols.append(x)

        # Validate symbols before downloading
        console.print("[yellow]Validating symbols...[/yellow]")
        valid_symbols = []
        invalid_symbols = []

        try:
            import yfinance as yf
            for symbol in unique_symbols:
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
            valid_symbols = unique_symbols

        if invalid_symbols:
            console.print(f"[yellow]Removed {len(invalid_symbols)} invalid symbols: {', '.join(invalid_symbols[:10])}{'...' if len(invalid_symbols) > 10 else ''}[/yellow]")

        console.print(f"[green]Proceeding with {len(valid_symbols)} valid symbols[/green]")

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

        # Define symbol categories for organization
        symbol_categories = {
            "stocks": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
                "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE", "CRM",
                "NFLX", "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "DHR", "ACN", "LLY",
                "NKE", "TXN", "QCOM", "HON", "ORCL", "LOW", "UPS", "INTU", "SPGI", "GILD",
                "AMD", "ISRG", "TGT", "ADI", "PLD", "REGN", "MDLZ", "VRTX", "PANW", "KLAC",
                "GOOG", "INTC", "CSCO", "IBM", "MU", "LRCX", "AMAT", "ASML",
                "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC", "COF",
                "PFE", "ABBV", "BMY", "MRK", "AMGN", "BIIB", "DXCM", "ALGN", "IDXX", "ILMN",
                "WMT", "SBUX", "MCD", "YUM", "CMCSA", "UA", "LULU", "ROST", "TJX", "MAR", "HLT", "BKNG",
                "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "PSX", "VLO", "MPC",
                "UBER", "LYFT", "SNAP", "PINS", "ZM", "ROKU", "SPOT", "BYND", "PLTR", "SNOW",
                "CRWD", "ZS", "OKTA", "TEAM", "DOCU", "TDOC", "RBLX", "HOOD", "COIN", "RIVN"
            ],
            "etfs": [
                "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT",
                "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY",
                "XLB", "XLU", "VNQ", "IEMG", "EFA", "EEM", "ACWI", "VT", "BNDX", "EMB"
            ],
            "indices": [
                "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
                "^BSESN", "^AXJO", "^TNX", "^TYX", "^IRX"
            ],
            "forex": [
                "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
                "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X",
                "EURCHF=X", "GBPCHF=X", "AUDCHF=X", "CADCHF=X", "NZDCHF=X"
            ],
            "crypto": [
                "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD",
                "XRP-USD", "BNB-USD", "SOL-USD", "AVAX-USD", "MATIC-USD", "UNI-USD", "ATOM-USD",
                "NEAR-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD"
            ],
            "commodities": [
                "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F",
                "LBS=F", "HE=F", "LE=F", "GF=F"
            ]
        }

        # Validate symbols before downloading
        console.print("[yellow]Validating symbols...[/yellow]")
        valid_symbols = []
        invalid_symbols = []

        try:
            import yfinance as yf
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
    _output_dir: Path | None = DEFAULT_OUTPUT_DIR_NONE,
    _source: str | None = DEFAULT_SOURCE,
) -> None:
    """
    Download market data for specified symbols.

    Uses the DataPipeline.download_data() function from src/trade_agent/data/pipeline.py
    to fetch historical market data from various sources.
    """
    console.print(f"[blue]PLACEHOLDER: Would download data for {symbols} from {start_date} to {end_date}[/blue]")
    console.print("[blue]Target module: src/trade_agent/data/pipeline.py - DataPipeline.download_data()[/blue]")


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

    except Exception as e:
        console.print(f"[red]Error during data preparation: {e}[/red]")
        logger.error(f"Data preparation failed: {e}", exc_info=True)
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
        console.print("[blue]Target module: src/trade_agent/data/pipeline.py[/blue]")

    except Exception as e:
        console.print(f"[red]Error during pipeline execution: {e}[/red]")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise typer.Exit(1) from e


@data_app.command(hidden=True)
def process(
    config_file: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force_rebuild: bool = DEFAULT_FORCE_REBUILD,
    parallel: bool = DEFAULT_PARALLEL,
) -> None:
    """
    [DEPRECATED] Use 'data prepare' instead.

    This command is deprecated and will be removed in a future version.
    Use 'data prepare' which combines processing and standardization.
    """
    console.print("[yellow]⚠️  DEPRECATED: 'data process' is deprecated.[/yellow]")
    console.print("[yellow]Use 'data prepare' instead, which combines processing and standardization.[/yellow]")
    console.print("[yellow]Example: trade-agent data prepare --input-path data/raw --output-dir outputs/datasets[/yellow]")

    # Call the new prepare command with the same parameters
    prepare(
        input_path=None,
        output_dir=output_dir,
        config_file=config_file,
        force_rebuild=force_rebuild,
        parallel=parallel,
        method=DEFAULT_STANDARDIZATION_METHOD,
        save_standardizer=True
    )





# ============================================================================
# TRAIN SUB-APP COMMANDS
# ============================================================================


@train_app.command(name="cnn_lstm")
def cnn_lstm(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    _epochs: int = DEFAULT_EPOCHS,
    _batch_size: int = DEFAULT_BATCH_SIZE,
    _learning_rate: float = DEFAULT_LEARNING_RATE,
    _output_dir: Path = DEFAULT_CNN_LSTM_OUTPUT,
    _gpu: bool = DEFAULT_GPU,
    _mixed_precision: bool = DEFAULT_MIXED_PRECISION,
) -> None:
    """
    Train CNN+LSTM models for pattern recognition.

    Uses the OptimizedTrainingManager.train() function from
    src/trade_agent/training/optimized_trainer.py to train
    CNN+LSTM models with advanced optimizations.
    """
    console.print(f"[blue]PLACEHOLDER: Would train CNN+LSTM model for {_epochs} epochs[/blue]")
    console.print(
        "[blue]Target module: src/trade_agent/training/optimized_trainer.py - "
        "OptimizedTrainingManager.train()[/blue]",
    )


@train_app.command(name="rl")
def rl(
    _agent_type: str | None = DEFAULT_AGENT_TYPE,
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    _timesteps: int = DEFAULT_TIMESTEPS,
    _output_dir: Path = DEFAULT_RL_OUTPUT,
    _ray_address: str | None = DEFAULT_RAY_ADDRESS,
    _num_workers: int = DEFAULT_NUM_WORKERS,
) -> None:
    """
    Train reinforcement learning agents.

    Uses the Trainer.train() function from src/trade_agent/agents/trainer.py
    to train RL agents (PPO, SAC, TD3) using Ray RLlib.
    """
    console.print(f"[blue]PLACEHOLDER: Would train {_agent_type} agent for {_timesteps} timesteps[/blue]")
    console.print("[blue]Target module: src/trade_agent/agents/trainer.py - Trainer.train()[/blue]")


@train_app.command(name="hybrid")
def hybrid(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    _cnn_lstm_path: Path | None = DEFAULT_CNN_LSTM_PATH,
    _rl_path: Path | None = DEFAULT_RL_PATH,
    _output_dir: Path = DEFAULT_HYBRID_OUTPUT,
) -> None:
    """
    Train hybrid models combining CNN+LSTM with RL agents.

    Uses the HybridAgent class from src/trade_agent/agents/hybrid.py
    to create and train hybrid models that combine supervised and RL components.
    """
    console.print("[blue]PLACEHOLDER: Would train hybrid model combining CNN+LSTM and RL[/blue]")
    console.print("[blue]Target module: src/trade_agent/agents/hybrid.py - HybridAgent[/blue]")


@train_app.command(name="hyperopt")
def hyperopt(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    _n_trials: int = DEFAULT_N_TRIALS,
    _output_dir: Path = DEFAULT_OPTIMIZATION_OUTPUT,
) -> None:
    """
    Perform hyperparameter optimization.

    Uses Optuna-based optimization from train.py to find optimal
    hyperparameters for models and training configurations.
    """
    console.print(f"[blue]PLACEHOLDER: Would run hyperparameter optimization with {_n_trials} trials[/blue]")
    console.print("[blue]Target module: train.py - hyperparameter optimization functions[/blue]")


# ============================================================================
# BACKTEST SUB-APP COMMANDS
# ============================================================================


@backtest_app.command()
def strategy(
    _data_path: Path | None = DEFAULT_DATA_PATH,
    _model_path: Path | None = DEFAULT_MODEL_PATH,
    _policy: str | None = DEFAULT_POLICY,
    _initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    _commission: float = DEFAULT_COMMISSION,
    _slippage: float = DEFAULT_SLIPPAGE,
    _output_dir: Path = DEFAULT_BACKTEST_OUTPUT,
) -> None:
    """
    Run backtesting on historical data.

    Uses the TradingSession class from src/trade_agent/core/live_trading.py
    adapted for backtesting to evaluate trading strategies on historical data.
    """
    console.print(f"[blue]PLACEHOLDER: Would backtest strategy on {_data_path} with ${_initial_capital} capital[/blue]")
    console.print("[blue]Target module: src/trade_agent/core/live_trading.py - TradingSession (adapted)[/blue]")


@backtest_app.command()
def evaluate(
    model_path: Path | None = DEFAULT_MODEL_PATH,
    _data_path: Path | None = DEFAULT_DATA_PATH,
    _output_dir: Path = DEFAULT_EVALUATION_OUTPUT,
    _initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> None:
    """
    Evaluate trained models on test data.

    Uses the OptimizedTrainingManager.evaluate() function from
    src/trade_agent/training/optimized_trainer.py to evaluate
    model performance on test datasets.
    """
    console.print(f"[blue]PLACEHOLDER: Would evaluate model {model_path} on test data[/blue]")
    console.print(
        "[blue]Target module: src/trade_agent/training/optimized_trainer.py - "
        "OptimizedTrainingManager.evaluate()[/blue]",
    )


@backtest_app.command()
def walk_forward(
    _data_path: Path | None = DEFAULT_DATA_PATH,
    _model_type: str = "cnn_lstm",
    _train_window_size: int = 252,
    _validation_window_size: int = 63,
    _test_window_size: int = 63,
    _step_size: int = 21,
    _output_dir: Path = DEFAULT_EVALUATION_OUTPUT,
    _confidence_level: float = 0.95,
    _generate_plots: bool = True,
    _save_results: bool = True,
) -> None:
    """
    Perform walk-forward analysis for robust model evaluation.

    Uses the WalkForwardAnalyzer class from src/trade_agent/eval/walk_forward_analyzer.py
    to evaluate model performance across multiple time windows.
    """
    console.print(f"[blue]PLACEHOLDER: Would perform walk-forward analysis on {_data_path}[/blue]")
    console.print(
        "[blue]Target module: src/trade_agent/eval/walk_forward_analyzer.py - WalkForwardAnalyzer[/blue]",
    )

    # TODO: Implement actual walk-forward analysis
    # from trade_agent.eval import WalkForwardAnalyzer, WalkForwardConfig
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
    _data_path: Path | None = DEFAULT_DATA_PATH,
    _output_dir: Path = DEFAULT_COMPARISON_OUTPUT,
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
    _output_dir: Path = DEFAULT_REPORTS_OUTPUT,
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
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    _symbols: str | None = DEFAULT_SYMBOLS_NONE,
    _model_path: Path | None = DEFAULT_MODEL_PATH,
    _paper_trading: bool = DEFAULT_PAPER_TRADING,
    _initial_capital: float = DEFAULT_TRADING_CAPITAL,
) -> None:
    """
    Start live trading session.

    Uses the LiveTradingEngine.create_session() and TradingSession.start() functions
    from src/trade_agent/core/live_trading.py to initiate live trading.
    """
    console.print(f"[blue]PLACEHOLDER: Would start live trading with ${_initial_capital} capital[/blue]")
    console.print(
        "[blue]Target module: src/trade_agent/core/live_trading.py - LiveTradingEngine, TradingSession[/blue]",
    )


@trade_app.command()
def stop(
    _session_id: str | None = DEFAULT_SESSION_ID_NONE,
    _all_sessions: bool = DEFAULT_ALL_SESSIONS_FALSE,
) -> None:
    """
    Stop live trading session(s).

    Uses the LiveTradingEngine.stop_all_sessions() function from
    src/trade_agent/core/live_trading.py to stop trading sessions.
    """
    console.print("[blue]PLACEHOLDER: Would stop trading session(s)[/blue]")
    console.print(
        "[blue]Target module: src/trade_agent/core/live_trading.py - LiveTradingEngine.stop_all_sessions()[/blue]",
    )


@trade_app.command()
def status(
    _session_id: str | None = DEFAULT_SESSION_ID_NONE,
    _detailed: bool = DEFAULT_DETAILED_FALSE,
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
    _session_id: str | None = DEFAULT_SESSION_ID_NONE,
    _metrics: str = DEFAULT_METRICS_ALL,
    _interval: int = DEFAULT_INTERVAL_60,
) -> None:
    """
    Monitor live trading session in real-time.

    Provides real-time monitoring of trading sessions with live updates
    on portfolio performance, risk metrics, and trading activity.
    """
    console.print(f"[blue]PLACEHOLDER: Would monitor trading session with {_interval}s interval[/blue]")
    console.print("[blue]Target module: Real-time monitoring functions[/blue]")


@trade_app.command()
def paper(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
    symbols: str = DEFAULT_PAPER_SYMBOLS,
    _duration: str = DEFAULT_PAPER_DURATION,
) -> None:
    """
    Start a paper trading session with simulated trades.
    """
    console.print(f"[blue]PLACEHOLDER: Would start paper trading for {symbols} for {_duration}[/blue]")
    console.print("[blue]Target module: Paper trading utilities[/blue]")


# ============================================================================
# SCENARIO SUB-APP COMMANDS
# ============================================================================


@scenario_app.command()
def scenario_evaluate(
    _config_file: Path | None = DEFAULT_CONFIG_FILE,
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
        from trade_agent.eval import AgentScenarioEvaluator

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


if __name__ == "__main__":
    app()
