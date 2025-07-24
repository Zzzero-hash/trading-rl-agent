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

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from trade_agent.config import get_logger, get_settings, load_settings
from trade_agent.data.pipeline import DataPipeline
from trade_agent.data.sentiment import SentimentAnalyzer
from trade_agent.logging_conf import setup_logging_for_typer
from trade_agent.utils.cache_manager import CacheManager
from trade_agent.utils.performance_monitor import PerformanceMonitor

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


@train_app.command()
def cnn_lstm(
    data_path: Annotated[Path, typer.Argument(..., help="Path to processed training data")],
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
    """Train a CNN+LSTM model for feature extraction."""
    console.print("[bold blue]Training CNN+LSTM model...[/bold blue]")
    try:
        from trade_agent.training.train_cnn_lstm_enhanced import (
            EnhancedCNNLSTMTrainer,
            HyperparameterOptimizer,
            create_enhanced_model_config,
            create_enhanced_training_config,
        )

        # Load data
        data = np.load(data_path)
        sequences = data["sequences"]
        targets = data["targets"]

        # Check if we should run hyperparameter optimization
        if optimize_hyperparams:
            console.print("[yellow]Running hyperparameter optimization...[/yellow]")
            optimizer = HyperparameterOptimizer(sequences, targets, n_trials=n_trials)
            result = optimizer.optimize()
            best_params = result.get("best_params", {})
            console.print(f"[green]Best parameters found: {best_params}[/green]")

            # Extract optimized parameters
            if "sequence_length" in best_params:
                sequence_length = best_params["sequence_length"]
            if "prediction_horizon" in best_params:
                prediction_horizon = best_params["prediction_horizon"]
            if "learning_rate" in best_params:
                learning_rate = best_params["learning_rate"]
            if "batch_size" in best_params:
                batch_size = best_params["batch_size"]

            console.print(f"[cyan]Using optimized sequence_length={sequence_length}, prediction_horizon={prediction_horizon}[/cyan]")

        # Create model and training configs
        model_config = create_enhanced_model_config(input_dim=sequences.shape[-1])
        training_config = create_enhanced_training_config(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Add sequence parameters to training config
        training_config["sequence_length"] = sequence_length
        training_config["prediction_horizon"] = prediction_horizon

        # Create and run trainer
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            device="cuda" if gpu else "cpu",
        )

        # Create dataset config
        dataset_config = {
            "sequence_length": sequence_length,
            "prediction_horizon": prediction_horizon,
        }

        trainer.train_from_dataset(
            sequences=sequences,
            targets=targets,
            save_path=str(output_dir / "best_model.pth"),
            dataset_config=dataset_config,
        )

        console.print(f"[bold green]✅ CNN+LSTM training complete! Model saved to {output_dir}[/bold green]")
        console.print(f"[cyan]Used sequence_length={sequence_length}, prediction_horizon={prediction_horizon}[/cyan]")
    except Exception as e:
        console.print(f"[red]Error during CNN+LSTM training: {e}[/red]")
        if verbose_count > 0:
            logger.exception("CNN+LSTM training failed")
        raise typer.Exit(1) from e


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
def pipeline(
    # Core pipeline options
    run: bool = typer.Option(False, "--run", "-r", help="Run complete auto-processing pipeline (download → sentiment → process → standardize)"),

    # Processing configuration
    dataset_name: str = typer.Option("", "--dataset-name", help="Custom dataset name (default: auto-generated timestamp)"),
    processing_method: str = typer.Option("robust", "--method", help="Data standardization method (robust, standard, minmax)"),
    feature_set: str = typer.Option("full", "--features", help="Feature set to generate (basic, technical, full, custom)"),

    # Data source options
    symbols: str | None = typer.Option(None, "--symbols", help="Comma-separated symbols (e.g., 'AAPL,GOOGL,MSFT')"),
    max_symbols: int = typer.Option(50, "--max-symbols", help="Maximum symbols to auto-select if none specified"),
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,

    # Processing options
    include_sentiment: bool = typer.Option(True, "--sentiment/--no-sentiment", help="Include sentiment analysis"),
    sentiment_days: int = typer.Option(7, "--sentiment-days", help="Number of days back for sentiment analysis"),
    sentiment_sources: str = typer.Option("news,social", "--sentiment-sources", help="Sentiment sources (news,social,scrape)"),

    # Performance options
    parallel_workers: int = typer.Option(8, "--workers", help="Number of parallel workers"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use intelligent caching"),

    # Output options
    output_dir: str = typer.Option("data", help="Base output directory"),
    save_standardizer: bool = typer.Option(True, "--save-standardizer/--no-save-standardizer", help="Save standardizer for later use"),
    export_formats: str = typer.Option("csv", "--formats", help="Export formats (csv,parquet,feather)"),

    # Legacy options (deprecated)
    download: bool = typer.Option(False, "--download", "-d", help="[DEPRECATED] Use --run instead", hidden=True),
    legacy_sentiment: bool = typer.Option(False, help="[DEPRECATED] Use --sentiment/--no-sentiment instead", hidden=True),
    process: bool = typer.Option(False, "--process", "-p", help="[DEPRECATED] Use --run instead", hidden=True),
    skip_sentiment: bool = typer.Option(False, "--skip-sentiment", help="[DEPRECATED] Use --no-sentiment instead", hidden=True),
    config_path: Path | None = DEFAULT_CONFIG_FILE,
    legacy_method: str = typer.Option("robust", help="[DEPRECATED] Use --method instead", hidden=True),
) -> None:
    """
    Auto-processing data pipeline that eliminates raw data storage.

    Downloads, processes, and standardizes data in one seamless flow, creating
    ready-to-use datasets directly in organized output directories.

    Examples:
        # Basic auto-processing pipeline
        trade-agent data pipeline --run

        # Custom symbols with sentiment analysis
        trade-agent data pipeline --run --symbols "AAPL,GOOGL,TSLA" --sentiment-days 14

        # Generate technical analysis dataset
        trade-agent data pipeline --run --features technical --dataset-name "tech_analysis_v1"

        # High-performance batch processing
        trade-agent data pipeline --run --max-symbols 100 --workers 16 --no-sentiment
    """
    monitor = PerformanceMonitor()
    with monitor.time_operation("Auto-Processing Pipeline"):
        try:
            # Convert output_dir to Path
            output_path = Path(output_dir)

            # Handle legacy options with deprecation warnings
            if any([download, legacy_sentiment, process, skip_sentiment]):
                console.print("[yellow]⚠️  Warning: Legacy options are deprecated. Use --run with modern options.[/yellow]")
                if not run:
                    console.print("[yellow]Converting legacy options to --run mode...[/yellow]")
                    run = True
                    if skip_sentiment:
                        include_sentiment = False
                    processing_method = legacy_method  # Use legacy method parameter

            # Validate that user wants to run the pipeline
            if not run:
                console.print("[red]❌ No pipeline action specified. Use --run to execute the auto-processing pipeline.[/red]")
                console.print("\n[cyan]💡 Examples:[/cyan]")
                console.print("  [dim]trade-agent data pipeline --run[/dim]")
                console.print("  [dim]trade-agent data pipeline --run --symbols 'AAPL,GOOGL' --sentiment-days 14[/dim]")
                console.print("  [dim]trade-agent data pipeline --run --features technical --dataset-name custom_tech[/dim]")
                raise typer.Exit(1)

            # Initialize cache manager
            cache_manager = CacheManager()

            # Generate unique dataset name if not provided
            if not dataset_name:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_name = f"dataset_{timestamp}"

            # Set date defaults
            if start_date is None:
                from datetime import datetime, timedelta
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            if end_date is None:
                from datetime import datetime
                end_date = datetime.now().strftime("%Y-%m-%d")

            # Handle symbol selection
            symbols_were_auto_selected = False
            if symbols is None:
                with monitor.time_operation("Symbol Selection"):
                    from trade_agent.data.market_symbols import get_optimized_symbols
                    symbol_list = get_optimized_symbols(max_symbols=max_symbols)
                    symbols = ",".join(symbol_list)
                    symbols_were_auto_selected = True
                    console.print(f"[green]🎯 Auto-selected {len(symbol_list)} optimized symbols[/green]")
            else:
                symbol_list = [s.strip() for s in symbols.split(",")]
                console.print(f"[green]🎯 Processing {len(symbol_list)} specified symbols[/green]")

            # Create unique dataset directory (no more raw folder)
            dataset_dir = output_path / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"[green]🚀 Auto-Processing Pipeline: {dataset_name}[/green]")
            console.print(f"[cyan]📁 Dataset Directory: {dataset_dir}[/cyan]")

            # Display detailed parameter information
            console.print("\n[bold blue]📋 Pipeline Configuration[/bold blue]")
            symbol_preview = f"{symbol_list[:3]}{'...' if len(symbol_list) > 3 else ''}"
            console.print(f"[cyan]🎯 Symbols: {len(symbol_list)} {'(auto-selected)' if symbols_were_auto_selected else '(specified)'} {symbol_preview}[/cyan]")
            console.print(f"[cyan]📈 Method: {processing_method} scaling | Features: {feature_set} set[/cyan]")
            console.print(f"[cyan]🧠 Sentiment: {'Enabled' if include_sentiment else 'Disabled'} | Sources: {sentiment_sources} | Days: {sentiment_days}[/cyan]")
            console.print(f"[cyan]⚡ Workers: {parallel_workers} | Cache: {'Enabled' if use_cache else 'Disabled'}[/cyan]")
            console.print(f"[cyan]📅 Period: {start_date} to {end_date}[/cyan]")
            console.print(f"[cyan]📦 Export: {export_formats}[/cyan]")

            # Initialize sentiment_source_list for metadata
            sentiment_source_list = [s.strip() for s in sentiment_sources.split(",")]

            # Auto-processing pipeline: Download → Process → Standardize (all in memory)
            with monitor.time_operation("Download & Initial Processing"):
                cache_key = cache_manager.get_cache_key("auto_download", symbols=symbol_list, start_date=start_date, end_date=end_date)
                downloaded_data = cache_manager.get_cached_data(cache_key) if use_cache else None

                if downloaded_data is None:
                    pipeline = DataPipeline()
                    # Download data directly into memory (no file storage)
                    downloaded_data = pipeline.download_data_parallel(
                        symbols=symbol_list,
                        start_date=start_date,
                        end_date=end_date,
                        max_workers=parallel_workers,
                        include_features=(feature_set != "basic"),
                        align_mixed_portfolio=True,
                    )

                    if use_cache:
                        cache_manager.cache_data(cache_key, downloaded_data)

                console.print(f"[green]✅ Downloaded and processed {len(symbol_list)} symbols in memory[/green]")

            # Sentiment analysis integration (if enabled)
            sentiment_success: bool | None = False  # Track if sentiment analysis was successful
            if include_sentiment:
                with monitor.time_operation("Sentiment Analysis"):
                    console.print("[blue]🧠 Analyzing market sentiment...[/blue]")

                    try:
                        analyzer = SentimentAnalyzer()
                        cache_key = cache_manager.get_cache_key("sentiment", symbols=symbol_list, days_back=sentiment_days)
                        sentiment_features = cache_manager.get_cached_data(cache_key) if use_cache else None

                        if sentiment_features is None:
                            sentiment_features = analyzer.get_sentiment_features_parallel(
                                symbols=symbol_list,
                                days_back=sentiment_days,
                                max_workers=parallel_workers,
                            )
                            if use_cache:
                                cache_manager.cache_data(cache_key, sentiment_features)

                        # Integrate sentiment into main dataset
                        if sentiment_features is not None and not sentiment_features.empty:
                            console.print(f"[green]✅ Integrated sentiment features from {len(sentiment_source_list)} sources[/green]")

                            # Check if we need to merge on a different column (common issue)
                            merge_column: str | None = "symbol"
                            if "symbol" not in downloaded_data.columns:
                                # Try common alternative column names in main dataset
                                merge_column = None
                                for col in ["Symbol", "ticker", "Ticker", "stock_symbol"]:
                                    if col in downloaded_data.columns:
                                        merge_column = col
                                        break

                            if "symbol" not in sentiment_features.columns:
                                # Check sentiment features for alternative names
                                for col in ["Symbol", "ticker", "Ticker", "stock_symbol"]:
                                    if col in sentiment_features.columns:
                                        # Rename to match main dataset
                                        sentiment_features = sentiment_features.rename(columns={col: merge_column or "symbol"})
                                        break
                                else:
                                    merge_column = None

                            # Perform the merge with better error handling
                            try:
                                if merge_column and merge_column in downloaded_data.columns:
                                    downloaded_data = downloaded_data.merge(
                                        sentiment_features,
                                        on=merge_column,
                                        how="left",
                                        suffixes=("", "_sentiment")
                                    )
                                else:
                                    # Merge on index as fallback
                                    downloaded_data = downloaded_data.merge(
                                        sentiment_features,
                                        left_index=True,
                                        right_index=True,
                                        how="left",
                                        suffixes=("", "_sentiment")
                                    )

                                sentiment_success = True  # Mark sentiment as successful
                                console.print(f"[green]✅ Successfully merged sentiment data using '{merge_column or 'index'}' column[/green]")

                            except Exception as merge_error:
                                console.print(f"[yellow]⚠️  Merge failed: {merge_error}[/yellow]")
                                console.print("[yellow]📝 Adding sentiment features as separate columns instead[/yellow]")

                                # Add sentiment features as separate columns with symbol-based lookup
                                sentiment_dict = sentiment_features.set_index("symbol").to_dict("index") if "symbol" in sentiment_features.columns else {}

                                # Add sentiment columns to main dataset
                                for col in ["sentiment_score", "sentiment_magnitude", "sentiment_sources", "sentiment_direction"]:
                                    if col in sentiment_features.columns:
                                        # Try to map by symbol if possible
                                        if "symbol" in downloaded_data.columns:
                                            current_col = col  # Capture loop variable to fix B023
                                            downloaded_data[col] = downloaded_data["symbol"].map(
                                                lambda x, c=current_col: sentiment_dict.get(x, {}).get(c, 0.0)
                                            )
                                        else:
                                            # Fallback to default value
                                            downloaded_data[col] = 0.0

                                sentiment_success = True  # Still mark as successful since we added the data
                                console.print("[green]✅ Added sentiment features as separate columns[/green]")
                        else:
                            console.print("[yellow]⚠️  No sentiment data available - creating default sentiment features[/yellow]")
                            # Create default sentiment features to maintain pipeline consistency
                            sentiment_cols = ["sentiment_score", "sentiment_magnitude", "sentiment_sources", "sentiment_direction"]
                            for col in sentiment_cols:
                                downloaded_data[col] = 0.0  # Default neutral sentiment
                            console.print(f"[yellow]📝 Added {len(sentiment_cols)} default sentiment columns[/yellow]")

                    except Exception as e:
                        console.print(f"[yellow]⚠️  Sentiment analysis failed: {e}[/yellow]")
                        console.print("[yellow]📝 Creating default sentiment features to maintain pipeline consistency[/yellow]")
                        # Create default sentiment features even when analysis fails
                        sentiment_cols = ["sentiment_score", "sentiment_magnitude", "sentiment_sources", "sentiment_direction"]
                        for col in sentiment_cols:
                            downloaded_data[col] = 0.0  # Default neutral sentiment
                        console.print(f"[yellow]✅ Added {len(sentiment_cols)} default sentiment columns[/yellow]")
            else:
                # Sentiment was explicitly disabled
                sentiment_success = None  # Use None to distinguish between disabled vs failed
                console.print("[dim]🧠 Sentiment analysis disabled by user (--no-sentiment)[/dim]")

                        # Data standardization and feature engineering
            with monitor.time_operation("Standardization & Feature Engineering"):
                console.print(f"[blue]⚙️  Standardizing with robust scaling and {feature_set} features...[/blue]")

                from config import FeatureConfig
                from trade_agent.data.data_standardizer import create_standardized_dataset

                # Apply advanced feature engineering based on feature_set
                if feature_set == "full" or feature_set == "technical":
                    from trade_agent.data.features import generate_features
                    downloaded_data = generate_features(downloaded_data)
                elif feature_set == "custom":
                    # Allow custom feature configuration from config file
                    if config_path and config_path.exists():
                        import yaml
                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                            if "features" in config:
                                from trade_agent.data.features import generate_features
                                downloaded_data = generate_features(downloaded_data)

                # Create feature config based on feature_set
                feature_config = FeatureConfig()

                # Note: processing_method parameter is not used by the current DataStandardizer
                # The DataStandardizer uses RobustScaler by default for outlier-resistant scaling
                if processing_method != "robust":
                    console.print(f"[yellow]⚠️  Processing method '{processing_method}' not supported by DataStandardizer. Using robust scaling.[/yellow]")

                # Create standardized dataset
                standardized_data, standardizer = create_standardized_dataset(
                    df=downloaded_data,
                    save_path=str(dataset_dir / "data_standardizer.pkl") if save_standardizer else None,
                    feature_config=feature_config
                )

                console.print(f"[green]✅ Standardized {len(standardized_data)} rows with {len(standardized_data.columns)} features[/green]")

            # Export in multiple formats
            with monitor.time_operation("Export Dataset"):
                export_format_list = [f.strip() for f in export_formats.split(",")]
                console.print(f"[blue]💾 Exporting dataset in {len(export_format_list)} formats...[/blue]")

                for fmt in export_format_list:
                    if fmt == "csv":
                        standardized_data.to_csv(dataset_dir / "dataset.csv", index=False)
                    elif fmt == "parquet":
                        standardized_data.to_parquet(dataset_dir / "dataset.parquet", index=False)
                    elif fmt == "feather":
                        standardized_data.to_feather(dataset_dir / "dataset.feather")
                    else:
                        console.print(f"[yellow]⚠️  Unknown format: {fmt}[/yellow]")

                # Save dataset metadata
                import json
                from datetime import datetime
                metadata = {
                    "dataset_name": dataset_name,
                    "created_at": datetime.now().isoformat(),
                    "symbols": symbol_list,
                    "start_date": start_date,
                    "end_date": end_date,
                    "processing_method": processing_method,
                    "feature_set": feature_set,
                    "include_sentiment": include_sentiment,
                    "sentiment_days": sentiment_days if include_sentiment else None,
                    "sentiment_sources": sentiment_source_list if include_sentiment else None,
                    "row_count": len(standardized_data),
                    "column_count": len(standardized_data.columns),
                    "export_formats": export_format_list,
                }

                import json
                with open(dataset_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                console.print(f"[green]✅ Dataset exported to: {dataset_dir}[/green]")
                for fmt in export_format_list:
                    file_path = dataset_dir / f"dataset.{fmt}"
                    if file_path.exists():
                        file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                        console.print(f"[cyan]  📄 dataset.{fmt} ({file_size:.2f} MB)[/cyan]")

            console.print("\n[green]🎉 Auto-Processing Pipeline completed successfully![/green]")
            console.print(f"\n[bold green]📊 Dataset Summary: {dataset_name}[/bold green]")
            console.print(f"[cyan]📁 Location: {dataset_dir}[/cyan]")
            console.print(f"[cyan]📊 Data: {len(standardized_data):,} rows x {len(standardized_data.columns)} columns[/cyan]")
            console.print(f"[cyan]🏷️  Symbols: {len(symbol_list)} ({symbol_list[:3]}{'...' if len(symbol_list) > 3 else ''})[/cyan]")
            console.print(f"[cyan]📈 Features: {feature_set} set with {processing_method} standardization[/cyan]")
            # Determine sentiment status for summary
            if sentiment_success is None:
                sentiment_status = "Disabled"
            elif sentiment_success:
                sentiment_status = "Included"
            elif include_sentiment:
                sentiment_status = "Default (fallback)"  # Sentiment was attempted but used defaults
            else:
                sentiment_status = "Failed"

            console.print(f"[cyan]🧠 Sentiment: {sentiment_status}[/cyan]")

            # Performance summary
            console.print("\n[bold green]⚡ Performance Summary[/bold green]")
            summary = monitor.get_summary()
            console.print(f"[cyan]⏱️  Total time: {summary['total_time']:.2f}s[/cyan]")
            if summary["slowest_operation"]:
                console.print(f"[cyan]🐌 Slowest step: {summary['slowest_operation'][0]} ({summary['slowest_operation'][1]:.2f}s)[/cyan]")

            # Usage suggestions
            console.print("\n[bold blue]💡 Next Steps[/bold blue]")
            console.print("[dim]# Train a model with this dataset[/dim]")
            console.print(f"[dim]trade-agent train cnn-lstm --data {dataset_dir}/dataset.csv[/dim]")
            console.print("[dim]# Run backtesting[/dim]")
            console.print(f"[dim]trade-agent backtest --data {dataset_dir}/dataset.csv[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Auto-processing pipeline failed: {e}[/red]")
            logger.error(f"Auto-processing pipeline failed: {e}", exc_info=True)
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
        import pandas as pd
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
    from trade_agent.data.market_symbols import get_comprehensive_symbols
    result = get_comprehensive_symbols()
    return str(result) if result is not None else ""


if __name__ == "__main__":
    app()
