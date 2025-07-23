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
from datetime import UTC
from pathlib import Path
from typing import Annotated, Any, TypeVar

import numpy as np
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


@train_app.command()
def cnn_lstm(
    data_path: Annotated[Path, typer.Argument(..., help="Path to processed training data")],
    output_dir: Path = DEFAULT_CNN_LSTM_OUTPUT,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    gpu: bool = DEFAULT_GPU,
) -> None:
    """Train a CNN+LSTM model for feature extraction."""
    console.print("[bold blue]Training CNN+LSTM model...[/bold blue]")
    try:
        from trade_agent.training.train_cnn_lstm_enhanced import (
            EnhancedCNNLSTMTrainer,
            create_enhanced_model_config,
            create_enhanced_training_config,
        )

        # Load data
        data = np.load(data_path)
        sequences = data["sequences"]
        targets = data["targets"]

        # Create model and training configs
        model_config = create_enhanced_model_config(input_dim=sequences.shape[-1])
        training_config = create_enhanced_training_config(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Create and run trainer
        trainer = EnhancedCNNLSTMTrainer(
            model_config=model_config,
            training_config=training_config,
            device="cuda" if gpu else "cpu",
        )
        trainer.train_from_dataset(
            sequences=sequences,
            targets=targets,
            save_path=str(output_dir / "best_model.pth"),
        )

        console.print(f"[bold green]✅ CNN+LSTM training complete! Model saved to {output_dir}[/bold green]")
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
    # Pipeline step options
    download: bool = typer.Option(False, "--download", "-d", help="Download market data"),
    sentiment: bool = typer.Option(False, "--sentiment", "-s", help="Collect sentiment analysis data"),
    process: bool = typer.Option(False, "--process", "-p", help="Process and standardize data"),
    run: bool = typer.Option(False, "--run", "-r", help="Run complete pipeline end-to-end (download → sentiment → process)"),

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

    # Sentiment-specific parameters
    sentiment_days: int = typer.Option(7, "--sentiment-days", help="Number of days back for sentiment analysis"),
    sentiment_sources: str = typer.Option("news,social", "--sentiment-sources", help="Comma-separated sentiment sources (news,social,scrape)"),
    include_sentiment_features: bool = typer.Option(True, "--include-sentiment-features", help="Include sentiment features in processed data"),
    include_historical_sentiment: bool = typer.Option(True, "--include-historical-sentiment", help="Include historical sentiment in market data"),
    sentiment_lookback_days: int = typer.Option(30, "--sentiment-lookback-days", help="Days back for historical sentiment collection"),
) -> None:
    """
    Unified data pipeline operations with sentiment analysis support.

    This is the primary command for all data operations. Use options to specify which pipeline steps to run:
    - --download: Download market data from specified sources
    - --sentiment: Collect sentiment analysis data for symbols
    - --process: Process and standardize downloaded data
    - --run: Execute complete pipeline (download → sentiment → process)

    The pipeline automatically handles:
    - Symbol validation and organization by asset type
    - Parallel data fetching with intelligent caching
    - Sentiment analysis from multiple sources (news, social media, web scraping)
    - Feature engineering and data standardization
    - Comprehensive reporting and metadata

    Examples:
        # Complete pipeline with sentiment analysis
        python main.py data pipeline --run --symbols "AAPL,GOOGL" --sentiment-days 14

        # Download and sentiment only
        python main.py data pipeline --download --sentiment --symbols "AAPL,GOOGL"

        # Process existing data with sentiment features
        python main.py data pipeline --process --include-sentiment-features

        # Custom sentiment sources
        python main.py data pipeline --run --sentiment-sources "news,scrape" --sentiment-days 30
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
            sentiment = True
            process = True

        if not any([download, sentiment, process]):
            # No steps specified, show help
            console.print("[yellow]No pipeline steps specified. Use --download, --sentiment, --process, or --run[/yellow]")
            console.print("[yellow]Example: python main.py data pipeline --run --symbols 'AAPL'[/yellow]")
            raise typer.Exit(1)

        console.print("[green]🚀 Unified Data Pipeline Operations[/green]")
        steps = []
        if download:
            steps.append("download")
        if sentiment:
            steps.append("sentiment")
        if process:
            steps.append("process")
        console.print(f"[cyan]Steps: {' → '.join(steps)}[/cyan]")
        console.print(f"[cyan]Output: {output_dir}[/cyan]")
        console.print(f"[cyan]Parallel: {parallel}[/cyan]")

        # Create output directory structure
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "raw"
        sentiment_dir = output_dir / "sentiment"
        processed_dir = output_dir / "processed"

        for dir_path in [raw_dir, sentiment_dir, processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Download (if requested)
        if download:
            console.print("\n[blue]Step 1: Downloading market data...[/blue]")

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
                symbol_list = [s.strip() for s in symbols.split(",")]
                console.print(f"[green]Pipeline Download: {symbols}[/green]")

            console.print(f"[cyan]Source: {source}[/cyan]")
            console.print(f"[cyan]Timeframe: {timeframe}[/cyan]")
            console.print(f"[cyan]Date range: {start_date} to {end_date}[/cyan]")
            console.print(f"[cyan]Output: {raw_dir}[/cyan]")
            console.print(f"[cyan]Force: {force}[/cyan]")
            console.print(f"[cyan]Include historical sentiment: {include_historical_sentiment}[/cyan]")
            if include_historical_sentiment:
                console.print(f"[cyan]Sentiment lookback days: {sentiment_lookback_days}[/cyan]")

            # Import and run download functionality
            from .data.pipeline import DataPipeline

            pipeline = DataPipeline()

            # Download data with historical sentiment enhancement
            downloaded_files = pipeline.download_data(
                symbols=symbol_list,
                start_date=start_date,
                end_date=end_date,
                output_dir=raw_dir,
                include_sentiment=include_historical_sentiment,
                sentiment_lookback_days=sentiment_lookback_days,
            )

            console.print(f"[green]✅ Downloaded {len(downloaded_files)} files with historical sentiment[/green]")
            for file_path in downloaded_files:
                console.print(f"[cyan]  - {file_path}[/cyan]")

        # Step 2: Sentiment Analysis (if requested)
        if sentiment:
            console.print("\n[blue]Step 2: Collecting sentiment analysis data...[/blue]")

            # Determine symbols for sentiment analysis
            if symbols is None:
                # Use comprehensive symbols if no specific symbols provided
                from .data.market_symbols import get_all_symbols
                symbol_list = get_all_symbols()
            else:
                symbol_list = [s.strip() for s in symbols.split(",")]

            # Parse sentiment sources
            sentiment_source_list = [s.strip() for s in sentiment_sources.split(",")]

            console.print(f"[cyan]Symbols: {', '.join(symbol_list[:10])}{'...' if len(symbol_list) > 10 else ''}[/cyan]")
            console.print(f"[cyan]Days back: {sentiment_days}[/cyan]")
            console.print(f"[cyan]Sources: {', '.join(sentiment_source_list)}[/cyan]")
            console.print(f"[cyan]Output: {sentiment_dir}[/cyan]")

            # Import sentiment analyzer with error handling
            try:
                from .data.sentiment import SentimentAnalyzer, SentimentConfig

                # Configure sentiment analysis
                sentiment_config = SentimentConfig(
                    enable_news="news" in sentiment_source_list,
                    enable_social="social" in sentiment_source_list,
                )

                analyzer = SentimentAnalyzer(sentiment_config)

                # Collect sentiment data with robust error handling
                console.print("[yellow]Collecting sentiment data (this may take a while)...[/yellow]")

                sentiment_results = {}
                successful_sentiment = 0
                failed_sentiment = 0

                for symbol in symbol_list:
                    try:
                        # Try to get market data for sentiment derivation
                        market_data = None
                        try:
                            from datetime import datetime, timedelta

                            import yfinance as yf

                            # Fetch market data for sentiment derivation with consistent timezone handling
                            utc_now = datetime.now(UTC)
                            end_date_dt = utc_now.replace(tzinfo=None)  # Make naive for yfinance
                            start_date_dt = end_date_dt - timedelta(
                                days=sentiment_lookback_days + 30
                            )
                            start_date = start_date_dt.strftime("%Y-%m-%d")
                            end_date = end_date_dt.strftime("%Y-%m-%d")
                            ticker = yf.Ticker(symbol)
                            market_data = ticker.history(start=start_date, end=end_date, interval="1d")

                            if not market_data.empty:
                                # Rename columns to match our sentiment derivation
                                market_data = market_data.rename(columns={
                                    "Close": "close", "Open": "open", "High": "high",
                                    "Low": "low", "Volume": "volume"
                                })

                                # Ensure timestamps are timezone-naive for consistency
                                if hasattr(market_data.index, "tz_localize"):
                                    market_data.index = market_data.index.tz_localize(None)
                        except Exception as e:
                            console.print(f"[yellow]⚠️  Could not fetch market data for {symbol}: {e}[/yellow]")
                            market_data = None

                        # Use enhanced sentiment with market fallback
                        sentiment_score = analyzer.get_symbol_sentiment_with_market_fallback(
                            symbol, sentiment_days, market_data
                        )
                        sentiment_data = analyzer.fetch_all_sentiment(symbol, sentiment_days)

                        # Determine sentiment source for logging
                        sentiment_source = "unknown"
                        if sentiment_data:
                            sources = {d.source for d in sentiment_data}
                            if "news_api" in sources:
                                sentiment_source = "news_api"
                            elif "news_scrape" in sources:
                                sentiment_source = "web_scrape"
                            elif "market_derived" in sources:
                                sentiment_source = "market_derived"
                            elif "historical_mock" in sources:
                                sentiment_source = "enhanced_mock"
                            else:
                                sentiment_source = "other"

                        # Ensure we have valid sentiment data
                        if sentiment_score is None or np.isnan(sentiment_score):
                            sentiment_score = 0.0

                        sentiment_results[symbol] = {
                            "score": float(sentiment_score),
                            "data_points": len(sentiment_data) if sentiment_data else 0,
                            "sources": list({d.source for d in sentiment_data}) if sentiment_data else [],
                            "primary_source": sentiment_source,
                            "timestamp": datetime.now().isoformat()
                        }

                        successful_sentiment += 1
                        console.print(f"[green]✓ {symbol}: score={sentiment_score:.3f}, source={sentiment_source}, data_points={len(sentiment_data) if sentiment_data else 0}[/green]")

                    except Exception as e:
                        failed_sentiment += 1
                        console.print(f"[red]✗ {symbol}: {e}[/red]")

                        # Add default sentiment data for failed symbols
                        sentiment_results[symbol] = {
                            "score": 0.0,
                            "data_points": 0,
                            "sources": [],
                            "primary_source": "failed",
                            "timestamp": datetime.now().isoformat(),
                            "error": str(e)
                        }

                # Save sentiment results with error handling
                try:
                    import json
                    sentiment_file = sentiment_dir / f"sentiment_{start_date}_{end_date}.json"
                    with open(sentiment_file, "w") as f:
                        json.dump(sentiment_results, f, indent=2)

                    # Save sentiment features as CSV for easy integration
                    sentiment_features = analyzer.get_sentiment_features(symbol_list, sentiment_days)
                    sentiment_csv = sentiment_dir / f"sentiment_features_{start_date}_{end_date}.csv"
                    sentiment_features.to_csv(sentiment_csv, index=False)

                    console.print("[green]✅ Sentiment analysis completed![/green]")
                    console.print(f"[cyan]Successful: {successful_sentiment}/{len(symbol_list)}[/cyan]")
                    console.print(f"[cyan]Failed: {failed_sentiment}[/cyan]")
                    console.print(f"[cyan]Results saved to: {sentiment_file}[/cyan]")
                    console.print(f"[cyan]Features saved to: {sentiment_csv}[/cyan]")

                    if failed_sentiment > 0:
                        console.print(f"[yellow]⚠️  {failed_sentiment} symbols failed sentiment analysis - using default values (0.0)[/yellow]")

                except Exception as e:
                    console.print(f"[red]Error saving sentiment results: {e}[/red]")
                    console.print("[yellow]⚠️  Sentiment analysis completed but failed to save results[/yellow]")

            except ImportError as e:
                console.print(f"[red]Error importing sentiment analyzer: {e}[/red]")
                console.print("[yellow]⚠️  Sentiment analysis skipped - sentiment features will default to 0[/yellow]")

                # Create default sentiment features DataFrame
                import pandas as pd
                sentiment_features = pd.DataFrame({
                    "symbol": symbol_list,
                    "sentiment_score": [0.0] * len(symbol_list),
                    "sentiment_magnitude": [0.0] * len(symbol_list),
                    "sentiment_sources": [0] * len(symbol_list),
                    "sentiment_direction": [0] * len(symbol_list),
                })

                # Save default sentiment features
                sentiment_csv = sentiment_dir / f"sentiment_features_{start_date}_{end_date}.csv"
                sentiment_features.to_csv(sentiment_csv, index=False)
                console.print(f"[cyan]Default sentiment features saved to: {sentiment_csv}[/cyan]")

            except Exception as e:
                console.print(f"[red]Unexpected error in sentiment analysis: {e}[/red]")
                console.print("[yellow]⚠️  Sentiment analysis failed - sentiment features will default to 0[/yellow]")

                # Create default sentiment features DataFrame
                import pandas as pd
                sentiment_features = pd.DataFrame({
                    "symbol": symbol_list,
                    "sentiment_score": [0.0] * len(symbol_list),
                    "sentiment_magnitude": [0.0] * len(symbol_list),
                    "sentiment_sources": [0] * len(symbol_list),
                    "sentiment_direction": [0] * len(symbol_list),
                })

                # Save default sentiment features
                sentiment_csv = sentiment_dir / f"sentiment_features_{start_date}_{end_date}.csv"
                sentiment_features.to_csv(sentiment_csv, index=False)
                console.print(f"[cyan]Default sentiment features saved to: {sentiment_csv}[/cyan]")

        # Step 3: Process (if requested)
        if process:
            console.print("\n[blue]Step 3: Processing and standardizing data...[/blue]")

            # Determine input path for processing
            if input_path is None:
                input_path = raw_dir if download else Path("data/raw")

            console.print(f"[cyan]Input: {input_path}[/cyan]")
            console.print(f"[cyan]Output: {processed_dir}[/cyan]")
            console.print(f"[cyan]Method: {method}[/cyan]")
            console.print(f"[cyan]Force rebuild: {force_rebuild}[/cyan]")
            console.print(f"[cyan]Include sentiment features: {include_sentiment_features}[/cyan]")

            # Import and run process functionality
            from .data.prepare import prepare_data

            # If sentiment analysis was performed and we want to include sentiment features
            if include_sentiment_features and sentiment:
                console.print("[yellow]Integrating sentiment features into processed data...[/yellow]")

                # Load sentiment features
                sentiment_csv = sentiment_dir / f"sentiment_features_{start_date}_{end_date}.csv"
                if sentiment_csv.exists():
                    import pandas as pd
                    sentiment_df = pd.read_csv(sentiment_csv)
                    console.print(f"[green]Loaded sentiment features for {len(sentiment_df)} symbols[/green]")
                else:
                    console.print("[yellow]No sentiment features found, proceeding without sentiment[/yellow]")
                    sentiment_df = None
            else:
                sentiment_df = None

            prepare_data(
                input_path=input_path,
                output_dir=processed_dir,
                config_path=config_path,
                method=method,
                save_standardizer=save_standardizer,
                sentiment_data=sentiment_df
            )

            console.print("[green]✅ Data processing completed[/green]")

        console.print("\n[green]✅ Pipeline operations completed successfully![/green]")
        console.print(f"[cyan]Output directory: {output_dir}[/cyan]")
        console.print("[cyan]Pipeline structure:[/cyan]")
        if download:
            console.print("  - raw/ (downloaded market data)")
        if sentiment:
            console.print("  - sentiment/ (sentiment analysis data)")
        if process:
            console.print("  - processed/ (standardized data with features)")
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
