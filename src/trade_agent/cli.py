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
    # Pipeline step options
    download: bool = typer.Option(False, "--download", "-d", help="Download market data"),
    sentiment: bool = typer.Option(False, "--sentiment", "-s", help="Collect sentiment analysis data"),
    process: bool = typer.Option(False, "--process", "-p", help="Process and standardize data"),
    run: bool = typer.Option(False, "--run", "-r", help="Run complete pipeline end-to-end (download → sentiment → process)"),

    # Optimization options
    max_symbols: int = typer.Option(50, "--max-symbols", help="Maximum symbols to process"),
    parallel_workers: int = typer.Option(8, "--workers", help="Number of parallel workers"),
    skip_sentiment: bool = typer.Option(False, "--skip-sentiment", help="Skip sentiment analysis"),
    use_cache: bool = typer.Option(True, "--use-cache", help="Use intelligent caching"),

    # Common parameters
    config_path: Path | None = DEFAULT_CONFIG_FILE,
    output_dir: Path = Path("data"),
    symbols: str | None = DEFAULT_SYMBOLS_STR,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
    save_standardizer: bool = True,
    # Sentiment-specific parameters
    sentiment_days: int = typer.Option(7, "--sentiment-days", help="Number of days back for sentiment analysis"),
    sentiment_sources: str = typer.Option(
        "news,social", "--sentiment-sources", help="Comma-separated sentiment sources (news,social,scrape)"
    ),
) -> None:
    """
    Unified data pipeline operations with sentiment analysis support.
    """
    monitor = PerformanceMonitor()
    with monitor.time_operation("Total Pipeline"):
        try:
            # Initialize cache manager
            cache_manager = CacheManager()

            # Set defaults
            if config_path is None:
                config_path = Path("config.yaml")

            if start_date is None:
                from datetime import datetime, timedelta

                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

            if end_date is None:
                from datetime import datetime

                end_date = datetime.now().strftime("%Y-%m-%d")

            # Optimize symbol selection
            if symbols is None:
                with monitor.time_operation("Select Symbols"):
                    # Get comprehensive symbols without validation to avoid API calls
                    from .data.market_symbols import get_optimized_symbols
                    symbol_list = get_optimized_symbols(max_symbols=max_symbols)
                    symbols = ",".join(symbol_list)
                    console.print("[green]Pipeline Download: Optimized Market Coverage[/green]")
                    console.print(f"[cyan]Total symbols: {len(symbol_list)}[/cyan]")
            else:
                symbol_list = [s.strip() for s in symbols.split(",")]

            # Determine which steps to run
            if run:
                download = True
                sentiment = True
                process = True

            if not any([download, sentiment, process]):
                console.print(
                    "[yellow]No pipeline steps specified. Use --download, --sentiment, --process, or --run[/yellow]"
                )
                console.print("[yellow]Example: python main.py data pipeline --run --symbols 'AAPL'[/yellow]")
                raise typer.Exit(1)

            console.print("[green]🚀 Unified Data Pipeline Operations[/green]")
            steps = []
            if download:
                steps.append("download")
            if sentiment and not skip_sentiment:
                steps.append("sentiment")
            if process:
                steps.append("process")
            console.print(f"[cyan]Steps: {' → '.join(steps)}[/cyan]")

            # Create output directory structure
            output_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = output_dir / "raw"
            sentiment_dir = output_dir / "sentiment"
            processed_dir = output_dir / "processed"
            for dir_path in [raw_dir, sentiment_dir, processed_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)

            sentiment_features = None

            # Step 1: Download (if requested)
            if download:
                with monitor.time_operation("Download Data"):
                    cache_key = cache_manager.get_cache_key("download", symbols=symbol_list, start_date=start_date, end_date=end_date)
                    downloaded_files = cache_manager.get_cached_data(cache_key) if use_cache else None

                    if downloaded_files is None:
                        pipeline = DataPipeline()
                        downloaded_files = pipeline.download_data_parallel(
                            symbols=symbol_list,
                            start_date=start_date,
                            end_date=end_date,
                            max_workers=parallel_workers,
                        )
                        if use_cache:
                            cache_manager.cache_data(cache_key, downloaded_files)

                console.print(f"[green]✅ Downloaded {len(downloaded_files)} files with historical sentiment[/green]")
                for file_path in downloaded_files:
                    console.print(f"[cyan]  - {file_path}[/cyan]")

            # Step 2: Sentiment Analysis (if requested)
            if sentiment and not skip_sentiment:
                console.print("\n[blue]Step 2: Collecting sentiment analysis data...[/blue]")
                sentiment_source_list = [s.strip() for s in sentiment_sources.split(",")]
                console.print(f"[cyan]Sources: {', '.join(sentiment_source_list)}[/cyan]")
                console.print(f"[cyan]Output: {sentiment_dir}[/cyan]")

                # Import sentiment analyzer with error handling
                try:
                    analyzer = SentimentAnalyzer()

                    with monitor.time_operation("Sentiment Analysis"):
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

                    # Save sentiment results with error handling
                    if sentiment_features is not None and not sentiment_features.empty:
                        sentiment_features.to_csv(sentiment_dir / "sentiment_features.csv", index=False)
                        console.print(f"[green]✅ Sentiment features saved to {sentiment_dir / 'sentiment_features.csv'}[/green]")
                    else:
                        console.print("[yellow]No sentiment features data found or cached.[/yellow]")
                except ImportError as e:
                    console.print(f"[red]Error importing SentimentAnalyzer: {e}[/red]")
                    raise typer.Exit(1) from e
                except Exception as e:
                    console.print(f"[red]Error during sentiment analysis: {e}[/red]")
                    logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
                    raise typer.Exit(1) from e

            # Step 3: Process (if requested)
            if process:
                with monitor.time_operation("Processing and standardizing data"):
                    from trade_agent.data.prepare import prepare_data

                    prepare_data(
                        input_path=raw_dir,
                        output_dir=processed_dir,
                        config_path=config_path,
                        method=method,
                        save_standardizer=save_standardizer,
                        sentiment_data=sentiment_features,
                    )

            console.print("\n[green]✅ Pipeline operations completed successfully![/green]")
            console.print("\n[bold green]Pipeline Performance Summary[/bold green]")
            summary = monitor.get_summary()
            console.print(f"Total time: {summary['total_time']:.2f}s")
            if summary["slowest_operation"]:
                console.print(f"Slowest operation: {summary['slowest_operation'][0]} ({summary['slowest_operation'][1]:.2f}s)")

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
    from .data.market_symbols import get_comprehensive_symbols
    return get_comprehensive_symbols()


if __name__ == "__main__":
    app()
