import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import get_settings, load_settings

# Import yfinance at module level for mocking in tests
try:
    import yfinance as yf
except ImportError:
    yf = None

from .console import console, print_metrics_table
from .core.unified_config import BacktestConfig
from .eval.backtest_evaluator import BacktestEvaluator
from .portfolio.transaction_costs import BrokerType, TransactionCostModel

app = typer.Typer(help="Backtesting CLI for Trading RL Agent")

DEFAULT_EXPORT_CSV: Path | None = None
DEFAULT_CONFIG_FILE: Path | None = None


def _load_historical_data(symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load historical data for backtesting.

    Args:
        symbols: List of symbols to load
        start_date: Start date for data
        end_date: End date for data

    Returns:
        DataFrame with OHLCV data
    """
    if yf is None:
        console.print("[red]yfinance not available. Please install with: pip install yfinance[/red]")
        raise typer.Exit(1) from None

    try:
        # Load data for all symbols
        data_frames = []
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            data["symbol"] = symbol
            data_frames.append(data)

        # Combine all data
        combined_data = pd.concat(data_frames, ignore_index=True)

        # Ensure required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        for col in required_columns:
            if col not in combined_data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Rename columns to lowercase
        combined_data.columns = [col.lower() for col in combined_data.columns]

        return combined_data

    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1) from e


def _generate_sample_signals(data: pd.DataFrame, strategy_type: str = "momentum") -> pd.Series:
    """
    Generate sample trading signals for demonstration.

    Args:
        data: Historical price data
        strategy_type: Type of strategy to generate

    Returns:
        Series of trading signals
    """
    # Handle empty data
    if data.empty:
        return pd.Series(dtype=float)

    # Handle data without required columns
    if "close" not in data.columns:
        # Create a simple random signal if no close price data
        np.random.seed(42)
        return pd.Series(np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3]), index=data.index)

    if strategy_type == "momentum":
        # Simple momentum strategy
        returns = data["close"].pct_change()
        signals = pd.Series(0, index=data.index)
        # Handle case where rolling mean might be NaN
        rolling_mean = returns.rolling(20).mean()
        signals[returns > rolling_mean] = 1  # Buy on positive momentum
        signals[returns < rolling_mean] = -1  # Sell on negative momentum
    elif strategy_type == "mean_reversion":
        # Simple mean reversion strategy
        sma_20 = data["close"].rolling(20).mean()
        signals = pd.Series(0, index=data.index)
        signals[data["close"] < sma_20 * 0.95] = 1  # Buy when price is below SMA
        signals[data["close"] > sma_20 * 1.05] = -1  # Sell when price is above SMA
    else:
        # Random strategy for testing
        np.random.seed(42)
        signals = pd.Series(np.random.choice([-1, 0, 1], size=len(data), p=[0.3, 0.4, 0.3]), index=data.index)

    return signals


@app.command()
def run(
    strategy: str = typer.Argument(..., help="Strategy name (momentum, mean_reversion, or custom)"),
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: str | None = None,
    export_csv: Path | None = None,
    config_file: Path | None = None,
    initial_capital: float | None = None,
    commission_rate: float | None = None,
    slippage_rate: float | None = None,
) -> None:
    """
    Run a single strategy over a date range and print summary metrics.
    """
    start_date = start_date if start_date is not None else typer.Option(None, "--start", help="Start date (YYYY-MM-DD)")
    end_date = end_date if end_date is not None else typer.Option(None, "--end", help="End date (YYYY-MM-DD)")
    symbols = (
        symbols if symbols is not None else typer.Option(None, "--symbols", help="Comma-separated list of symbols")
    )
    export_csv = (
        export_csv
        if export_csv is not None
        else typer.Option(DEFAULT_EXPORT_CSV, "--export-csv", help="Export summary metrics to CSV")
    )
    config_file = (
        config_file
        if config_file is not None
        else typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c", help="Path to config file")
    )

    try:
        # Load settings
        settings = load_settings(config_path=config_file) if config_file else get_settings()

        # Use config or CLI overrides
        start = start_date or settings.backtest.start_date
        end = end_date or settings.backtest.end_date
        symbol_list = (
            [s.strip() for s in symbols.split(",")]
            if symbols and isinstance(symbols, str)
            else settings.backtest.symbols
        )
        capital = initial_capital or settings.backtest.initial_capital
        commission = commission_rate or settings.backtest.commission_rate
        slippage = slippage_rate or settings.backtest.slippage_rate

        console.print(f"[green]Running backtest: {strategy} ({start} to {end})[/green]")
        console.print(f"[blue]Symbols: {symbol_list}[/blue]")
        console.print(f"[blue]Initial Capital: ${capital:,.2f}[/blue]")

        # Load historical data
        console.print("[yellow]Loading historical data...[/yellow]")
        data = _load_historical_data(symbol_list, start, end)
        console.print(f"[green]Loaded {len(data)} data points[/green]")

        # Generate signals
        console.print("[yellow]Generating trading signals...[/yellow]")
        signals = _generate_sample_signals(data, strategy)

        # Configure backtest
        backtest_config = BacktestConfig(
            start_date=start,
            end_date=end,
            symbols=symbol_list,
            initial_capital=capital,
            commission_rate=commission,
            slippage_rate=slippage,
            max_position_size=settings.backtest.max_position_size,
            max_leverage=settings.backtest.max_leverage,
            stop_loss_pct=settings.backtest.stop_loss_pct,
            take_profit_pct=settings.backtest.take_profit_pct,
        )

        # Create transaction cost model
        cost_model = TransactionCostModel.create_broker_model(BrokerType.RETAIL)

        # Initialize backtest evaluator
        evaluator = BacktestEvaluator(backtest_config, cost_model)

        # Run backtest
        console.print("[yellow]Executing backtest...[/yellow]")
        results = evaluator.run_backtest(data, signals, strategy_name=strategy)

        # Display results
        console.print("\n[bold green]Backtest Results:[/bold green]")
        summary = [
            {
                "strategy": strategy,
                "period": f"{start} to {end}",
                "total_return": f"{results.total_return:.2%}",
                "sharpe_ratio": f"{results.sharpe_ratio:.2f}",
                "max_drawdown": f"{results.max_drawdown:.2%}",
                "win_rate": f"{results.win_rate:.2%}",
                "num_trades": results.num_trades,
                "total_costs": f"${results.total_transaction_costs:.2f}",
            }
        ]
        print_metrics_table(summary)

        # Generate detailed report
        if settings.backtest.save_trades:
            try:
                output_dir = (
                    str(settings.backtest.output_dir) if hasattr(settings.backtest.output_dir, "__str__") else "reports"
                )
                report_path = Path(output_dir) / f"{strategy}_report.txt"
                report_path.parent.mkdir(parents=True, exist_ok=True)

                with open(report_path, "w") as f:
                    f.write(f"# Backtest Report: {strategy}\n\n")
                    f.write(f"Period: {start} to {end}\n")
                    f.write(f"Symbols: {symbol_list}\n")
                    f.write(f"Initial Capital: ${capital:,.2f}\n\n")
                    f.write(f"Total Return: {results.total_return:.2%}\n")
                    f.write(f"Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
                    f.write(f"Max Drawdown: {results.max_drawdown:.2%}\n")
                    f.write(f"Win Rate: {results.win_rate:.2%}\n")
                    f.write(f"Number of Trades: {results.num_trades}\n")
                    f.write(f"Total Transaction Costs: ${results.total_transaction_costs:.2f}\n")

                console.print(f"[blue]Detailed report saved to {report_path}[/blue]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save detailed report: {e}[/yellow]")

        # Export to CSV
        if export_csv is not None and export_csv != typer.Option(None):
            try:
                summary_df = pd.DataFrame(summary)
                summary_df.to_csv(export_csv, index=False)
                console.print(f"[blue]Exported summary to {export_csv}[/blue]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not export CSV: {e}[/yellow]")

        raise typer.Exit(0)
    except typer.Exit:
        # Re-raise typer.Exit exceptions without modification
        raise
    except Exception as e:
        console.print(f"[red]Backtest error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def batch(
    strategies: str = typer.Argument(..., help="Comma-separated list of strategies"),
    periods: str = typer.Argument(
        ..., help="Comma-separated list of periods as start:end (e.g. 2023-01-01:2023-06-01,2023-06-02:2023-12-31)"
    ),
    symbols: str | None = None,
    export_csv: Path | None = None,
    config_file: Path | None = None,
    initial_capital: float | None = None,
) -> None:
    """
    Run multiple strategies over a grid of periods and print summary metrics.
    """
    export_csv = (
        export_csv
        if export_csv is not None
        else typer.Option(DEFAULT_EXPORT_CSV, "--export-csv", help="Export summary metrics to CSV")
    )
    config_file = (
        config_file
        if config_file is not None
        else typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c", help="Path to config file")
    )

    try:
        # Load settings
        settings = load_settings(config_path=config_file) if config_file else get_settings()

        # Parse inputs
        strat_list = [s.strip() for s in strategies.split(",")]
        period_list = [p.strip() for p in periods.split(",")]
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else settings.backtest.symbols
        capital = initial_capital or settings.backtest.initial_capital

        results = []

        for strat in strat_list:
            for period in period_list:
                if ":" not in period:
                    console.print(f"[yellow]Skipping invalid period: {period}[/yellow]")
                    continue

                start, end = period.split(":", 1)
                console.print(f"[green]Backtesting {strat} ({start} to {end})[/green]")

                try:
                    # Load data
                    data = _load_historical_data(symbol_list, start, end)

                    # Generate signals
                    signals = _generate_sample_signals(data, strat)

                    # Configure backtest
                    backtest_config = BacktestConfig(
                        start_date=start,
                        end_date=end,
                        symbols=symbol_list,
                        initial_capital=capital,
                        commission_rate=settings.backtest.commission_rate,
                        slippage_rate=settings.backtest.slippage_rate,
                        max_position_size=settings.backtest.max_position_size,
                        max_leverage=settings.backtest.max_leverage,
                        stop_loss_pct=settings.backtest.stop_loss_pct,
                        take_profit_pct=settings.backtest.take_profit_pct,
                    )

                    # Create transaction cost model
                    cost_model = TransactionCostModel.create_broker_model(BrokerType.RETAIL)

                    # Initialize backtest evaluator
                    evaluator = BacktestEvaluator(backtest_config, cost_model)

                    # Run backtest
                    backtest_results = evaluator.run_backtest(data, signals, strategy_name=strat)

                    results.append(
                        {
                            "strategy": strat,
                            "period": f"{start} to {end}",
                            "total_return": f"{backtest_results.total_return:.2%}",
                            "sharpe_ratio": f"{backtest_results.sharpe_ratio:.2f}",
                            "max_drawdown": f"{backtest_results.max_drawdown:.2%}",
                            "win_rate": f"{backtest_results.win_rate:.2%}",
                            "num_trades": backtest_results.num_trades,
                            "total_costs": f"${backtest_results.total_transaction_costs:.2f}",
                        }
                    )

                except Exception as be:
                    console.print(f"[red]Backtest failed for {strat} ({start} to {end}): {be}[/red]")
                    results.append(
                        {
                            "strategy": strat,
                            "period": f"{start} to {end}",
                            "total_return": "0.00%",
                            "sharpe_ratio": "0.00",
                            "max_drawdown": "0.00%",
                            "win_rate": "0.00%",
                            "num_trades": 0,
                            "total_costs": "$0.00",
                        }
                    )

        print_metrics_table(results)

        if export_csv is not None and export_csv != typer.Option(None):
            try:
                pd.DataFrame(results).to_csv(export_csv, index=False)
                console.print(f"[blue]Exported summary to {export_csv}[/blue]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not export CSV: {e}[/yellow]")

        raise typer.Exit(0)
    except typer.Exit:
        # Re-raise typer.Exit exceptions without modification
        raise
    except Exception as e:
        console.print(f"[red]Batch backtest error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


@app.command()
def compare(
    strategies: str = typer.Argument(..., help="Comma-separated list of strategies to compare"),
    start_date: str | None = None,
    end_date: str | None = None,
    symbols: str | None = None,
    config_file: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Compare multiple strategies using the unified backtest evaluator.
    """
    start_date = start_date if start_date is not None else typer.Option(None, "--start", help="Start date (YYYY-MM-DD)")
    end_date = end_date if end_date is not None else typer.Option(None, "--end", help="End date (YYYY-MM-DD)")
    symbols = (
        symbols if symbols is not None else typer.Option(None, "--symbols", help="Comma-separated list of symbols")
    )
    config_file = (
        config_file
        if config_file is not None
        else typer.Option(DEFAULT_CONFIG_FILE, "--config", "-c", help="Path to config file")
    )

    try:
        # Load settings
        settings = load_settings(config_path=config_file) if config_file else get_settings()

        # Parse inputs
        strat_list = [s.strip() for s in strategies.split(",")]
        start = start_date or settings.backtest.start_date
        end = end_date or settings.backtest.end_date
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else settings.backtest.symbols

        # Handle output directory
        if output_dir is not None:
            output_path = Path(output_dir)
        else:
            try:
                output_path = (
                    Path(str(settings.backtest.output_dir))
                    if hasattr(settings.backtest.output_dir, "__str__")
                    else Path("reports")
                )
            except Exception:
                output_path = Path("reports")

        console.print(f"[green]Comparing strategies: {strat_list}[/green]")
        console.print(f"[blue]Period: {start} to {end}[/blue]")
        console.print(f"[blue]Symbols: {symbol_list}[/blue]")

        # Load historical data
        console.print("[yellow]Loading historical data...[/yellow]")
        data = _load_historical_data(symbol_list, start, end)

        # Generate signals for each strategy
        strategy_signals = {}
        for strategy in strat_list:
            signals = _generate_sample_signals(data, strategy)
            strategy_signals[strategy] = signals

        # Configure backtest
        backtest_config = BacktestConfig(
            start_date=start,
            end_date=end,
            symbols=symbol_list,
            initial_capital=settings.backtest.initial_capital,
            commission_rate=settings.backtest.commission_rate,
            slippage_rate=settings.backtest.slippage_rate,
            max_position_size=settings.backtest.max_position_size,
            max_leverage=settings.backtest.max_leverage,
            stop_loss_pct=settings.backtest.stop_loss_pct,
            take_profit_pct=settings.backtest.take_profit_pct,
        )

        # Create transaction cost model
        cost_model = TransactionCostModel.create_broker_model(BrokerType.RETAIL)

        # Initialize backtest evaluator
        evaluator = BacktestEvaluator(backtest_config, cost_model)

        # Compare strategies
        console.print("[yellow]Running strategy comparison...[/yellow]")
        comparison_results: dict | Any = evaluator.compare_strategies(data, strategy_signals)

        # Generate comparison report
        if settings.backtest.save_trades:
            output_path.mkdir(parents=True, exist_ok=True)
            comparison_report_path = output_path / "strategy_comparison_report.txt"

            with open(comparison_report_path, "w") as f:
                f.write("# Strategy Comparison Report\n\n")
                if isinstance(comparison_results, dict):
                    for strategy_name, results in comparison_results.items():
                        f.write(f"## {strategy_name}\n")
                        f.write(f"- Total Return: {results.total_return:.2%}\n")
                        f.write(f"- Sharpe Ratio: {results.sharpe_ratio:.2f}\n")
                        f.write(f"- Max Drawdown: {results.max_drawdown:.2%}\n")
                        f.write(f"- Win Rate: {results.win_rate:.2%}\n")
                        f.write(f"- Number of Trades: {results.num_trades}\n")
                        f.write(f"- Total Transaction Costs: ${results.total_transaction_costs:.2f}\n\n")
                else:
                    # Handle non-dict results - this is reachable in some cases
                    f.write("## Comparison Results\n")
                    f.write("Results format not supported for detailed reporting.\n")

            console.print(f"[blue]Comparison report saved to {comparison_report_path}[/blue]")

        # Display results in console
        if isinstance(comparison_results, dict):
            results_list = []
            for strategy_name, results in comparison_results.items():
                results_list.append(
                    {
                        "strategy": strategy_name,
                        "total_return": f"{results.total_return:.2%}",
                        "sharpe_ratio": f"{results.sharpe_ratio:.2f}",
                        "max_drawdown": f"{results.max_drawdown:.2%}",
                        "win_rate": f"{results.win_rate:.2%}",
                        "num_trades": results.num_trades,
                    }
                )
            print_metrics_table(results_list)

        console.print("[bold green]✅ Strategy comparison complete[/bold green]")
        raise typer.Exit(0)

    except typer.Exit:
        # Re-raise typer.Exit exceptions without modification
        raise
    except Exception as e:
        console.print(f"[red]Strategy comparison error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
