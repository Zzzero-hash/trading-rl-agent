import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import typer

# Add root directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import get_settings, load_settings

from .console import console, print_metrics_table

app = typer.Typer(help="Backtesting CLI for Trading RL Agent")

# Placeholder for the backtest engine import
bt_mod: Any | None = None

DEFAULT_EXPORT_CSV: Path | None = None
DEFAULT_CONFIG_FILE: Path | None = None


@app.command()
def run(
    strategy: str = typer.Argument(..., help="Strategy name (must be implemented in backtest engine)"),
    start_date: str | None = None,
    end_date: str | None = None,
    export_csv: Path | None = None,
    config_file: Path | None = None,
) -> None:
    """
    Run a single strategy over a date range and print summary metrics.
    """
    start_date = start_date if start_date is not None else typer.Option(None, "--start", help="Start date (YYYY-MM-DD)")
    end_date = end_date if end_date is not None else typer.Option(None, "--end", help="End date (YYYY-MM-DD)")
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
        settings = load_settings(config_path=config_file) if config_file else get_settings()
        if not bt_mod:
            raise ImportError("Backtest engine module not found.")
        if not hasattr(bt_mod, "run_backtest"):
            raise ImportError("run_backtest function not found in backtest engine.")

        # Use config or CLI overrides
        start = start_date or settings.backtest.start_date
        end = end_date or settings.backtest.end_date

        console.print(f"[green]Running backtest: {strategy} ({start} to {end})[/green]")
        result = bt_mod.run_backtest(
            strategy=strategy,
            start_date=start,
            end_date=end,
            settings=settings,
        )
        # Assume result is a dict with keys: cagr, sharpe, max_drawdown
        summary = [
            {
                "strategy": strategy,
                "period": f"{start} to {end}",
                **{k: result.get(k, 0) for k in ("cagr", "sharpe", "max_drawdown")},
            }
        ]
        print_metrics_table(summary)
        if export_csv:
            pd.DataFrame(summary).to_csv(export_csv, index=False)
            console.print(f"[blue]Exported summary to {export_csv}[/blue]")
        raise typer.Exit(0)
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
    export_csv: Path | None = None,
    config_file: Path | None = None,
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
        settings = load_settings(config_path=config_file) if config_file else get_settings()
        if not bt_mod:
            raise ImportError("Backtest engine module not found.")
        if not hasattr(bt_mod, "run_backtest"):
            raise ImportError("run_backtest function not found in backtest engine.")

        strat_list = [s.strip() for s in strategies.split(",")]
        period_list = [p.strip() for p in periods.split(",")]
        results = []
        for strat in strat_list:
            for period in period_list:
                if ":" not in period:
                    console.print(f"[yellow]Skipping invalid period: {period}[/yellow]")
                    continue
                start, end = period.split(":", 1)
                console.print(f"[green]Backtesting {strat} ({start} to {end})[/green]")
                try:
                    result = bt_mod.run_backtest(
                        strategy=strat,
                        start_date=start,
                        end_date=end,
                        settings=settings,
                    )
                    results.append(
                        {
                            "strategy": strat,
                            "period": f"{start} to {end}",
                            **{k: result.get(k, 0) for k in ("cagr", "sharpe", "max_drawdown")},
                        }
                    )
                except Exception as be:
                    console.print(f"[red]Backtest failed for {strat} ({start} to {end}): {be}[/red]")
                    results.append(
                        {"strategy": strat, "period": f"{start} to {end}", "cagr": 0, "sharpe": 0, "max_drawdown": 0}
                    )
        print_metrics_table(results)
        if export_csv:
            pd.DataFrame(results).to_csv(export_csv, index=False)
            console.print(f"[blue]Exported summary to {export_csv}[/blue]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Batch backtest error: {e}[/red]")
        traceback.print_exc()
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
