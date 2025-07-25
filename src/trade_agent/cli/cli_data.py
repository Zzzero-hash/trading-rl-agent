"""
Data pipeline CLI commands.

This module contains all data-related CLI commands including:
- Data download and fetching
- Data processing and standardization
- Pipeline operations
"""

from pathlib import Path
from typing import Annotated

import typer

from .cli_main import (
    DEFAULT_DATA_OUTPUT,
    DEFAULT_DATA_SOURCE,
    DEFAULT_END_DATE,
    DEFAULT_FORCE,
    DEFAULT_PARALLEL,
    DEFAULT_PIPELINE_OUTPUT,
    DEFAULT_REFRESH_DAYS,
    DEFAULT_STANDARDIZATION_METHOD,
    DEFAULT_STANDARDIZED_OUTPUT,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOLS_STR,
    DEFAULT_TIMEFRAME,
    console,
    logger,
)

# Data operations sub-app
data_app = typer.Typer(
    name="data",
    help="Data pipeline operations: download, prepare, pipeline",
    rich_markup_mode="rich",
)


@data_app.command()
def pipeline(
    symbols: Annotated[str, typer.Argument(..., help="Comma-separated list of symbols (e.g., 'AAPL,GOOGL,MSFT')")],
    output_dir: Path = DEFAULT_PIPELINE_OUTPUT,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    source: str = DEFAULT_DATA_SOURCE,
    timeframe: str | None = DEFAULT_TIMEFRAME,
    force_rebuild: bool = DEFAULT_FORCE,
    parallel: bool = DEFAULT_PARALLEL,
) -> None:
    """
    Run the complete data pipeline for specified symbols.

    This command downloads raw data, applies feature engineering, and prepares
    the dataset for training. It's the recommended way to prepare data for model training.

    Examples:
        trade-agent data pipeline AAPL,GOOGL,MSFT
        trade-agent data pipeline "AAPL,TSLA" --start-date 2020-01-01 --source yfinance
        trade-agent data pipeline MSFT --force-rebuild --parallel
    """
    console.print(f"[bold blue]Running data pipeline for symbols: {symbols}[/bold blue]")

    try:
        from trade_agent.data.pipeline import DataPipeline

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        console.print(f"Processing symbols: {symbol_list}")

        # Initialize pipeline
        pipeline = DataPipeline()

        # Configure pipeline parameters
        pipeline_config = {
            "symbols": symbol_list,
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
            "timeframe": timeframe,
            "output_dir": output_dir,
            "force_rebuild": force_rebuild,
            "parallel": parallel,
        }

        console.print("[yellow]Pipeline configuration:[/yellow]")
        for key, value in pipeline_config.items():
            if value is not None:
                console.print(f"  {key}: {value}")

        # Run pipeline
        console.print("[green]Starting data pipeline...[/green]")
        result = pipeline.run(**pipeline_config)

        if result and result.get("success", False):
            console.print("[bold green]✓ Pipeline completed successfully![/bold green]")
            console.print(f"Output saved to: {output_dir}")

            # Show summary statistics
            if "stats" in result:
                stats = result["stats"]
                console.print("\nDataset summary:")
                console.print(f"  Records: {stats.get('total_records', 'Unknown')}")
                console.print(f"  Features: {stats.get('total_features', 'Unknown')}")
                console.print(f"  Date range: {stats.get('date_range', 'Unknown')}")
        else:
            console.print("[bold red]✗ Pipeline failed![/bold red]")
            if result and "error" in result:
                console.print(f"Error: {result['error']}")

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
        console.print("Please ensure all dependencies are installed.")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@data_app.command()
def download(
    symbols: Annotated[str, typer.Argument(..., help="Comma-separated list of symbols")],
    output_dir: Path = DEFAULT_DATA_OUTPUT,
    start_date: str | None = DEFAULT_START_DATE,
    end_date: str | None = DEFAULT_END_DATE,
    source: str = DEFAULT_DATA_SOURCE,
    force: bool = DEFAULT_FORCE,
) -> None:
    """
    Download raw market data for specified symbols.

    Downloads OHLCV data from the specified data source and saves it in CSV format.
    This is the first step in the data pipeline.

    Examples:
        trade-agent data download AAPL,GOOGL,MSFT
        trade-agent data download "TSLA,NVDA" --start-date 2023-01-01 --source yfinance
    """
    console.print(f"[bold blue]Downloading data for symbols: {symbols}[/bold blue]")

    try:
        from trade_agent.data.fetcher import DataFetcher

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Initialize fetcher
        fetcher = DataFetcher(source=source)

        # Download data
        console.print(f"[green]Downloading from {source}...[/green]")

        for symbol in symbol_list:
            console.print(f"  Fetching {symbol}...")

            try:
                data = fetcher.fetch_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    force=force
                )

                # Save data
                output_file = output_dir / f"{symbol}.csv"
                output_dir.mkdir(parents=True, exist_ok=True)
                data.to_csv(output_file, index=True)

                console.print(f"    ✓ Saved {len(data)} records to {output_file}")

            except Exception as e:
                console.print(f"    ✗ Failed to fetch {symbol}: {e}")
                continue

        console.print("[bold green]✓ Download completed![/bold green]")

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Download error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@data_app.command()
def standardize(
    input_dir: Annotated[Path, typer.Argument(..., help="Directory containing raw CSV data files")],
    output_dir: Path = DEFAULT_STANDARDIZED_OUTPUT,
    method: str = DEFAULT_STANDARDIZATION_METHOD,
    symbols: str | None = DEFAULT_SYMBOLS_STR,
) -> None:
    """
    Standardize raw data files to a consistent format.

    Applies data standardization, cleaning, and feature engineering to prepare
    data for model training.

    Examples:
        trade-agent data standardize data/raw --method robust
        trade-agent data standardize data/raw --symbols AAPL,GOOGL --output-dir data/processed
    """
    console.print(f"[bold blue]Standardizing data from {input_dir}[/bold blue]")

    try:
        from trade_agent.data.standardizer import DataStandardizer

        # Initialize standardizer
        standardizer = DataStandardizer(method=method)

        # Parse symbols if provided
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            console.print(f"Processing symbols: {symbol_list}")

        # Standardize data
        console.print(f"[green]Standardizing using {method} method...[/green]")

        result = standardizer.standardize(
            input_dir=input_dir,
            output_dir=output_dir,
            symbols=symbol_list
        )

        if result.get("success", False):
            console.print("[bold green]✓ Standardization completed![/bold green]")
            console.print(f"Output saved to: {output_dir}")

            # Show summary
            if "processed_files" in result:
                console.print(f"Processed {len(result['processed_files'])} files")
        else:
            console.print("[bold red]✗ Standardization failed![/bold red]")
            if "error" in result:
                console.print(f"Error: {result['error']}")

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Standardization error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e


@data_app.command()
def refresh(
    symbols: Annotated[str, typer.Argument(..., help="Comma-separated list of symbols")],
    days: int = DEFAULT_REFRESH_DAYS,
    output_dir: Path = DEFAULT_DATA_OUTPUT,
) -> None:
    """
    Refresh existing data with latest market data.

    Updates existing datasets with the most recent market data for specified symbols.
    Useful for keeping datasets current without full re-download.

    Examples:
        trade-agent data refresh AAPL,GOOGL,MSFT
        trade-agent data refresh "TSLA,NVDA" --days 7
    """
    console.print(f"[bold blue]Refreshing {days} days of data for: {symbols}[/bold blue]")

    try:
        from trade_agent.data.fetcher import DataFetcher

        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(",")]

        # Initialize fetcher
        fetcher = DataFetcher()

        console.print("[green]Refreshing data...[/green]")

        for symbol in symbol_list:
            console.print(f"  Updating {symbol}...")

            try:
                result = fetcher.refresh_data(
                    symbol=symbol,
                    days=days,
                    output_dir=output_dir
                )

                if result.get("success", False):
                    new_records = result.get("new_records", 0)
                    console.print(f"    ✓ Added {new_records} new records")
                else:
                    console.print(f"    ✗ Failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                console.print(f"    ✗ Failed to refresh {symbol}: {e}")
                continue

        console.print("[bold green]✓ Refresh completed![/bold green]")

    except ImportError as e:
        console.print(f"[bold red]Import error: {e}[/bold red]")
    except Exception as e:
        logger.error(f"Refresh error: {e}")
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1) from e
