import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import ray
import yaml

from .csv_utils import save_csv_chunked
from .features import generate_features
from .historical import fetch_historical_data
from .live import fetch_live_data
from .loaders import load_alphavantage, load_ccxt, load_yfinance
from .synthetic import fetch_synthetic_data


class DataPipeline:
    """Unified data pipeline for CLI operations."""

    def __init__(self) -> None:
        """Initialize the data pipeline."""

    def download_data(
        self,
        symbols: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        output_dir: Path = Path("data"),
    ) -> list[str]:
        """Download market data for specified symbols."""
        from .professional_feeds import ProfessionalDataProvider

        provider = ProfessionalDataProvider("yahoo")  # Default to Yahoo Finance

        if not symbols:
            from .market_symbols import get_default_symbols_list
            symbols = get_default_symbols_list()  # Default symbols

        if not start_date:
            start_date = "2023-01-01"

        if not end_date:
            from datetime import datetime

            end_date = datetime.now().strftime("%Y-%m-%d")

        # Download data with mixed portfolio alignment
            data = provider.get_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        include_features=True,
        align_mixed_portfolio=True,  # Enable timestamp alignment for mixed portfolios
        alignment_strategy="last_known_value",  # Use last_known_value as default strategy
    )

        # Save to output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded_files = []
        for symbol in symbols:
            symbol_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data
            file_path = output_dir / f"{symbol}.csv"
            symbol_data.to_csv(file_path, index=False)
            downloaded_files.append(str(file_path))

        return downloaded_files


@ray.remote
def _fetch_data_remote(fetch_fn: Callable[..., pd.DataFrame], **kwargs: Any) -> pd.DataFrame:
    """Execute a data fetch function as a Ray remote task."""
    return fetch_fn(**kwargs)


def load_cached_csvs(directory: str = "data/processed") -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate them.

    Each resulting row gains a ``source`` column derived from the file name.

    Parameters
    ----------
    directory : str, default="data/processed"
        Folder containing CSV files previously produced by :func:`run_pipeline`.

    Returns
    -------
    pandas.DataFrame
        Single DataFrame with data from all CSVs combined. If the directory
        contains no CSV files an empty DataFrame is returned.
    """

    path = Path(directory)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    csv_files = sorted(path.glob("*.csv"))
    frames = []
    for csv in csv_files:
        df = pd.read_csv(csv, index_col=0)
        df["source"] = csv.stem
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_pipeline(config_path: str) -> dict[str, pd.DataFrame]:
    """Run the data ingestion pipeline using Ray for parallelism.

    Parameters
    ----------
    config_path : str
        Path to a YAML configuration file defining symbols and output options.
        Supported YAML keys include:
        ``coinbase_perp_symbols``, ``oanda_fx_symbols``, ``yfinance_symbols``,
        ``alphavantage_symbols`` and ``ccxt`` (mapping of exchange to symbols).

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of ``source`` keys to DataFrames containing OHLCV data.
    """
    with Path(config_path).open() as f:
        cfg = yaml.safe_load(f)

    start = cfg.get("start")
    end = cfg.get("end")
    timestep = cfg.get("timestep", "day")
    timezone = cfg.get("timezone", "America/New_York")  # Get timezone from config
    coinbase_symbols = cfg.get("coinbase_perp_symbols", [])
    oanda_symbols = cfg.get("oanda_fx_symbols", [])
    yfinance_symbols = cfg.get("yfinance_symbols", [])
    alphavantage_symbols = cfg.get("alphavantage_symbols", [])
    ccxt_sources = cfg.get("ccxt", {})
    output_dir = cfg.get("output_dir", "data/raw")
    to_csv = cfg.get("to_csv", True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=cfg.get("ray_address", os.getenv("RAY_ADDRESS")),
            log_to_driver=False,
        )
        started_ray = True

    tasks = {}

    for symbol in coinbase_symbols:
        key = f"coinbase_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_historical_data,
            symbol=symbol,
            start=start,
            end=end,
            timestep=timestep,
            timezone=timezone,  # Pass timezone to data fetching
        )

    for symbol in oanda_symbols:
        key = f"oanda_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_historical_data,
            symbol=symbol,
            start=start,
            end=end,
            timestep=timestep,
            timezone=timezone,  # Pass timezone to data fetching
        )

    for symbol in yfinance_symbols:
        key = f"yfinance_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            load_yfinance,
            symbol=symbol,
            start=start,
            end=end,
            interval=timestep,
            timezone=timezone,  # Pass timezone to data fetching
        )

    for symbol in alphavantage_symbols:
        key = f"alphavantage_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            load_alphavantage,
            symbol=symbol,
            start=start,
            end=end,
            interval=timestep,
            timezone=timezone,  # Pass timezone to data fetching
        )

    for exch, symbols in ccxt_sources.items():
        for symbol in symbols:
            key = f"{exch}_{symbol.replace('/', '')}"
            tasks[key] = _fetch_data_remote.remote(
                load_ccxt,
                symbol=symbol,
                start=start,
                end=end,
                interval=timestep,
                exchange=exch,
                timezone=timezone,  # Pass timezone to data fetching
            )

    freq_map = {"day": "D", "hour": "H", "minute": "T"}
    n_samples = 1
    if start and end:
        freq = freq_map.get(timestep, timestep)
        n_samples = len(pd.date_range(start=start, end=end, freq=freq))

    for symbol in cfg.get("synthetic_symbols", []):
        key = f"synthetic_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_synthetic_data,
            n_samples=n_samples,
            timeframe=timestep,
        )

    for symbol in cfg.get("live_symbols", []):
        key = f"live_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_live_data,
            symbol=symbol,
            start=start,
            end=end,
            timestep=timestep,
        )

    fetched = ray.get(list(tasks.values())) if tasks else []

    results = {}
    for key, df in zip(tasks.keys(), fetched, strict=False):
        if key.startswith("synthetic_") and "close" in df.columns:
            df = generate_features(df)
        if key.startswith("live_"):
            df = generate_features(df)

        results[key] = df
        if to_csv:
            csv_path = Path(output_dir) / f"{key}.csv"
            # Use chunked CSV saving for better efficiency
            save_csv_chunked(df, csv_path, chunk_size=10000, show_progress=True)

    if started_ray:
        ray.shutdown()

    return results


# CLI functionality moved to unified CLI in trading_rl_agent.cli
