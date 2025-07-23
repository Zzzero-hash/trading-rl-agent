import os
import sys
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
import yaml

# Add src to path to resolve imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from trade_agent.utils.ray_utils import parallel_execute

from .csv_utils import save_csv_chunked
from .features import generate_features
from .historical import fetch_historical_data
from .live import fetch_live_data
from .loaders import load_alphavantage, load_ccxt, load_yfinance
from .sentiment import SentimentData
from .synthetic import fetch_synthetic_data


@ray.remote
def _download_symbol_data(
    symbol: str, start_date: str, end_date: str, provider: str = "yahoo"
) -> tuple[str, pd.DataFrame]:
    """Download data for a single symbol as a Ray remote task."""
    try:
        from .professional_feeds import ProfessionalDataProvider

        provider_instance = ProfessionalDataProvider(provider)

        data = provider_instance.get_market_data(
            symbols=[symbol],
            start_date=start_date,
            end_date=end_date,
            include_features=False,  # Defer feature generation
            align_mixed_portfolio=False,  # Defer alignment
        )

        return symbol, data
    except Exception as e:
        print(f"Failed to download {symbol}: {e}")
        return symbol, pd.DataFrame()


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
        include_sentiment: bool = True,
        sentiment_lookback_days: int = 30,
        sources: list[str] | None = None,
    ) -> list[str]:
        """Download market data for specified symbols with comprehensive historical collection."""
        from .professional_feeds import ProfessionalDataProvider

        provider = ProfessionalDataProvider("yahoo")  # Default to Yahoo Finance

        if not symbols:
            from .market_symbols import get_default_symbols_list
            symbols = get_default_symbols_list()  # Default symbols

        if not start_date:
            start_date = "2023-01-01"

        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if sources is None:
            sources = ["yfinance", "alphavantage"]  # Default sources

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

        downloaded_files: list[str] = []
        for symbol in symbols:
            # Ensure symbol_data is a DataFrame
            symbol_data_slice = data[data["symbol"] == symbol] if "symbol" in data.columns else data
            symbol_data: pd.DataFrame = symbol_data_slice.copy() if isinstance(symbol_data_slice, pd.DataFrame) else pd.DataFrame()


            # Enhance with historical sentiment if requested
            if include_sentiment and not symbol_data.empty:
                symbol_data = self._enhance_with_historical_sentiment(
                    symbol_data, symbol, start_date, end_date, sentiment_lookback_days
                )

            file_path = output_dir / f"{symbol}.csv"
            if not symbol_data.empty:
                symbol_data.to_csv(file_path, index=False)
            downloaded_files.append(str(file_path))

        return downloaded_files

    def download_data_parallel(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        max_workers: int = 8,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Download data for multiple symbols in parallel."""
        from .professional_feeds import ProfessionalDataProvider

        provider = ProfessionalDataProvider("yahoo")

        def _download_symbol(symbol: str) -> pd.DataFrame:
            return provider.get_market_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                include_features=False,
                align_mixed_portfolio=False,
            )

        results: list[pd.DataFrame] = parallel_execute(_download_symbol, symbols, max_workers=max_workers)

        all_data = [data for data in results if not data.empty]

        if not all_data:
            return pd.DataFrame()

        combined_data = pd.concat(all_data, ignore_index=True)

        if kwargs.get("include_features", True):
            from .features import generate_features
            combined_data = generate_features(combined_data)

        if kwargs.get("align_mixed_portfolio", True):
            from .market_calendar import get_trading_calendar
            calendar = get_trading_calendar()
            combined_data = calendar.align_data_timestamps(
                combined_data, symbols, alignment_strategy=kwargs.get("alignment_strategy", "last_known_value")
            )

        return combined_data


    def _enhance_with_historical_sentiment(
        self,
        market_data: pd.DataFrame,
        symbol: str,
        start_date: str,
        end_date: str,
        sentiment_lookback_days: int,
    ) -> pd.DataFrame:
        """Enhance market data with historical sentiment using market-derived fallback."""
        from .sentiment import NewsSentimentProvider

        # Initialize sentiment provider
        sentiment_provider = NewsSentimentProvider()

        # Calculate sentiment collection period
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        collection_start_dt = start_dt - timedelta(days=sentiment_lookback_days)

        # Get historical sentiment
        sentiment_data = sentiment_provider.fetch_sentiment(
            symbol,
            days_back=(end_dt - collection_start_dt).days,
        )

        if not sentiment_data:
            # Fallback to market-derived sentiment
            derived_sentiment = self._get_market_derived_sentiment(market_data, symbol, end_dt)
            if derived_sentiment:
                sentiment_data = [SentimentData(symbol=symbol, **derived_sentiment)]


        if not sentiment_data:
            return market_data

        # Convert to DataFrame
        sentiment_df = pd.DataFrame([vars(s) for s in sentiment_data])

        # Ensure 'timestamp' column is present and in the correct format
        sentiment_df = self._ensure_timestamp_column(sentiment_df)
        market_data = self._ensure_timestamp_column(market_data)


        # Merge sentiment data with market data
        return self._merge_market_sentiment(market_data, sentiment_df)

    def _aggregate_daily_sentiment(
        self,
        sentiment_data: list[Any],
        target_date: datetime
    ) -> dict[str, Any]:
        """Aggregate daily sentiment data."""
        if not sentiment_data:
            return {
                "timestamp": target_date,
                "sentiment_score": 0.0,
                "sentiment_magnitude": 0.0,
                "sentiment_sources": 0,
                "sentiment_direction": 0,
                "sentiment_source": "no_data"
            }

        # Filter sentiment data for the target date
        daily_sentiments = [
            s for s in sentiment_data
            if s.timestamp.date() == target_date.date()
        ]

        if not daily_sentiments:
            return {
                "timestamp": target_date,
                "sentiment_score": 0.0,
                "sentiment_magnitude": 0.0,
                "sentiment_sources": 0,
                "sentiment_direction": 0,
                "sentiment_source": "no_data"
            }

        # Calculate aggregated metrics
        scores = [s.score for s in daily_sentiments]
        magnitudes = [s.magnitude for s in daily_sentiments]
        sources = list({s.source for s in daily_sentiments})

        return {
            "timestamp": target_date,
            "sentiment_score": sum(scores) / len(scores),
            "sentiment_magnitude": sum(magnitudes) / len(magnitudes),
            "sentiment_sources": len(sources),
            "sentiment_direction": 1 if sum(scores) > 0 else (-1 if sum(scores) < 0 else 0),
            "sentiment_source": ",".join(sources)
        }

    def _get_market_derived_sentiment(
        self,
        market_data: pd.DataFrame,
        _symbol: str,
        target_date: datetime,
    ) -> dict[str, Any] | None:
        """Derive sentiment from market data as a proxy."""
        if market_data.empty:
            return None

        # Ensure target_date is timezone-naive
        if target_date.tzinfo is not None:
            target_date = target_date.replace(tzinfo=None)

        # Set timestamp as index for asof
        if "timestamp" in market_data.columns:
            market_data = market_data.set_index("timestamp")

        # Use asof to find the closest available data point
        closest_data = market_data.asof(target_date)

        if pd.isna(closest_data).all():
            return None

        # Simple momentum-based sentiment
        price_change = (closest_data["close"] - closest_data["open"]) / closest_data["open"]
        volume_change = (closest_data["volume"] - closest_data.get("volume_prev_day", 0)) / closest_data.get(
            "volume_prev_day", 1
        )

        # Normalize to score between -1 and 1
        score = np.tanh(price_change * 10)  # Scale to be more sensitive
        magnitude = np.tanh(abs(volume_change))  # Volume change affects confidence

        return {
            "timestamp": target_date,
            "score": float(score),
            "magnitude": float(magnitude),
            "source": "market_derived",
            "raw_data": {}
        }

    def _merge_market_sentiment(
        self,
        market_data: pd.DataFrame,
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge market data with sentiment data."""

        if market_data.empty:
            return market_data

        if sentiment_data.empty:
            # Add default sentiment columns
            market_data["sentiment_score"] = 0.0
            market_data["sentiment_magnitude"] = 0.0
            market_data["sentiment_sources"] = 0
            market_data["sentiment_direction"] = 0
            market_data["sentiment_source"] = "no_data"
            return market_data

        # Ensure market data has a timestamp column
        market_data = self._ensure_timestamp_column(market_data)

        # Ensure sentiment data has a timestamp column
        sentiment_data = self._ensure_timestamp_column(sentiment_data)

        # Ensure timestamps are compatible
        market_data["date"] = pd.to_datetime(market_data["timestamp"]).dt.date
        sentiment_data["date"] = pd.to_datetime(sentiment_data["timestamp"]).dt.date

        # Merge on date
        merged = market_data.merge(
            sentiment_data,
            on="date",
            how="left",
            suffixes=("", "_sentiment")
        )

        # Fill missing sentiment values
        sentiment_columns = ["sentiment_score", "sentiment_magnitude", "sentiment_sources", "sentiment_direction", "sentiment_source"]
        for col in sentiment_columns:
            if col in merged.columns:
                if col == "sentiment_source":
                    merged[col] = merged[col].fillna("no_data")
                else:
                    merged[col] = merged[col].fillna(0.0)

        # Clean up
        merged = merged.drop("date", axis=1)

        return merged

    def _ensure_timestamp_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has a timestamp column, creating one if needed."""
        df_copy = df.copy()

        # Check if timestamp column exists
        if "timestamp" in df_copy.columns:
            return df_copy

        # Check if date column exists and rename it
        if "date" in df_copy.columns:
            df_copy = df_copy.rename(columns={"date": "timestamp"})
            return df_copy

        # Check if timestamp is in the index
        if df_copy.index.name == "timestamp" or isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy = df_copy.reset_index()
            if "index" in df_copy.columns:
                df_copy = df_copy.rename(columns={"index": "timestamp"})
            return df_copy

        # If no timestamp found, create one from the first available datetime column
        datetime_columns = [col for col in df_copy.columns if pd.api.types.is_datetime64_any_dtype(df_copy[col])]
        if datetime_columns:
            df_copy["timestamp"] = df_copy[datetime_columns[0]]
            return df_copy

        # Last resort: create a dummy timestamp (this should rarely happen)
        # logger.warning("No timestamp column found, creating dummy timestamp") # This line was commented out in the original file
        df_copy["timestamp"] = pd.Timestamp.now()

        return df_copy


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
