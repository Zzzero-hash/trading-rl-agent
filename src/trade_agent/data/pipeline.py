import os
from collections.abc import Callable
from datetime import datetime, timedelta
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

        downloaded_files = []
        for symbol in symbols:
            symbol_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data

            # Enhance with historical sentiment if requested
            if include_sentiment:
                symbol_data = self._enhance_with_historical_sentiment(
                    symbol_data, symbol, start_date, end_date, sentiment_lookback_days
                )

            file_path = output_dir / f"{symbol}.csv"
            symbol_data.to_csv(file_path, index=False)
            downloaded_files.append(str(file_path))

        return downloaded_files

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

        # Collect sentiment for each day in the period
        sentiment_records = []

        current_date = start_dt
        while current_date <= end_dt:
            try:
                # Try to get sentiment for this specific date
                sentiment_data = sentiment_provider.fetch_sentiment(
                    symbol, days_back=sentiment_lookback_days
                )

                if sentiment_data:
                    # Aggregate sentiment for this date
                    daily_sentiment = self._aggregate_daily_sentiment(sentiment_data, current_date)
                    sentiment_records.append(daily_sentiment)
                else:
                    # Use market-derived sentiment as fallback
                    daily_sentiment = self._get_market_derived_sentiment(
                        market_data, symbol, current_date
                    )
                    sentiment_records.append(daily_sentiment)

            except Exception as e:
                # Fallback to market-derived sentiment
                daily_sentiment = self._get_market_derived_sentiment(
                    market_data, symbol, current_date
                )
                sentiment_records.append(daily_sentiment)

            current_date += timedelta(days=1)

        if sentiment_records:
            sentiment_df = pd.DataFrame(sentiment_records)
            return self._merge_market_sentiment(market_data, sentiment_df)

        return market_data

    def _aggregate_daily_sentiment(
        self,
        sentiment_data: list,
        target_date: datetime
    ) -> dict:
        """Aggregate sentiment data for a specific date."""

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
    ) -> dict:
        """Generate market-derived sentiment as fallback."""

        # Find market data for the target date
        market_data_copy = market_data.copy()
        market_data_copy["date"] = pd.to_datetime(market_data_copy["timestamp"]).dt.date
        target_date_only = target_date.date()

        # Get data for the target date and surrounding days
        date_data = market_data_copy[
            market_data_copy["date"] == target_date_only
        ]

        if date_data.empty:
            return {
                "timestamp": target_date,
                "sentiment_score": 0.0,
                "sentiment_magnitude": 0.0,
                "sentiment_sources": 1,
                "sentiment_direction": 0,
                "sentiment_source": "market_derived"
            }

        # Calculate market-based sentiment indicators
        row = date_data.iloc[0]

        # Price momentum (if we have historical data)
        if len(market_data_copy) > 1:
            # Get previous day's data
            prev_data = market_data_copy[
                market_data_copy["date"] < target_date_only
            ].tail(1)

            if not prev_data.empty:
                prev_close = prev_data.iloc[0]["close"]
                current_close = row["close"]

                # Calculate price change
                price_change = (current_close - prev_close) / prev_close

                # Volume analysis
                volume_ratio = 1.0
                if "volume" in row and "volume" in prev_data.columns:
                    avg_volume = market_data_copy["volume"].rolling(20).mean().iloc[-1]
                    volume_ratio = row["volume"] / avg_volume if avg_volume > 0 else 1.0

                # High-Low spread
                hl_spread = 0.02  # Default 2%
                if "high" in row and "low" in row:
                    hl_spread = (row["high"] - row["low"]) / row["close"]

                # Combine indicators into sentiment score
                momentum_score = price_change * 10  # Scale price change
                volume_score = (volume_ratio - 1) * 0.2  # Volume impact
                spread_score = -hl_spread * 10  # Tighter spreads = more positive

                combined_score = momentum_score + volume_score + spread_score
                sentiment_score = max(-1.0, min(1.0, combined_score))

                # Calculate confidence based on data quality
                confidence = 0.5 + 0.3 * (1 - abs(sentiment_score))

                return {
                    "timestamp": target_date,
                    "sentiment_score": sentiment_score,
                    "sentiment_magnitude": confidence,
                    "sentiment_sources": 1,
                    "sentiment_direction": 1 if sentiment_score > 0 else (-1 if sentiment_score < 0 else 0),
                    "sentiment_source": "market_derived"
                }

        # Fallback to neutral sentiment
        return {
            "timestamp": target_date,
            "sentiment_score": 0.0,
            "sentiment_magnitude": 0.5,
            "sentiment_sources": 1,
            "sentiment_direction": 0,
            "sentiment_source": "market_derived"
        }

    def _merge_market_sentiment(
        self,
        market_data: pd.DataFrame,
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge market data with sentiment data."""

        if sentiment_data.empty:
            # Add default sentiment columns
            market_data["sentiment_score"] = 0.0
            market_data["sentiment_magnitude"] = 0.0
            market_data["sentiment_sources"] = 0
            market_data["sentiment_direction"] = 0
            market_data["sentiment_source"] = "no_data"
            return market_data

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
