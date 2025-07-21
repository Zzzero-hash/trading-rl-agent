"""
Unified data loader for trading RL agent.

Provides a centralized interface for loading data from various sources
including Alpaca, Yahoo Finance, and other market data providers.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from .alpaca_integration import AlpacaConfig, AlpacaIntegration
from .data_standardizer import DataStandardizer
from .loaders.alphavantage_loader import load_alphavantage
from .loaders.ccxt_loader import load_ccxt
from .loaders.yfinance_loader import load_yfinance

logger = structlog.get_logger(__name__)


class DataLoader:
    """
    Unified data loader for multiple data sources.

    Provides a centralized interface for loading market data from various
    sources with automatic data standardization and caching.
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        cache_dir: str = "data/cache",
        enable_cache: bool = True,
    ):
        """
        Initialize the data loader.

        Args:
            config: Configuration dictionary for data sources
            cache_dir: Directory for caching data
            enable_cache: Whether to enable data caching
        """
        self.config = config or {}
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache

        # Create cache directory if it doesn't exist
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data loaders
        self.alpaca_loader: AlpacaIntegration | None = None
        if "alpaca" in self.config:
            alpaca_config = AlpacaConfig(**self.config["alpaca"])
            self.alpaca_loader = AlpacaIntegration(alpaca_config)

        # Initialize data standardizer
        self.standardizer = DataStandardizer()

        logger.info("DataLoader initialized", cache_dir=str(self.cache_dir))

    def load_historical_data(
        self,
        symbols: str | list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        interval: str = "1d",
        source: str = "alpaca",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load historical market data.

        Args:
            symbols: Stock symbols to load data for
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            source: Data source (alpaca, yfinance, ccxt, alphavantage)
            **kwargs: Additional arguments for specific loaders

        Returns:
            Standardized DataFrame with market data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        # Convert string dates to datetime objects
        start_dt = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        end_dt = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        # Check cache first
        cache_key = self._generate_cache_key(symbols, start_dt, end_dt, interval, source)
        if self.enable_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.info("Loaded data from cache", cache_key=cache_key)
                return cached_data

        # Load data from source
        try:
            if source == "alpaca":
                if self.alpaca_loader is None:
                    raise ValueError("Alpaca loader not initialized. Check configuration.")
                data = self.alpaca_loader.get_historical_data(symbols, start_date, end_date, interval, **kwargs)
            elif source == "yfinance":
                # Handle multiple symbols for yfinance
                all_data = []
                for symbol in symbols:
                    symbol_data = load_yfinance(
                        symbol,
                        start_dt.strftime("%Y-%m-%d"),
                        end_dt.strftime("%Y-%m-%d"),
                        interval,
                    )
                    symbol_data["symbol"] = symbol
                    all_data.append(symbol_data)
                data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            elif source == "ccxt":
                # Handle multiple symbols for ccxt
                all_data = []
                for symbol in symbols:
                    symbol_data = load_ccxt(
                        symbol,
                        start_dt.strftime("%Y-%m-%d"),
                        end_dt.strftime("%Y-%m-%d"),
                        interval,
                        **kwargs,
                    )
                    symbol_data["symbol"] = symbol
                    all_data.append(symbol_data)
                data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            elif source == "alphavantage":
                # Handle multiple symbols for alphavantage
                all_data = []
                for symbol in symbols:
                    symbol_data = load_alphavantage(
                        symbol,
                        start_dt.strftime("%Y-%m-%d"),
                        end_dt.strftime("%Y-%m-%d"),
                        interval,
                        **kwargs,
                    )
                    symbol_data["symbol"] = symbol
                    all_data.append(symbol_data)
                data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            else:
                raise ValueError(f"Unsupported data source: {source}")

            # Standardize data
            standardized_data = self.standardizer.transform(data)

            # Cache the data
            if self.enable_cache:
                self._save_to_cache(cache_key, standardized_data)

            logger.info(
                "Loaded historical data",
                symbols=symbols,
                source=source,
                shape=standardized_data.shape,
            )

            return standardized_data

        except Exception as exc:
            logger.exception(
                "Failed to load historical data",
                symbols=symbols,
                source=source,
                error=str(exc),
            )
            raise

    def load_live_data(
        self,
        symbols: str | list[str],
        source: str = "alpaca",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load live market data.

        Args:
            symbols: Stock symbols to load data for
            source: Data source (alpaca, yfinance, ccxt)
            **kwargs: Additional arguments for specific loaders

        Returns:
            DataFrame with live market data
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        try:
            if source == "alpaca":
                if self.alpaca_loader is None:
                    raise ValueError("Alpaca loader not initialized. Check configuration.")
                data = self.alpaca_loader.get_real_time_quotes(symbols, **kwargs)
                # Convert to DataFrame format
                data = pd.DataFrame(data)
            elif source == "yfinance":
                # For live data, we'll use the most recent historical data
                end_date = datetime.now()
                start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
                all_data = []
                for symbol in symbols:
                    symbol_data = load_yfinance(
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        "1m",
                    )
                    symbol_data["symbol"] = symbol
                    all_data.append(symbol_data)
                data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            elif source == "ccxt":
                # For live data, we'll use the most recent historical data
                end_date = datetime.now()
                start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
                all_data = []
                for symbol in symbols:
                    symbol_data = load_ccxt(
                        symbol,
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                        "1m",
                        **kwargs,
                    )
                    symbol_data["symbol"] = symbol
                    all_data.append(symbol_data)
                data = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
            else:
                raise ValueError(f"Unsupported data source: {source}")

            # Standardize data
            standardized_data = self.standardizer.transform(data)

            logger.info(
                "Loaded live data",
                symbols=symbols,
                source=source,
                shape=standardized_data.shape,
            )

            return standardized_data

        except Exception as exc:
            logger.exception(
                "Failed to load live data",
                symbols=symbols,
                source=source,
                error=str(exc),
            )
            raise

    async def load_data_async(
        self,
        symbols: str | list[str],
        start_date: str | datetime,
        end_date: str | datetime,
        interval: str = "1d",
        source: str = "alpaca",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load historical data asynchronously.

        Args:
            symbols: Stock symbols to load data for
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval
            source: Data source
            **kwargs: Additional arguments

        Returns:
            DataFrame with market data
        """
        # Run the synchronous method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.load_historical_data,
            symbols,
            start_date,
            end_date,
            interval,
            source,
            **kwargs,
        )

    def get_available_symbols(self, source: str = "alpaca") -> list[str]:
        """
        Get available symbols from a data source.

        Args:
            source: Data source to query

        Returns:
            List of available symbols
        """
        try:
            if source == "alpaca":
                if self.alpaca_loader is None:
                    return []
                # This would need to be implemented in AlpacaIntegration
                return []
            elif source == "yfinance":
                # Return some common symbols
                return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
            elif source == "ccxt":
                # Return some common crypto pairs
                return ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
            elif source == "alphavantage":
                # Return some common symbols
                return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
            else:
                raise ValueError(f"Unsupported data source: {source}")

        except Exception as exc:
            logger.exception(
                "Failed to get available symbols",
                source=source,
                error=str(exc),
            )
            return []

    def _generate_cache_key(
        self,
        symbols: list[str],
        start_date: datetime,
        end_date: datetime,
        interval: str,
        source: str,
    ) -> str:
        """Generate a cache key for the data request."""
        symbols_str = "_".join(sorted(symbols))
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")
        return f"{source}_{symbols_str}_{start_str}_{end_str}_{interval}.parquet"

    def _load_from_cache(self, cache_key: str) -> pd.DataFrame | None:
        """Load data from cache if available."""
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning("Failed to load from cache", cache_key=cache_key, error=str(e))
        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame) -> None:
        """Save data to cache."""
        try:
            cache_path = self.cache_dir / cache_key
            data.to_parquet(cache_path)
            logger.debug("Saved data to cache", cache_key=cache_key)
        except Exception as e:
            logger.warning("Failed to save to cache", cache_key=cache_key, error=str(e))

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.parquet"):
                cache_file.unlink()
            logger.info("Cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cached data."""
        if not self.cache_dir.exists():
            return {"cache_size": 0, "cache_files": 0}

        cache_files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "cache_size": total_size,
            "cache_files": len(cache_files),
            "cache_dir": str(self.cache_dir),
        }
