"""
Professional Data Feeds Integration

Industry-grade data providers for production trading systems.
Supports Alpaca, Yahoo Finance, and extensible to Bloomberg/Refinitiv.
"""

from datetime import datetime, timedelta
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import alpaca_trade_api as tradeapi

    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logging.warning(
        "Alpaca Trade API not available. Install with: pip install alpaca-trade-api"
    )

try:
    from finrl.finrl_meta.data_processors.processor_alpaca import AlpacaProcessor

    FINRL_ALPACA_AVAILABLE = True
except ImportError:
    FINRL_ALPACA_AVAILABLE = False
    logging.warning("FinRL Alpaca processor not available")

import yfinance as yf

from src.data.features import generate_features

logger = logging.getLogger(__name__)


class ProfessionalDataProvider:
    """
    Industry-grade data provider with multiple professional feeds.

    Supports:
    - Alpaca Markets (commission-free trading data)
    - Yahoo Finance (fallback for development)
    - Extensible to Bloomberg, Refinitiv, Interactive Brokers
    """

    def __init__(self, provider: str = "alpaca", **kwargs):
        """
        Initialize professional data provider.

        Args:
            provider: Data provider ('alpaca', 'yahoo', 'bloomberg')
            **kwargs: Provider-specific configuration
        """
        self.provider = provider
        self.config = kwargs

        if provider == "alpaca":
            self._init_alpaca()
        elif provider == "yahoo":
            self._init_yahoo()
        elif provider == "bloomberg":
            self._init_bloomberg()
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _init_alpaca(self):
        """Initialize Alpaca Markets data feed."""
        if not ALPACA_AVAILABLE:
            raise ImportError("Alpaca Trade API required: pip install alpaca-trade-api")

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not api_key or not secret_key:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY"
            )

        self.alpaca_api = tradeapi.REST(api_key, secret_key, base_url, api_version="v2")

        # Initialize FinRL processor if available
        if FINRL_ALPACA_AVAILABLE:
            self.finrl_processor = AlpacaProcessor(
                API_KEY=api_key, API_SECRET=secret_key, API_BASE_URL=base_url
            )

        logger.info("Alpaca Markets data provider initialized")

    def _init_yahoo(self):
        """Initialize Yahoo Finance data feed (development/fallback)."""
        logger.info("Yahoo Finance data provider initialized (development mode)")

    def _init_bloomberg(self):
        """Initialize Bloomberg data feed (enterprise)."""
        # Placeholder for Bloomberg API integration
        # Requires Bloomberg Terminal and blpapi library
        logger.info("Bloomberg data provider initialized (placeholder)")
        raise NotImplementedError("Bloomberg integration requires enterprise license")

    def get_market_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1Day",
        include_features: bool = True,
    ) -> pd.DataFrame:
        """
        Get professional market data with optional feature engineering.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data frequency ('1Min', '5Min', '1Hour', '1Day')
            include_features: Whether to generate technical features

        Returns:
            DataFrame with OHLCV data and optional technical features
        """
        if self.provider == "alpaca":
            data = self._get_alpaca_data(symbols, start_date, end_date, timeframe)
        elif self.provider == "yahoo":
            data = self._get_yahoo_data(symbols, start_date, end_date)
        else:
            raise NotImplementedError(
                f"Data retrieval not implemented for {self.provider}"
            )

        if include_features and not data.empty:
            logger.info("Generating technical features...")
            data = self._add_technical_features(data)

        return data

    def _get_alpaca_data(
        self, symbols: list[str], start_date: str, end_date: str, timeframe: str
    ) -> pd.DataFrame:
        """Get data from Alpaca Markets."""
        try:
            # Use FinRL processor if available (recommended)
            if FINRL_ALPACA_AVAILABLE:
                return self.finrl_processor.download_data(
                    ticker_list=symbols,
                    start_date=start_date,
                    end_date=end_date,
                    time_interval=timeframe,
                )

            # Fallback to direct Alpaca API
            all_data = []

            for symbol in symbols:
                try:
                    bars = self.alpaca_api.get_bars(
                        symbol,
                        timeframe,
                        start=start_date,
                        end=end_date,
                        adjustment="raw",
                    ).df

                    if not bars.empty:
                        bars["symbol"] = symbol
                        bars.reset_index(inplace=True)
                        bars.rename(
                            columns={"timestamp": "date", "volume": "volume"},
                            inplace=True,
                        )

                        all_data.append(bars)

                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data["date"] = pd.to_datetime(combined_data["date"])
                return combined_data
            else:
                logger.error("No data retrieved from Alpaca")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Alpaca data retrieval error: {e}")
            return pd.DataFrame()

    def _get_yahoo_data(
        self, symbols: list[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get data from Yahoo Finance (fallback/development)."""
        try:
            all_data = []

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(start=start_date, end=end_date)

                    if not data.empty:
                        data.reset_index(inplace=True)
                        data["symbol"] = symbol
                        data.rename(
                            columns={
                                "Date": "date",
                                "Open": "open",
                                "High": "high",
                                "Low": "low",
                                "Close": "close",
                                "Volume": "volume",
                            },
                            inplace=True,
                        )

                        # Select relevant columns
                        data = data[
                            ["date", "symbol", "open", "high", "low", "close", "volume"]
                        ]
                        all_data.append(data)

                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
                    continue

            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                combined_data["date"] = pd.to_datetime(combined_data["date"])
                return combined_data.sort_values(["symbol", "date"]).reset_index(
                    drop=True
                )
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Yahoo Finance data retrieval error: {e}")
            return pd.DataFrame()

    def _add_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features to market data."""
        try:
            enhanced_data = []

            for symbol in data["symbol"].unique():
                symbol_data = data[data["symbol"] == symbol].copy()

                # Generate features using our existing pipeline
                symbol_data = generate_features(symbol_data)
                enhanced_data.append(symbol_data)

            return pd.concat(enhanced_data, ignore_index=True)

        except Exception as e:
            logger.error(f"Feature generation error: {e}")
            return data

    def get_real_time_data(self, symbols: list[str]) -> dict[str, Any]:
        """
        Get real-time market data for live trading.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dictionary with real-time market data
        """
        if self.provider == "alpaca":
            return self._get_alpaca_real_time(symbols)
        else:
            raise NotImplementedError(
                f"Real-time data not available for {self.provider}"
            )

    def _get_alpaca_real_time(self, symbols: list[str]) -> dict[str, Any]:
        """Get real-time data from Alpaca."""
        try:
            quotes = {}
            for symbol in symbols:
                quote = self.alpaca_api.get_latest_quote(symbol)
                quotes[symbol] = {
                    "bid_price": float(quote.bid_price),
                    "ask_price": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp,
                }
            return quotes

        except Exception as e:
            logger.error(f"Real-time data error: {e}")
            return {}

    def validate_connection(self) -> bool:
        """Validate connection to data provider."""
        try:
            if self.provider == "alpaca":
                account = self.alpaca_api.get_account()
                logger.info(f"Alpaca connection validated. Account: {account.id}")
                return True
            elif self.provider == "yahoo":
                # Test with a simple query
                test_data = yf.download("AAPL", period="1d")
                return not test_data.empty
            else:
                return False

        except Exception as e:
            logger.error(f"Connection validation failed: {e}")
            return False


def main():
    """Command-line interface for data download."""
    import argparse

    parser = argparse.ArgumentParser(description="Download professional market data")
    parser.add_argument(
        "--symbols", required=True, help="Comma-separated symbols (e.g., AAPL,GOOGL)"
    )
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument(
        "--end",
        help="End date (YYYY-MM-DD)",
        default=datetime.now().strftime("%Y-%m-%d"),
    )
    parser.add_argument(
        "--provider", choices=["alpaca", "yahoo"], default="yahoo", help="Data provider"
    )
    parser.add_argument("--output", help="Output CSV file", default="market_data.csv")
    parser.add_argument(
        "--features", action="store_true", help="Include technical features"
    )

    args = parser.parse_args()

    # Initialize provider
    provider = ProfessionalDataProvider(args.provider)

    # Validate connection
    if not provider.validate_connection():
        logger.error("Failed to connect to data provider")
        return

    # Download data
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    logger.info(f"Downloading data for {symbols} from {args.start} to {args.end}")

    data = provider.get_market_data(
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        include_features=args.features,
    )

    if not data.empty:
        data.to_csv(args.output, index=False)
        logger.info(f"Data saved to {args.output}")
        logger.info(f"Shape: {data.shape}")
        logger.info(f"Columns: {list(data.columns)}")
    else:
        logger.error("No data retrieved")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
