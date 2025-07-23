"""
Market Calendar and Trading Hours Management

This module provides comprehensive handling of different market schedules:
- Traditional markets (stocks, ETFs, forex) with specific trading hours and holidays
- Crypto markets that trade 24/7
- Mixed portfolios with both asset types
"""

import logging
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)


class MarketType(Enum):
    """Market types with different trading schedules."""
    STOCK = "stock"
    ETF = "etf"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"


class TradingCalendar:
    """
    Comprehensive trading calendar that handles different market types.

    Supports:
    - Traditional markets with specific hours and holidays
    - Crypto markets trading 24/7
    - Mixed portfolios with both asset types
    """

    def __init__(self, timezone: str = "America/New_York"):
        """
        Initialize trading calendar.

        Args:
            timezone: Primary timezone for market operations
        """
        self.timezone = ZoneInfo(timezone)
        self.ny_timezone = ZoneInfo("America/New_York")

        # Market hours for traditional markets (NYSE/NASDAQ)
        self.market_open = time(9, 30)  # 9:30 AM ET
        self.market_close = time(16, 0)  # 4:00 PM ET

        # Extended hours (pre-market and after-hours)
        self.pre_market_open = time(4, 0)  # 4:00 AM ET
        self.after_market_close = time(20, 0)  # 8:00 PM ET

        # US Market Holidays (NYSE/NASDAQ)
        self.us_holidays = self._get_us_market_holidays()

        # Asset type classification
        self.asset_classification = self._get_asset_classification()

        logger.info(f"Trading calendar initialized for timezone: {timezone}")

    def _get_us_market_holidays(self) -> set[str]:
        """Get US market holidays for the current year."""
        # This is a simplified list - in production, you'd use a proper holiday calendar
        current_year = datetime.now().year
        holidays = {
            f"{current_year}-01-01",  # New Year's Day
            f"{current_year}-01-15",  # Martin Luther King Jr. Day (3rd Monday)
            f"{current_year}-02-19",  # Presidents' Day (3rd Monday)
            f"{current_year}-04-15",  # Good Friday (varies)
            f"{current_year}-05-27",  # Memorial Day (last Monday)
            f"{current_year}-06-19",  # Juneteenth
            f"{current_year}-07-04",  # Independence Day
            f"{current_year}-09-02",  # Labor Day (1st Monday)
            f"{current_year}-11-28",  # Thanksgiving (4th Thursday)
            f"{current_year}-12-25",  # Christmas Day
        }
        return holidays

    def _get_asset_classification(self) -> dict[str, MarketType]:
        """Classify assets by market type."""
        return {
            # Stocks
            "AAPL": MarketType.STOCK, "GOOGL": MarketType.STOCK, "MSFT": MarketType.STOCK,
            "TSLA": MarketType.STOCK, "AMZN": MarketType.STOCK, "NVDA": MarketType.STOCK,
            "META": MarketType.STOCK, "NFLX": MarketType.STOCK, "JPM": MarketType.STOCK,
            "V": MarketType.STOCK, "PG": MarketType.STOCK, "HD": MarketType.STOCK,

            # ETFs
            "SPY": MarketType.ETF, "QQQ": MarketType.ETF, "IWM": MarketType.ETF,
            "VTI": MarketType.ETF, "VOO": MarketType.ETF, "AGG": MarketType.ETF,
            "GLD": MarketType.ETF, "SLV": MarketType.ETF, "USO": MarketType.ETF,

            # Forex
            "EURUSD=X": MarketType.FOREX, "GBPUSD=X": MarketType.FOREX,
            "USDJPY=X": MarketType.FOREX, "USDCHF=X": MarketType.FOREX,
            "AUDUSD=X": MarketType.FOREX, "USDCAD=X": MarketType.FOREX,

            # Crypto
            "BTC-USD": MarketType.CRYPTO, "ETH-USD": MarketType.CRYPTO,
            "ADA-USD": MarketType.CRYPTO, "DOT-USD": MarketType.CRYPTO,
            "LINK-USD": MarketType.CRYPTO, "LTC-USD": MarketType.CRYPTO,
            "XRP-USD": MarketType.CRYPTO, "BNB-USD": MarketType.CRYPTO,

            # Indices
            "^GSPC": MarketType.INDEX, "^DJI": MarketType.INDEX,
            "^IXIC": MarketType.INDEX, "^RUT": MarketType.INDEX,
            "^VIX": MarketType.INDEX,

            # Commodities
            "GC=F": MarketType.COMMODITY, "SI=F": MarketType.COMMODITY,
            "CL=F": MarketType.COMMODITY, "NG=F": MarketType.COMMODITY,
        }

    def get_market_type(self, symbol: str) -> MarketType:
        """Get market type for a given symbol."""
        return self.asset_classification.get(symbol, MarketType.STOCK)

    def is_market_open(self, symbol: str, timestamp: datetime | None = None) -> bool:
        """
        Check if market is open for a given symbol and timestamp.

        Args:
            symbol: Trading symbol
            timestamp: Timestamp to check (defaults to current time)

        Returns:
            True if market is open, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.now(self.timezone)

        market_type = self.get_market_type(symbol)

        if market_type == MarketType.CRYPTO:
            return True  # Crypto trades 24/7

        # For traditional markets, check hours and holidays
        return self._is_traditional_market_open(timestamp)

    def _is_traditional_market_open(self, timestamp: datetime) -> bool:
        """Check if traditional market is open."""
        # Convert to NY timezone for market hours
        ny_time = timestamp.astimezone(self.ny_timezone)

        # Check if it's a weekend
        if ny_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        # Check if it's a holiday
        date_str = ny_time.strftime("%Y-%m-%d")
        if date_str in self.us_holidays:
            return False

        # Check market hours
        current_time = ny_time.time()
        return self.market_open <= current_time <= self.market_close

    def get_next_market_open(self, symbol: str, from_timestamp: datetime | None = None) -> datetime:
        """
        Get the next market open time for a symbol.

        Args:
            symbol: Trading symbol
            from_timestamp: Starting timestamp (defaults to current time)

        Returns:
            Next market open timestamp
        """
        if from_timestamp is None:
            from_timestamp = datetime.now(self.timezone)

        market_type = self.get_market_type(symbol)

        if market_type == MarketType.CRYPTO:
            return from_timestamp  # Crypto is always open

        return self._get_next_traditional_market_open(from_timestamp)

    def _get_next_traditional_market_open(self, from_timestamp: datetime) -> datetime:
        """Get next traditional market open time."""
        ny_time = from_timestamp.astimezone(self.ny_timezone)

        # If current time is before market open today
        if ny_time.time() < self.market_open:
            next_open = ny_time.replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0
            )
        else:
            # Move to next day
            next_open = (ny_time + timedelta(days=1)).replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0
            )

        # Skip weekends and holidays
        while not self._is_traditional_market_open(next_open):
            next_open += timedelta(days=1)

        return next_open.astimezone(self.timezone)

    def get_previous_market_close(self, symbol: str, from_timestamp: datetime | None = None) -> datetime:
        """
        Get the previous market close time for a symbol.

        Args:
            symbol: Trading symbol
            from_timestamp: Starting timestamp (defaults to current time)

        Returns:
            Previous market close timestamp
        """
        if from_timestamp is None:
            from_timestamp = datetime.now(self.timezone)

        market_type = self.get_market_type(symbol)

        if market_type == MarketType.CRYPTO:
            return from_timestamp  # Crypto is always open

        return self._get_previous_traditional_market_close(from_timestamp)

    def _get_previous_traditional_market_close(self, from_timestamp: datetime) -> datetime:
        """Get previous traditional market close time."""
        ny_time = from_timestamp.astimezone(self.ny_timezone)

        # If current time is after market close today
        if ny_time.time() > self.market_close:
            prev_close = ny_time.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                second=0,
                microsecond=0
            )
        else:
            # Move to previous day
            prev_close = (ny_time - timedelta(days=1)).replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                second=0,
                microsecond=0
            )

        # Skip weekends and holidays
        while not self._is_traditional_market_open(prev_close):
            prev_close -= timedelta(days=1)

        return prev_close.astimezone(self.timezone)

    def align_data_timestamps(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        alignment_strategy: str = "last_known_value"
    ) -> pd.DataFrame:
        """
        Align data timestamps for mixed portfolios.

        NEW BEHAVIOR: Retain all crypto candles and fill in traditional assets
        with last known values during weekends/holidays.

        Args:
            df: DataFrame with timestamp and symbol columns
            symbols: List of symbols in the portfolio
            alignment_strategy: Strategy for traditional asset alignment ('last_known_value', 'forward_fill', 'interpolate')

        Returns:
            DataFrame with aligned timestamps - all crypto data retained, traditional assets filled
        """
        if df.empty or "timestamp" not in df.columns or "symbol" not in df.columns:
            return df

        # Separate crypto and traditional assets
        crypto_symbols = [s for s in symbols if self.get_market_type(s) == MarketType.CRYPTO]
        traditional_symbols = [s for s in symbols if self.get_market_type(s) != MarketType.CRYPTO]

        if not crypto_symbols or not traditional_symbols:
            # No mixed portfolio, return as-is
            return df

        logger.info(f"Aligning timestamps for mixed portfolio: {len(crypto_symbols)} crypto, {len(traditional_symbols)} traditional")
        logger.info(f"Using alignment strategy: {alignment_strategy}")
        logger.info("NEW BEHAVIOR: Retaining all crypto candles, filling traditional assets")

        # NEW STRATEGY: Use crypto timestamps as base, fill traditional assets
        aligned_data = []

        # Get all unique timestamps from crypto data (these will be our base)
        crypto_data = df[df["symbol"].isin(crypto_symbols)].copy()
        crypto_data["timestamp"] = pd.to_datetime(crypto_data["timestamp"])
        all_timestamps = crypto_data["timestamp"].unique()
        all_timestamps = sorted(all_timestamps)

        logger.info(f"Using {len(all_timestamps)} crypto timestamps as base for alignment")

        for symbol in symbols:
            symbol_data = df[df["symbol"] == symbol].copy()
            symbol_data["timestamp"] = pd.to_datetime(symbol_data["timestamp"])

            if self.get_market_type(symbol) == MarketType.CRYPTO:
                # For crypto, keep all original data points
                symbol_data["data_source"] = "crypto_original"
                symbol_data["alignment_method"] = "none"
                aligned_data.append(symbol_data)
            else:
                # For traditional assets, fill in missing timestamps with last known values
                symbol_data = self._fill_traditional_asset_to_crypto_timestamps(
                    symbol_data, all_timestamps, alignment_strategy
                )
                aligned_data.append(symbol_data)

        # Combine all aligned data
        result = pd.concat(aligned_data, ignore_index=True)

        # Sort by timestamp and symbol
        result = result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        logger.info(f"Timestamp alignment complete. Original: {len(df)} rows, Aligned: {len(result)} rows")
        return result

    def _align_crypto_to_traditional_market(self, crypto_df: pd.DataFrame, strategy: str = "last_known_value") -> pd.DataFrame:
        """
        Align crypto data to traditional market hours using last known value strategy.

        Strategy:
        1. Create a traditional market calendar
        2. Use specified strategy to align crypto data to market timestamps
        3. Preserve OHLC integrity and handle volume appropriately
        4. Support multiple alignment strategies for different use cases
        """
        if crypto_df.empty:
            return crypto_df

        # Get the date range
        start_date = crypto_df["timestamp"].min()
        end_date = crypto_df["timestamp"].max()

        # Create traditional market calendar for this period
        market_calendar = self._create_market_calendar(start_date, end_date)

        # Convert to datetime if needed
        crypto_df = crypto_df.copy()
        crypto_df["timestamp"] = pd.to_datetime(crypto_df["timestamp"])
        crypto_df = crypto_df.sort_values("timestamp")

        if strategy == "last_known_value":
            return self._align_crypto_last_known_value(crypto_df, market_calendar)
        elif strategy == "forward_fill":
            return self._align_crypto_forward_fill(crypto_df, market_calendar)
        elif strategy == "interpolate":
            return self._align_crypto_interpolate(crypto_df, market_calendar)
        else:
            logger.warning(f"Unknown alignment strategy: {strategy}, using last_known_value")
            return self._align_crypto_last_known_value(crypto_df, market_calendar)

    def _align_crypto_last_known_value(self, crypto_df: pd.DataFrame, market_calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align crypto data using last known value strategy.

        This is the recommended approach for most use cases as it:
        - Preserves OHLC integrity
        - Avoids misleading forward-filling
        - Sets volume to zero during market closure
        - Maintains price continuity
        """
        aligned_data = []

        for market_time in market_calendar:
            # Find the last crypto data point before or at this market time
            mask = crypto_df["timestamp"] <= market_time
            if mask.any():
                last_crypto_data = crypto_df[mask].iloc[-1].copy()

                # Create aligned row with market timestamp
                aligned_row = {
                    "timestamp": market_time,
                    "open": last_crypto_data["open"],
                    "high": last_crypto_data["high"],
                    "low": last_crypto_data["low"],
                    "close": last_crypto_data["close"],
                    "volume": 0,  # Zero volume during market closure
                    "symbol": last_crypto_data["symbol"],
                    "data_source": "crypto_aligned",
                    "original_timestamp": last_crypto_data["timestamp"],
                    "alignment_method": "last_known_value"
                }

                aligned_data.append(aligned_row)

        aligned_df = pd.DataFrame(aligned_data)

        # Add metadata about the alignment
        if not aligned_df.empty:
            aligned_df["crypto_data_points_used"] = len(crypto_df)
            aligned_df["market_alignment_points"] = len(market_calendar)

        return aligned_df

    def _align_crypto_forward_fill(self, crypto_df: pd.DataFrame, market_calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align crypto data using forward-fill strategy.

        This is the original approach that may be useful for some applications:
        - Simple forward-filling of all OHLCV data
        - May be misleading for volume data
        - Preserves all original data points
        """
        # Set timestamp as index for reindexing
        crypto_df_indexed = crypto_df.set_index("timestamp")

        # Forward fill crypto data to match market calendar
        aligned_df = crypto_df_indexed.reindex(market_calendar, method="ffill")

        # Reset index to get timestamp back as column
        aligned_df = aligned_df.reset_index()
        aligned_df = aligned_df.rename(columns={"index": "timestamp"})

        # Add metadata
        aligned_df["data_source"] = "crypto_aligned"
        aligned_df["alignment_method"] = "forward_fill"

        return aligned_df

    def _align_crypto_interpolate(self, crypto_df: pd.DataFrame, market_calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align crypto data using interpolation strategy.

        This approach interpolates between crypto data points:
        - May be useful for high-frequency data
        - Interpolates OHLC values (use with caution)
        - Sets volume to zero during market closure
        """
        aligned_data = []

        for market_time in market_calendar:
            # Find crypto data points before and after this market time
            before_mask = crypto_df["timestamp"] <= market_time
            after_mask = crypto_df["timestamp"] > market_time

            if before_mask.any() and after_mask.any():
                # Interpolate between two crypto data points
                before_data = crypto_df[before_mask].iloc[-1]
                after_data = crypto_df[after_mask].iloc[0]

                # Calculate interpolation factor
                time_diff = (after_data["timestamp"] - before_data["timestamp"]).total_seconds()
                market_diff = (market_time - before_data["timestamp"]).total_seconds()
                factor = market_diff / time_diff if time_diff > 0 else 0

                # Interpolate OHLC values
                aligned_row = {
                    "timestamp": market_time,
                    "open": before_data["open"] + (after_data["open"] - before_data["open"]) * factor,
                    "high": before_data["high"] + (after_data["high"] - before_data["high"]) * factor,
                    "low": before_data["low"] + (after_data["low"] - before_data["low"]) * factor,
                    "close": before_data["close"] + (after_data["close"] - before_data["close"]) * factor,
                    "volume": 0,  # Zero volume during market closure
                    "symbol": before_data["symbol"],
                    "data_source": "crypto_aligned",
                    "original_timestamp": before_data["timestamp"],
                    "alignment_method": "interpolate"
                }

                aligned_data.append(aligned_row)
            elif before_mask.any():
                # Use last known value if no future data
                last_crypto_data = crypto_df[before_mask].iloc[-1].copy()
                aligned_row = {
                    "timestamp": market_time,
                    "open": last_crypto_data["open"],
                    "high": last_crypto_data["high"],
                    "low": last_crypto_data["low"],
                    "close": last_crypto_data["close"],
                    "volume": 0,
                    "symbol": last_crypto_data["symbol"],
                    "data_source": "crypto_aligned",
                    "original_timestamp": last_crypto_data["timestamp"],
                    "alignment_method": "interpolate"
                }
                aligned_data.append(aligned_row)

        aligned_df = pd.DataFrame(aligned_data)

        # Add metadata
        if not aligned_df.empty:
            aligned_df["crypto_data_points_used"] = len(crypto_df)
            aligned_df["market_alignment_points"] = len(market_calendar)

        return aligned_df

    def _filter_to_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter traditional market data to market hours only."""
        if df.empty:
            return df

        # Create mask for market hours
        df_copy = df.copy()
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"])

        # Convert to NY timezone for market hours check
        df_copy["ny_time"] = df_copy["timestamp"].dt.tz_localize(self.timezone).dt.tz_convert(self.ny_timezone)

        # Check market hours
        market_open_mask = df_copy["ny_time"].dt.time >= self.market_open
        market_close_mask = df_copy["ny_time"].dt.time <= self.market_close

        # Check weekdays
        weekday_mask = df_copy["ny_time"].dt.weekday < 5

        # Check holidays
        date_strs = df_copy["ny_time"].dt.strftime("%Y-%m-%d")
        holiday_mask = ~date_strs.isin(self.us_holidays)

        # Combine all masks
        market_hours_mask = market_open_mask & market_close_mask & weekday_mask & holiday_mask

        filtered_df = df_copy[market_hours_mask].copy()

        # Clean up temporary columns
        filtered_df = filtered_df.drop(columns=["ny_time"])
        filtered_df["data_source"] = "traditional_market"

        return filtered_df

    def _fill_traditional_asset_to_crypto_timestamps(
        self,
        traditional_df: pd.DataFrame,
        crypto_timestamps: list[datetime],
        strategy: str = "last_known_value"
    ) -> pd.DataFrame:
        """
        Fill traditional asset data to match crypto timestamps using last known values.

        This method ensures traditional assets have values for all crypto timestamps,
        including weekends and holidays, using the last known market values.

        Args:
            traditional_df: DataFrame with traditional asset data
            crypto_timestamps: List of all crypto timestamps to align to
            strategy: Strategy for filling missing values ('last_known_value', 'forward_fill', 'interpolate')

        Returns:
            DataFrame with traditional asset data filled to crypto timestamps
        """
        if traditional_df.empty:
            return traditional_df

        # Sort traditional data by timestamp
        traditional_df = traditional_df.sort_values("timestamp").reset_index(drop=True)

        filled_data = []
        last_known_values = None

        # Sort traditional data by timestamp for efficient lookup
        traditional_df_sorted = traditional_df.sort_values("timestamp")

        # Set to keep track of symbols for which data is not found
        no_data_warning_issued = False

        for crypto_timestamp in crypto_timestamps:
            # Find the last known traditional data point at or before the crypto timestamp
            relevant_data = traditional_df_sorted[traditional_df_sorted["timestamp"] <= crypto_timestamp]

            if not relevant_data.empty:
                last_known_values = relevant_data.iloc[-1]

                if last_known_values["timestamp"] == crypto_timestamp:
                    # Exact match, use this data directly
                    matched_row = last_known_values.copy()
                    matched_row["data_source"] = "traditional_aligned"
                    matched_row["alignment_method"] = strategy
                    matched_row["original_timestamp"] = matched_row["timestamp"]
                    filled_data.append(matched_row)
                else:
                    # Traditional data exists but not at this exact timestamp
                    # Use last known values (forward fill)
                    if last_known_values is not None:
                        filled_row = last_known_values.copy()
                        filled_row["timestamp"] = crypto_timestamp
                        filled_row["data_source"] = "traditional_filled"
                        filled_row["alignment_method"] = strategy
                        filled_row["original_timestamp"] = last_known_values["timestamp"]
                        filled_data.append(filled_row)
                    else:
                        # No previous data available
                        if not no_data_warning_issued:
                            logger.warning(
                                f"No previous data available for {traditional_df['symbol'].iloc[0]} at {crypto_timestamp}"
                            )
                            no_data_warning_issued = True
            else:
                # No traditional data available at or before this timestamp
                if last_known_values is not None:
                    # Use last known values
                    filled_row = last_known_values.copy()
                    filled_row["timestamp"] = crypto_timestamp
                    filled_row["data_source"] = "traditional_filled"
                    filled_row["alignment_method"] = strategy
                    filled_row["original_timestamp"] = last_known_values["timestamp"]
                    filled_data.append(filled_row)
                else:
                    # No data available at all
                    if not no_data_warning_issued:
                        logger.warning(f"No data available for {traditional_df['symbol'].iloc[0]} at {crypto_timestamp}")
                        no_data_warning_issued = True

        filled_df = pd.DataFrame(filled_data)

        # Add metadata about the filling process
        if not filled_df.empty:
            original_count = len(traditional_df)
            filled_count = len(filled_df)
            filled_df["original_data_points"] = original_count
            filled_df["crypto_alignment_points"] = len(crypto_timestamps)
            filled_df["filling_ratio"] = filled_count / len(crypto_timestamps)

            logger.info(f"Traditional asset {traditional_df['symbol'].iloc[0]}: "
                       f"{original_count} original -> {filled_count} aligned points "
                       f"({filled_count/len(crypto_timestamps)*100:.1f}% coverage)")

        return filled_df

    def _create_market_calendar(self, start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
        """Create a traditional market calendar for the given date range."""
        # Generate all business days in the range
        business_days = pd.bdate_range(start=start_date, end=end_date, freq="B")

        # Filter out holidays
        business_days = business_days[~business_days.strftime("%Y-%m-%d").isin(self.us_holidays)]

        # Create market hours for each business day
        market_times = []
        for day in business_days:
            # Market open time
            market_open = day.replace(
                hour=self.market_open.hour,
                minute=self.market_open.minute,
                second=0,
                microsecond=0
            )
            market_times.append(market_open)

            # Market close time
            market_close = day.replace(
                hour=self.market_close.hour,
                minute=self.market_close.minute,
                second=0,
                microsecond=0
            )
            market_times.append(market_close)

        return pd.DatetimeIndex(market_times)

    def get_market_hours_info(self, symbol: str) -> dict[str, Any]:
        """Get market hours information for a symbol."""
        market_type = self.get_market_type(symbol)

        info = {
            "symbol": symbol,
            "market_type": market_type.value,
            "trades_24_7": market_type == MarketType.CRYPTO,
            "timezone": "America/New_York" if market_type != MarketType.CRYPTO else "UTC",
        }

        if market_type != MarketType.CRYPTO:
            info.update({
                "market_open": self.market_open.strftime("%H:%M"),
                "market_close": self.market_close.strftime("%H:%M"),
                "pre_market_open": self.pre_market_open.strftime("%H:%M"),
                "after_market_close": self.after_market_close.strftime("%H:%M"),
                "holidays": str(list(self.us_holidays)),
            })

        return info


# Global calendar instance
_calendar = None


def get_market_timezone() -> str:
    """
    Get the configured market timezone from unified configuration.

    Returns:
        Market timezone string (defaults to "America/New_York")
    """
    try:
        from src.trade_agent.core.unified_config import load_config
        config = load_config()
        return config.data.market_timezone
    except (ImportError, AttributeError, Exception):
        # Fallback to default timezone
        return "America/New_York"


def get_trading_calendar(timezone: str | None = None) -> TradingCalendar:
    """
    Get a trading calendar instance with the specified timezone.

    Args:
        timezone: Timezone for the calendar (defaults to configured market timezone)

    Returns:
        TradingCalendar instance
    """
    if timezone is None:
        timezone = get_market_timezone()

    return TradingCalendar(timezone=timezone)


def classify_portfolio_assets(symbols: list[str]) -> dict[str, Any]:
    """
    Classify portfolio assets by market type.

    Args:
        symbols: List of trading symbols

    Returns:
        Dictionary mapping market types to lists of symbols
    """
    calendar = get_trading_calendar()

    classification: dict[str, Any] = {
        "crypto": [],
        "traditional": [],
        "mixed": False
    }

    for symbol in symbols:
        market_type = calendar.get_market_type(symbol)
        if market_type == MarketType.CRYPTO:
            classification["crypto"].append(symbol)
        else:
            classification["traditional"].append(symbol)

    classification["mixed"] = bool(classification["crypto"] and classification["traditional"])

    return classification
