"""
Symbol validation utilities for Yahoo Finance.

This module provides functions to validate symbols and filter out delisted
or unavailable symbols before attempting to download data.
"""

import logging
from typing import Any

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.getLogger(__name__).warning("yfinance not available for symbol validation")

logger = logging.getLogger(__name__)


def validate_symbols(symbols: list[str], min_data_points: int = 10) -> tuple[list[str], list[str]]:
    """
    Validate symbols against Yahoo Finance to filter out delisted/unavailable ones.

    Args:
        symbols: List of symbols to validate
        min_data_points: Minimum number of data points required to consider a symbol valid

    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available, returning all symbols as valid")
        return symbols, []

    valid_symbols = []
    invalid_symbols = []

    logger.info(f"Validating {len(symbols)} symbols...")

    for i, symbol in enumerate(symbols):
        try:
            # Test with a small date range to check if symbol has data
            ticker = yf.Ticker(symbol)

            # Try to get recent data (last 30 days)
            from datetime import datetime, timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            data = ticker.history(start=start_date, end=end_date, interval="1d")

            # Check if we have enough data points and the symbol is active
            if not data.empty and len(data) >= min_data_points:
                # Additional check: ensure we have price data
                if data["Close"].notna().sum() >= min_data_points:
                    valid_symbols.append(symbol)
                    logger.debug(f"✓ {symbol}: {len(data)} data points")
                else:
                    invalid_symbols.append(symbol)
                    logger.debug(f"✗ {symbol}: insufficient price data")
            else:
                invalid_symbols.append(symbol)
                logger.debug(f"✗ {symbol}: no data or insufficient data points")

        except Exception as e:
            invalid_symbols.append(symbol)
            logger.debug(f"✗ {symbol}: error - {e!s}")

        # Progress indicator for large lists
        if (i + 1) % 50 == 0:
            logger.info(f"Validated {i + 1}/{len(symbols)} symbols...")

    logger.info(f"Validation complete: {len(valid_symbols)} valid, {len(invalid_symbols)} invalid")

    if invalid_symbols:
        logger.info(f"Invalid symbols: {', '.join(invalid_symbols[:10])}{'...' if len(invalid_symbols) > 10 else ''}")

    return valid_symbols, invalid_symbols


def get_symbol_info(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """
    Get basic information about symbols from Yahoo Finance.

    Args:
        symbols: List of symbols to get info for

    Returns:
        Dictionary mapping symbols to their info
    """
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available for symbol info")
        return {}

    symbol_info = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract relevant information
            symbol_info[symbol] = {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown"),
                "quote_type": info.get("quoteType", "Unknown"),
                "regular_market_price": info.get("regularMarketPrice", 0),
                "regular_market_volume": info.get("regularMarketVolume", 0),
            }

        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
            symbol_info[symbol] = {
                "name": symbol,
                "sector": "Unknown",
                "industry": "Unknown",
                "market_cap": 0,
                "currency": "USD",
                "exchange": "Unknown",
                "quote_type": "Unknown",
                "regular_market_price": 0,
                "regular_market_volume": 0,
            }

    return symbol_info


def filter_symbols_by_criteria(
    symbols: list[str],
    min_market_cap: float = 0,
    sectors: list[str] | None = None,
    exchanges: list[str] | None = None,
    quote_types: list[str] | None = None
) -> list[str]:
    """
    Filter symbols based on various criteria.

    Args:
        symbols: List of symbols to filter
        min_market_cap: Minimum market cap in USD
        sectors: List of allowed sectors
        exchanges: List of allowed exchanges
        quote_types: List of allowed quote types

    Returns:
        Filtered list of symbols
    """
    if not symbols:
        return []

    # Get symbol info
    symbol_info = get_symbol_info(symbols)

    filtered_symbols = []

    for symbol, info in symbol_info.items():
        # Check market cap
        if info["market_cap"] < min_market_cap:
            continue

        # Check sector
        if sectors and info["sector"] not in sectors:
            continue

        # Check exchange
        if exchanges and info["exchange"] not in exchanges:
            continue

        # Check quote type
        if quote_types and info["quote_type"] not in quote_types:
            continue

        filtered_symbols.append(symbol)

    logger.info(f"Filtered {len(symbols)} symbols to {len(filtered_symbols)} based on criteria")

    return filtered_symbols


def get_popular_symbols() -> list[str]:
    """
    Get a curated list of popular, actively traded symbols.

    Returns:
        List of popular symbols
    """
    popular_symbols = [
        # Major US Stocks
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "JPM", "JNJ",
        "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE", "CRM", "NKE",
        "INTC", "AMD", "ORCL", "CSCO", "IBM", "QCOM", "TXN", "AVGO", "MU", "LRCX",
        "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC",
        "PFE", "ABBV", "TMO", "DHR", "BMY", "ABT", "LLY", "MRK", "AMGN", "GILD",
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "PSX", "VLO", "MPC",
        "KO", "PEP", "WMT", "COST", "TGT", "LOW", "SBUX", "MCD", "YUM", "CMCSA",

        # Major ETFs
        "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT",
        "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY",
        "XLB", "XLU", "VNQ", "IEMG", "EFA", "EEM", "ACWI", "VT", "BNDX", "EMB",

        # Major Indices
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
        "^BSESN", "^AXJO", "^TNX", "^TYX", "^IRX",

        # Major Forex Pairs
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X",
        "EURCHF=X", "GBPCHF=X", "AUDCHF=X", "CADCHF=X", "NZDCHF=X",

        # Major Cryptocurrencies
        "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD",
        "XRP-USD", "BNB-USD", "SOL-USD", "AVAX-USD", "MATIC-USD", "UNI-USD", "ATOM-USD",
        "NEAR-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD",

        # Major Commodities
        "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F",
        "LBS=F", "HE=F", "LE=F", "GF=F"
    ]

    return popular_symbols


def validate_and_filter_symbols(
    symbols: list[str] | None = None,
    use_popular: bool = True,
    min_data_points: int = 10,
    min_market_cap: float = 0,
    sectors: list[str] | None = None,
    exchanges: list[str] | None = None,
    quote_types: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """
    Comprehensive symbol validation and filtering.

    Args:
        symbols: List of symbols to validate (if None, uses popular symbols)
        use_popular: Whether to use popular symbols if no symbols provided
        min_data_points: Minimum data points required
        min_market_cap: Minimum market cap filter
        sectors: Allowed sectors filter
        exchanges: Allowed exchanges filter
        quote_types: Allowed quote types filter

    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    # Get symbols to validate
    if symbols is None:
        if use_popular:
            symbols = get_popular_symbols()
        else:
            return [], []

    # First validate symbols
    valid_symbols, invalid_symbols = validate_symbols(symbols, min_data_points)

    # Then apply additional filters
    if any([min_market_cap > 0, sectors, exchanges, quote_types]):
        valid_symbols = filter_symbols_by_criteria(
            valid_symbols, min_market_cap, sectors, exchanges, quote_types
        )

    return valid_symbols, invalid_symbols
