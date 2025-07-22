"""
Market symbols definitions for comprehensive data collection.

This module provides centralized access to market symbols organized by asset type,
with validation and filtering capabilities.
"""


from rich.console import Console

# Initialize console for logging
console = Console()

# Comprehensive symbol list organized by asset type
COMPREHENSIVE_SYMBOLS = {
    "stocks": [
        # Major US Stocks (S&P 500 top components) - Current as of 2024
        "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "UNH", "JNJ",
        "JPM", "V", "PG", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE", "CRM",
        "NFLX", "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "DHR", "ACN", "LLY",
        "NKE", "TXN", "QCOM", "HON", "ORCL", "LOW", "UPS", "INTU", "SPGI", "GILD",
        "AMD", "ISRG", "TGT", "ADI", "PLD", "REGN", "MDLZ", "VRTX", "PANW", "KLAC",
        # Tech Giants
        "INTC", "CSCO", "IBM", "MU", "LRCX", "AMAT", "ASML",
        # Financial Sector
        "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB", "PNC", "COF",
        "TFC", "KEY", "HBAN", "RF", "ZION", "CMA", "FITB", "MTB",
        # Healthcare
        "PFE", "ABBV", "BMY", "MRK", "AMGN", "BIIB", "DXCM", "ALGN", "IDXX", "ILMN",
        # Consumer
        "WMT", "SBUX", "MCD", "YUM", "CMCSA", "UA", "LULU", "ROST", "TJX", "MAR", "HLT", "BKNG",
        # Energy
        "XOM", "CVX", "COP", "EOG", "SLB", "HAL", "BKR", "PSX", "VLO", "MPC",
        "OXY", "DVN", "FANG", "HES", "APA",
        # Additional Popular Stocks
        "UBER", "LYFT", "SNAP", "PINS", "ZM", "ROKU", "SPOT", "BYND", "PLTR", "SNOW",
        "CRWD", "ZS", "OKTA", "TEAM", "DOCU", "TDOC", "RBLX", "HOOD", "COIN", "RIVN",
        "LCID", "NIO", "XPEV", "LI", "BIDU", "JD", "BABA", "TCEHY", "PDD", "NTES",
        "BILI", "XNET", "ZTO", "TME", "VIPS"
    ],
    "etfs": [
        # ETFs (Major categories) - Current symbols
        "SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG", "BND", "TLT",
        "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLY",
        "XLB", "XLU", "VNQ", "IEMG", "EFA", "EEM", "ACWI", "VT", "BNDX", "EMB"
    ],
    "indices": [
        # Market Indices - Current symbols
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI",
        "^BSESN", "^AXJO", "^TNX", "^TYX", "^IRX"
    ],
    "forex": [
        # Forex (Major pairs) - Current symbols
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
        "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "NZDJPY=X",
        "EURCHF=X", "GBPCHF=X", "AUDCHF=X", "CADCHF=X", "NZDCHF=X"
    ],
    "crypto": [
        # Cryptocurrencies - Current symbols
        "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD", "LTC-USD", "BCH-USD",
        "XRP-USD", "BNB-USD", "SOL-USD", "AVAX-USD", "MATIC-USD", "UNI-USD", "ATOM-USD",
        "NEAR-USD", "ALGO-USD", "VET-USD", "ICP-USD", "FIL-USD"
    ],
    "commodities": [
        # Commodities - Current symbols
        "GC=F", "SI=F", "CL=F", "NG=F", "ZC=F", "ZS=F", "ZW=F", "KC=F", "CC=F", "CT=F",
        "LBS=F", "HE=F", "LE=F", "GF=F"
    ]
}

# Default symbols for quick access
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT"]


def get_symbols_by_type(asset_types: list[str] | None = None) -> dict[str, list[str]]:
    """
    Get symbols organized by asset type.

    Args:
        asset_types: List of asset types to include. If None, returns all types.

    Returns:
        Dictionary mapping asset types to lists of symbols
    """
    if asset_types is None:
        return COMPREHENSIVE_SYMBOLS.copy()

    return {asset_type: COMPREHENSIVE_SYMBOLS[asset_type]
            for asset_type in asset_types
            if asset_type in COMPREHENSIVE_SYMBOLS}


def get_all_symbols(asset_types: list[str] | None = None, deduplicate: bool = True) -> list[str]:
    """
    Get a flat list of all symbols, optionally filtered by asset type.

    Args:
        asset_types: List of asset types to include. If None, includes all types.
        deduplicate: Whether to remove duplicate symbols across categories.

    Returns:
        List of unique symbols
    """
    symbols_by_type = get_symbols_by_type(asset_types)

    if not deduplicate:
        # Just flatten without deduplication
        all_symbols = []
        for symbols in symbols_by_type.values():
            all_symbols.extend(symbols)
        return all_symbols

    # Deduplicate symbols within each category first
    for category_type in symbols_by_type:
        seen_in_category = set()
        deduplicated = []
        for symbol in symbols_by_type[category_type]:
            if symbol not in seen_in_category:
                seen_in_category.add(symbol)
                deduplicated.append(symbol)
        symbols_by_type[category_type] = deduplicated

    # Flatten all symbols
    all_symbols = []
    for symbols in symbols_by_type.values():
        all_symbols.extend(symbols)

    # Remove duplicates across all categories while preserving order
    seen = set()
    unique_symbols = []
    for symbol in all_symbols:
        if symbol not in seen:
            seen.add(symbol)
            unique_symbols.append(symbol)

    return unique_symbols


def validate_symbols(symbols: list[str], verbose: bool = True) -> tuple[list[str], list[str]]:
    """
    Validate symbols using yfinance.

    Args:
        symbols: List of symbols to validate
        verbose: Whether to print validation progress

    Returns:
        Tuple of (valid_symbols, invalid_symbols)
    """
    valid_symbols = []
    invalid_symbols = []

    if verbose:
        console.print("[yellow]Validating symbols...[/yellow]")

    try:
        import yfinance as yf
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if info and "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    valid_symbols.append(symbol)
                else:
                    invalid_symbols.append(symbol)
            except Exception:
                invalid_symbols.append(symbol)
    except ImportError:
        if verbose:
            console.print("[yellow]yfinance not available for validation, proceeding with all symbols[/yellow]")
        valid_symbols = symbols

    if verbose and invalid_symbols:
        console.print(f"[yellow]Removed {len(invalid_symbols)} invalid symbols: {', '.join(invalid_symbols[:10])}{'...' if len(invalid_symbols) > 10 else ''}[/yellow]")

    if verbose:
        console.print(f"[green]Proceeding with {len(valid_symbols)} valid symbols[/green]")

    return valid_symbols, invalid_symbols


def get_comprehensive_symbols(asset_types: list[str] | None = None, validate: bool = True, verbose: bool = True) -> str:
    """
    Get comprehensive market symbols as a comma-separated string.

    Args:
        asset_types: List of asset types to include. If None, includes all types.
        validate: Whether to validate symbols using yfinance.
        verbose: Whether to print validation progress.

    Returns:
        Comma-separated string of symbols
    """
    symbols = get_all_symbols(asset_types, deduplicate=True)

    if validate:
        valid_symbols, _ = validate_symbols(symbols, verbose=verbose)
        return ",".join(valid_symbols)

    return ",".join(symbols)


def get_symbols_list(asset_types: list[str] | None = None, validate: bool = True, verbose: bool = True) -> list[str]:
    """
    Get comprehensive market symbols as a list.

    Args:
        asset_types: List of asset types to include. If None, includes all types.
        validate: Whether to validate symbols using yfinance.
        verbose: Whether to print validation progress.

    Returns:
        List of symbols
    """
    symbols = get_all_symbols(asset_types, deduplicate=True)

    if validate:
        valid_symbols, _ = validate_symbols(symbols, verbose=verbose)
        return valid_symbols

    return symbols


def get_default_symbols() -> str:
    """
    Get default symbols as a comma-separated string.

    Returns:
        Comma-separated string of default symbols
    """
    return ",".join(DEFAULT_SYMBOLS)


def get_default_symbols_list() -> list[str]:
    """
    Get default symbols as a list.

    Returns:
        List of default symbols
    """
    return DEFAULT_SYMBOLS.copy()
