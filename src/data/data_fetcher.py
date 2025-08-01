# Data Fetcher

import pandas as pd
import yfinance as yf

symbols: dict[str, list[str]] = {
    "Stocks": [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "FB", "NVDA", "BRK-B", "JPM", "V",
        "JNJ", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "VZ", "NFLX", "INTC"
    ],
    "ETFs": [
        "SPY", "IVV", "VOO", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLF", "XLY",
        "XLC", "XLI", "XLB", "XLP", "XLC", "XLV", "XBI", "XLK", "XLU", "XTL"
    ],
    "Cryptocurrencies": [
        "BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "BCH-USD", "ADA-USD",
        "SOL-USD", "DOT-USD", "LINK-USD", "DOGE-USD", "MATIC-USD",
        "UNI-USD", "XLM-USD", "AVAX-USD", "ATOM-USD", "ALGO-USD", "TRX-USD",
        "ETC-USD", "FIL-USD", "AAVE-USD", "SUSHI-USD"
    ],
    "Indices": [
        "DXY", "VIX", "NDX", "RUT", "SPX"
    ],
    "Forex": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "USDMXN=X", "USDHKD=X", "USDCNY=X"
    ],
    "Commodities": [
        "CL=F", "GC=F", "SI=F", "HG=F", "NG=F"
    ]
}


def fetch_data(symbols: list[str], start_date: str, end_date: str, interval: str = '1h') -> pd.DataFrame:
    """
    Fetch historical stock data for given symbols.

    Parameters:
    - symbols: List of stock symbols to fetch data for.
    - start_date: Start date for the data in 'YYYY-MM-DD' format.
    - end_date: End date for the data in 'YYYY-MM-DD' format.
    - interval: Data interval (default is '1h').

    Returns:
    - DataFrame with stock symbols as keys and their historical data as values.
    """
    df = yf.download(tickers=symbols, start=start_date, end=end_date, interval=interval)  # type: ignore
    df_clean: pd.DataFrame = pd.DataFrame(df).dropna(axis=0, how='all')
    return df_clean
