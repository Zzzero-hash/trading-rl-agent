import pandas as pd
import yfinance as yf

# use yfinance for free historical data
client = yf

def fetch_historical_data(symbol: str, start: str, end: str, timestep: str = "day") -> pd.DataFrame:
    """
    Fetch historical data using yfinance.
    :param symbol: Stock ticker symbol
    :param start: Start date (YYYY-MM-DD)
    :param end: End date (YYYY-MM-DD)
    :param timestep: Interval (e.g. 'day', '1m', '1h')
    :return: DataFrame with open/high/low/close/volume indexed by timestamp
    """
    # map human-friendly timestep to yfinance interval
    interval_map = {
        "day": "1d",
        "hour": "1h",
        "minute": "1m"
    }
    interval = interval_map.get(timestep, timestep)
    # download data
    df = client.Ticker(symbol).history(start=start, end=end, interval=interval)
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    # select and rename columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    # ensure timestamp index
    df.index.name = "timestamp"
    return df