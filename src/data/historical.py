import os
import pandas as pd
from polygon import RESTClient

def fetch_historical_data(symbol: str,
                          start: str,
                          end: str,
                          timestep: str = "day"
                          ) -> pd.DataFrame:
    """
    Pulls aggregated bars from polygon.io
    :param symbol: The stock symbol to fetch data for
    :param start: The start date for the data in YYYY-MM-DD format
    :param end: The end date for the data in YYYY-MM-DD format
    :param timestep: The time step for the data (e.g. "day", "hour", "minute")
    :return: A pandas DataFrame containing the historical data
    """
    # batch pagination for large datasets
    all_records = []
    limit = 50000
    current_from = start
    API_KEY = os.getenv("POLYGON_API_KEY")
    client = RESTClient(API_KEY)
    while True:
        resp = client.get_aggs(
            symbol.upper(),
            1,                # multiplier
            timestep,         # "day", "minute", etc.
            _from=current_from,
            to=end,
            limit=limit
        )
        bars = resp.results
        if not bars:
            break
        for bar in bars:
            all_records.append({
                "timestamp": pd.to_datetime(bar.timestamp, unit="ms"),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })
        if len(bars) < limit:
            break
        # advance start just after last timestamp
        last_ts = bars[-1].timestamp
        current_from = (pd.to_datetime(last_ts, unit="ms") + pd.Timedelta(milliseconds=1)).strftime("%Y-%m-%d")
    df = pd.DataFrame(all_records).set_index("timestamp")
    return df