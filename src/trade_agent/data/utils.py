"""
Data utilities for consistent timezone handling and data processing.
"""


import pandas as pd


def normalize_timestamps(
    df: pd.DataFrame,
    timezone: str = "America/New_York",
    timestamp_column: str | None = None
) -> pd.DataFrame:
    """
    Normalize timestamps in a DataFrame to a consistent timezone format.

    Args:
        df: DataFrame containing timestamps
        timezone: Target timezone for normalization
        timestamp_column: Column name containing timestamps (if None, uses index)

    Returns:
        DataFrame with normalized timestamps
    """
    df_copy = df.copy()

    if timestamp_column is not None:
        # Normalize specific column
        df_copy[timestamp_column] = _normalize_timestamp_series(
            df_copy[timestamp_column], timezone
        )
    else:
        # Normalize index
        df_copy.index = _normalize_timestamp_series(df_copy.index, timezone)

    return df_copy


def _normalize_timestamp_series(
    timestamps: pd.Series | pd.DatetimeIndex,
    timezone: str = "America/New_York"
) -> pd.Series | pd.DatetimeIndex:
    """
    Normalize timestamp series to consistent timezone format.

    Args:
        timestamps: Series or DatetimeIndex of timestamps
        timezone: Target timezone for normalization

    Returns:
        Normalized timestamp series
    """
    try:
        # Convert to datetime if not already
        timestamps = pd.to_datetime(timestamps)

        # Check if timezone-aware
        if hasattr(timestamps, "tz") and timestamps.tz is not None:
            # Convert to target timezone
            return timestamps.tz_convert(timezone)
        elif hasattr(timestamps, "dt") and timestamps.dt.tz is not None:
            # Handle Series with timezone-aware timestamps
            return timestamps.dt.tz_convert(timezone)
        else:
            # Assume UTC if no timezone info, then convert to target
            if hasattr(timestamps, "tz_localize"):
                return timestamps.tz_localize("UTC").tz_convert(timezone)
            else:
                return timestamps.dt.tz_localize("UTC").dt.tz_convert(timezone)
    except (AttributeError, TypeError, ValueError):
        # Fallback: try to localize to UTC first, then convert
        try:
            timestamps = pd.to_datetime(timestamps)
            if hasattr(timestamps, "tz_localize"):
                return timestamps.tz_localize("UTC").tz_convert(timezone)
            else:
                return timestamps.dt.tz_localize("UTC").dt.tz_convert(timezone)
        except (AttributeError, TypeError, ValueError):
            # Last resort: create naive timestamps in target timezone
            timestamps = pd.to_datetime(timestamps)
            if hasattr(timestamps, "tz_localize"):
                return timestamps.tz_localize(timezone)
            else:
                return timestamps.dt.tz_localize(timezone)


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


def ensure_timezone_aware(
    timestamps: pd.Series | pd.DatetimeIndex,
    timezone: str | None = None
) -> pd.Series | pd.DatetimeIndex:
    """
    Ensure timestamps are timezone-aware, using configured timezone if not specified.

    Args:
        timestamps: Series or DatetimeIndex of timestamps
        timezone: Target timezone (defaults to configured market timezone)

    Returns:
        Timezone-aware timestamps
    """
    if timezone is None:
        timezone = get_market_timezone()

    return _normalize_timestamp_series(timestamps, timezone)


def convert_to_naive_timestamps(
    timestamps: pd.Series | pd.DatetimeIndex
) -> pd.Series | pd.DatetimeIndex:
    """
    Convert timezone-aware timestamps to naive timestamps (for compatibility).

    Args:
        timestamps: Series or DatetimeIndex of timestamps

    Returns:
        Naive timestamps
    """
    try:
        # Convert to datetime if not already
        timestamps = pd.to_datetime(timestamps)

        if hasattr(timestamps, "tz"):
            # Handle DatetimeIndex
            if timestamps.tz is not None:
                return timestamps.tz_localize(None)
            else:
                return timestamps
        else:
            # Handle Series
            if timestamps.dt.tz is not None:
                # For Series, we need to convert the entire series to naive
                return timestamps.dt.tz_localize(None)
            else:
                return timestamps
    except (AttributeError, TypeError, ValueError):
        # Already naive or conversion failed
        return timestamps
