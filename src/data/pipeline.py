import os
import yaml
from pathlib import Path
import pandas as pd

from .historical import fetch_historical_data
from .features import generate_features
from .synthetic import fetch_synthetic_data
from .live import fetch_live_data


def load_cached_csvs(directory: str) -> pd.DataFrame:
    """Load all CSV files from a directory and concatenate them.

    Each resulting row gains a ``source`` column derived from the file name.

    Parameters
    ----------
    directory : str
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


def run_pipeline(config_path: str):
    """
    Run data pipeline to fetch historical OHLCV data for configured symbols.

    Returns a dict mapping symbol keys to DataFrames.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    start = cfg.get('start')
    end = cfg.get('end')
    timestep = cfg.get('timestep', 'day')
    coinbase_symbols = cfg.get('coinbase_perp_symbols', [])
    oanda_symbols = cfg.get('oanda_fx_symbols', [])
    output_dir = cfg.get('output_dir', 'data/raw')
    to_csv = cfg.get('to_csv', True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = {}

    # Fetch Coinbase perpetual futures data
    for symbol in coinbase_symbols:
        key = f"coinbase_{symbol}"
        df = fetch_historical_data(symbol, start, end, timestep)
        results[key] = df
        if to_csv:
            csv_path = Path(output_dir) / f"{key}.csv"
            df.to_csv(csv_path)

    # Fetch OANDA FX data
    for symbol in oanda_symbols:
        key = f"oanda_{symbol}"
        df = fetch_historical_data(symbol, start, end, timestep)
        results[key] = df
        if to_csv:
            csv_path = Path(output_dir) / f"{key}.csv"
            df.to_csv(csv_path)

    # Generate Synthetic data sources
    for symbol in cfg.get('synthetic_symbols', []):
        key = f"synthetic_{symbol}"
        df = fetch_synthetic_data(symbol, start, end, timestep)
        # Only apply feature generation to OHLCV-style data
        if 'close' in df.columns:
            df = generate_features(df)
        results[key] = df
        if to_csv:
            csv_path = Path(output_dir) / f"{key}.csv"
            df.to_csv(csv_path)

    # Fetch Live data sources
    for symbol in cfg.get('live_symbols', []):
        key = f"live_{symbol}"
        df = fetch_live_data(symbol, start, end, timestep)
        df = generate_features(df)
        results[key] = df
        if to_csv:
            csv_path = Path(output_dir) / f"{key}.csv"
            df.to_csv(csv_path)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fetch OHLCV data pipeline for Coinbase and OANDA')
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='src/configs/data/pipeline.yaml',
        help='Path to pipeline configuration YAML'
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config}")
    run_pipeline(args.config)
    print("Data pipeline completed.")
