import os
from pathlib import Path

import pandas as pd
import ray
import ray.data as rdata
import yaml

from .features import generate_features
from .historical import fetch_historical_data
from .live import fetch_live_data
from .synthetic import fetch_synthetic_data


@ray.remote
def _fetch_data_remote(fetch_fn, **kwargs):
    """Execute a data fetch function as a Ray remote task."""
    return fetch_fn(**kwargs)


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
    """Run the data ingestion pipeline using Ray for parallelism.

    Parameters
    ----------
    config_path : str
        Path to a YAML configuration file defining symbols and output options.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping of ``source`` keys to DataFrames containing OHLCV data.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    start = cfg.get("start")
    end = cfg.get("end")
    timestep = cfg.get("timestep", "day")
    coinbase_symbols = cfg.get("coinbase_perp_symbols", [])
    oanda_symbols = cfg.get("oanda_fx_symbols", [])
    output_dir = cfg.get("output_dir", "data/raw")
    to_csv = cfg.get("to_csv", True)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    started_ray = False
    if not ray.is_initialized():
        ray.init(
            address=cfg.get("ray_address", os.getenv("RAY_ADDRESS")),
            log_to_driver=False,
        )
        started_ray = True

    tasks = {}

    for symbol in coinbase_symbols:
        key = f"coinbase_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_historical_data,
            symbol=symbol,
            start=start,
            end=end,
            timestep=timestep,
        )

    for symbol in oanda_symbols:
        key = f"oanda_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_historical_data,
            symbol=symbol,
            start=start,
            end=end,
            timestep=timestep,
        )

    freq_map = {"day": "D", "hour": "H", "minute": "T"}
    n_samples = 1
    if start and end:
        freq = freq_map.get(timestep, timestep)
        n_samples = len(pd.date_range(start=start, end=end, freq=freq))

    for symbol in cfg.get("synthetic_symbols", []):
        key = f"synthetic_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_synthetic_data,
            n_samples=n_samples,
            timeframe=timestep,
        )

    for symbol in cfg.get("live_symbols", []):
        key = f"live_{symbol}"
        tasks[key] = _fetch_data_remote.remote(
            fetch_live_data,
            symbol=symbol,
            start=start,
            end=end,
            timestep=timestep,
        )

    fetched = ray.get(list(tasks.values())) if tasks else []

    results = {}
    for key, df in zip(tasks.keys(), fetched):
        if key.startswith("synthetic_"):
            if "close" in df.columns:
                df = generate_features(df)
        if key.startswith("live_"):
            df = generate_features(df)

        results[key] = df
        if to_csv:
            csv_path = Path(output_dir) / f"{key}.csv"
            df.to_csv(csv_path)

    if started_ray:
        ray.shutdown()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch OHLCV data pipeline for Coinbase and OANDA"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="src/configs/data/pipeline.yaml",
        help="Path to pipeline configuration YAML",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config}")
    run_pipeline(args.config)
    print("Data pipeline completed.")
