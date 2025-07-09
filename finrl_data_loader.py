"""Simple FinRL-based data loading utilities."""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from finrl.meta.data_processor import DataProcessor
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from trading_rl_agent.data.synthetic import generate_gbm_prices


def load_real_data(config_path: str) -> pd.DataFrame:
    """Load real market data using FinRL's DataProcessor."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")

    # Validate required fields
    required_fields = ["tickers", "start_date", "end_date"]
    for field in required_fields:
        if not cfg.get(field):
            raise ValueError(f"Missing required config field: {field}")

    try:
        dp = DataProcessor(data_source=cfg.get("data_source", "yahoofinance"))
        df = dp.download_data(
            ticker_list=cfg.get("tickers", []),
            start_date=cfg.get("start_date"),
            end_date=cfg.get("end_date"),
            time_interval=cfg.get("time_interval", "1D"),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to download data: {e}")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=cfg.get("tech_indicators", []),
    )
    df = fe.preprocess_data(df)
    output = cfg.get("output")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
    return df


def load_synthetic_data(config_path: str) -> pd.DataFrame:
    """Generate synthetic data and apply FinRL feature engineering."""
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        raise ValueError(f"Failed to load config file {config_path}: {e}")

    synth = cfg.get("synthetic", {})
    n_days = synth.get("n_days", 100)
    mu = synth.get("mu", 0.0002)
    sigma = synth.get("sigma", 0.01)
    n_symbols = synth.get("n_symbols", 1)

    # Validate parameters
    if not isinstance(n_days, int) or n_days <= 0:
        raise ValueError("n_days must be a positive integer")
    if not isinstance(n_symbols, int) or n_symbols <= 0:
        raise ValueError("n_symbols must be a positive integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    frames: List[pd.DataFrame] = []
    for i in range(n_symbols):
        df = generate_gbm_prices(n_days, mu=mu, sigma=sigma)
        df["symbol"] = f"SYN_{i+1}"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=cfg.get("tech_indicators", []),
    )
    df = fe.preprocess_data(df)
    output = cfg.get("output")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Load data via FinRL")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument(
        "--synthetic", action="store_true", help="Generate synthetic data"
    )
    args = parser.parse_args()
    if args.synthetic:
        df = load_synthetic_data(args.config)
    else:
        df = load_real_data(args.config)
    print(df.head())


if __name__ == "__main__":
    main()
