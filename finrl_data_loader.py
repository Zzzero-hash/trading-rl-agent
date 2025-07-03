"""Simple FinRL-based data loading utilities."""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import yaml

from finrl.meta.data_processor import DataProcessor
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from src.data.synthetic import generate_gbm_prices


def load_real_data(config_path: str) -> pd.DataFrame:
    """Load real market data using FinRL's DataProcessor."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    dp = DataProcessor(data_source=cfg.get("data_source", "yahoofinance"))
    df = dp.download_data(
        ticker_list=cfg.get("tickers", []),
        start_date=cfg.get("start_date"),
        end_date=cfg.get("end_date"),
        time_interval=cfg.get("time_interval", "1D"),
    )
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
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    synth = cfg.get("synthetic", {})
    n_days = synth.get("n_days", 100)
    mu = synth.get("mu", 0.0002)
    sigma = synth.get("sigma", 0.01)
    n_symbols = synth.get("n_symbols", 1)
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
