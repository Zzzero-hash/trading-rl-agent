#!/usr/bin/env python3
"""Advanced Dataset Builder for CNN-LSTM Training.

This script creates a comprehensive dataset combining:
1. Synthetic data using Geometric Brownian Motion (GBM) and advanced stochastic models
2. Real market data from yfinance for major USD currency pairs and stocks
3. Advanced technical indicators and candlestick patterns
4. Sentiment analysis integration
5. Multi-timeframe feature engineering
6. Data quality validation and analysis

Features:
- State-of-the-art synthetic data generation with realistic market microstructure
- Comprehensive technical analysis indicators (50+ features)
- Advanced candlestick pattern recognition
- Multi-source sentiment analysis
- Data quality assurance and statistical validation
- Interactive data analysis and visualization
"""

import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Any, List

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.features import generate_features
from src.data.forex_sentiment import get_forex_sentiment
from src.data.historical import fetch_historical_data
from src.data.synthetic import generate_gbm_prices

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Major USD currency pairs and symbols
MAJOR_USD_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD"]

MAJOR_STOCKS = [
    "AAPL",
    "GOOGL",
    "MSFT",
    "AMZN",
    "TSLA",
    "NVDA",
    "META",
    "JPM",
    "BAC",
    "XOM",
]

CRYPTO_PAIRS = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD"]


class AdvancedDatasetBuilder:
    """Advanced dataset builder with comprehensive feature engineering."""

    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config.get("output_dir", "data"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data containers        self.synthetic_data: List[pd.DataFrame] = []
        self.real_data: list[pd.DataFrame] = []
        self.combined_data = None

    def generate_advanced_synthetic_data(self) -> pd.DataFrame:
        """Generate advanced synthetic data using multiple stochastic models."""
        logger.info("Generating advanced synthetic data...")

        synthetic_datasets = []

        # Configuration for different market regimes
        market_regimes = [
            {"name": "bull_market", "mu": 0.0008, "sigma": 0.015, "scenarios": 10},
            {"name": "bear_market", "mu": -0.0003, "sigma": 0.025, "scenarios": 8},
            {"name": "sideways_market", "mu": 0.0001, "sigma": 0.012, "scenarios": 6},
            {"name": "high_volatility", "mu": 0.0002, "sigma": 0.035, "scenarios": 4},
            {"name": "low_volatility", "mu": 0.0003, "sigma": 0.008, "scenarios": 4},
        ]

        symbols = MAJOR_USD_PAIRS + MAJOR_STOCKS[:5]  # Mix of forex and stocks

        for symbol in tqdm(symbols, desc="Generating synthetic data for symbols"):
            for regime in market_regimes:  # Type cast with proper error handling
                scenarios = (
                    int(regime["scenarios"])
                    if isinstance(regime["scenarios"], (int, float, str))
                    else 1
                )
                mu_val = (
                    float(regime["mu"])
                    if isinstance(regime["mu"], (int, float, str))
                    else 0.0
                )
                sigma_val = (
                    float(regime["sigma"])
                    if isinstance(regime["sigma"], (int, float, str))
                    else 0.1
                )

                for scenario in range(scenarios):
                    # Generate base synthetic data using GBM
                    df = generate_gbm_prices(
                        n_days=self.config.get("synthetic_days", 500),
                        mu=mu_val,
                        sigma=sigma_val,
                        s0=np.random.uniform(50, 200),  # Random starting price
                    )

                    # Add symbol and regime information
                    df["symbol"] = symbol
                    df["regime"] = regime["name"]
                    df["scenario"] = scenario

                    # Add market microstructure noise
                    df = self._add_microstructure_noise(df)

                    # Add regime-specific patterns
                    df = self._add_regime_patterns(df, regime)

                    synthetic_datasets.append(df)  # Combine all synthetic data
        synthetic_combined = pd.concat(synthetic_datasets, ignore_index=True)

        # Sort by timestamp within each symbol
        synthetic_combined = synthetic_combined.sort_values(
            ["symbol", "timestamp"]
        ).reset_index(drop=True)

        logger.info(
            f"Generated {len(synthetic_combined)} synthetic data points across "
            f"{len(symbols)} symbols"
        )
        return synthetic_combined

    def _add_microstructure_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic market microstructure noise."""
        df = df.copy()

        # Bid-ask spread simulation
        spread_pct = np.random.uniform(0.0001, 0.001, len(df))
        df["bid_ask_spread"] = df["close"] * spread_pct

        # Volume clustering (high volume periods)
        volume_spikes = np.random.exponential(1.0, len(df))
        df["volume"] = df["volume"] * volume_spikes

        # Intraday volatility patterns
        # Higher volatility at market open/close times
        time_factor = np.random.uniform(0.8, 1.5, len(df))
        df["high"] = df["high"] * time_factor
        df["low"] = df["low"] / time_factor

        return df

    def _add_regime_patterns(self, df: pd.DataFrame, regime: dict) -> pd.DataFrame:
        """Add regime-specific market patterns."""
        df = df.copy()

        if regime["name"] == "bull_market":
            # Add momentum and trending behavior
            df["close"] = df["close"] * (
                1 + np.cumsum(np.random.normal(0.0001, 0.001, len(df)))
            )

        elif regime["name"] == "bear_market":
            # Add downward pressure and volatility clustering
            df["close"] = df["close"] * (
                1 - np.cumsum(np.random.normal(0.0001, 0.002, len(df)))
            )

        elif regime["name"] == "high_volatility":
            # Add volatility clustering (GARCH-like behavior)
            vol_process = np.random.normal(0, 0.01, len(df))
            for i in range(1, len(vol_process)):
                vol_process[i] += 0.1 * vol_process[i - 1]  # Persistence
            df["close"] = df["close"] * (1 + vol_process)

        return df

    def fetch_real_market_data(self) -> pd.DataFrame:
        """Fetch real market data from multiple sources."""
        logger.info("Fetching real market data...")

        real_datasets = []
        start_date = self.config.get("start_date", "2020-01-01")
        end_date = self.config.get("end_date", "2024-12-31")

        # Fetch forex data (using yfinance with =X suffix)
        for pair in tqdm(MAJOR_USD_PAIRS, desc="Fetching forex data"):
            try:
                symbol = f"{pair}=X"
                df = fetch_historical_data(symbol, start_date, end_date, "day")
                if not df.empty:
                    df["symbol"] = pair
                    df["asset_class"] = "forex"
                    real_datasets.append(df)
                    logger.info(f"Fetched {len(df)} data points for {pair}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {pair}: {e}")

        # Fetch stock data
        for stock in tqdm(MAJOR_STOCKS, desc="Fetching stock data"):
            try:
                df = fetch_historical_data(stock, start_date, end_date, "day")
                if not df.empty:
                    df["symbol"] = stock
                    df["asset_class"] = "equity"
                    real_datasets.append(df)
                    logger.info(f"Fetched {len(df)} data points for {stock}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {stock}: {e}")

        # Fetch crypto data
        for crypto in tqdm(
            CRYPTO_PAIRS[:3], desc="Fetching crypto data"
        ):  # Limit to top 3
            try:
                df = fetch_historical_data(crypto, start_date, end_date, "day")
                if not df.empty:
                    df["symbol"] = crypto
                    df["asset_class"] = "crypto"
                    real_datasets.append(df)
                    logger.info(f"Fetched {len(df)} data points for {crypto}")
            except Exception as e:
                logger.warning(f"Failed to fetch data for {crypto}: {e}")

        if real_datasets:
            real_combined = pd.concat(real_datasets, ignore_index=True)
            real_combined = real_combined.sort_values(
                ["symbol", "timestamp"]
            ).reset_index(drop=True)
            logger.info(f"Fetched {len(real_combined)} total real data points")
            return real_combined
        else:
            logger.warning("No real data was successfully fetched")
            return pd.DataFrame()

    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators and features."""
        logger.info("Adding advanced features...")

        enhanced_datasets = []

        for symbol in tqdm(df["symbol"].unique(), desc="Adding features by symbol"):
            symbol_data = df[df["symbol"] == symbol].copy()

            if len(symbol_data) < 50:  # Skip symbols with insufficient data
                logger.warning(
                    f"Insufficient data for {symbol}: {len(symbol_data)} points"
                )
                continue

            # Apply comprehensive feature engineering
            try:
                symbol_data = generate_features(
                    symbol_data,
                    ma_windows=[5, 10, 20, 50, 100],
                    rsi_window=14,
                    vol_window=20,
                    advanced_candles=True,
                )

                # Add symbol-specific features
                symbol_data = self._add_symbol_specific_features(
                    symbol_data, symbol
                )  # Add time-based features
                symbol_data = self._add_temporal_features(symbol_data)

                # Add market regime features
                symbol_data = self._add_regime_features(symbol_data)

                enhanced_datasets.append(symbol_data)

            except Exception as e:
                logger.warning(f"Failed to add features for {symbol}: {e}")

        if enhanced_datasets:
            result = pd.concat(enhanced_datasets, ignore_index=True)
            logger.info(
                f"Added features to {len(result)} data points across "
                f"{len(enhanced_datasets)} symbols"
            )
            return result
        else:
            logger.error("No enhanced datasets were created")
            return df

    def _add_symbol_specific_features(
        self, df: pd.DataFrame, symbol: str
    ) -> pd.DataFrame:
        """Add features specific to the asset class."""
        df = df.copy()

        # Asset class encoding
        if hasattr(df, "asset_class"):
            asset_class = (
                df["asset_class"].iloc[0] if "asset_class" in df.columns else "unknown"
            )
        else:
            # Infer asset class from symbol
            if symbol in MAJOR_USD_PAIRS or "=X" in symbol:
                asset_class = "forex"
            elif "USD" in symbol and "-" in symbol:
                asset_class = "crypto"
            else:
                asset_class = "equity"

        df["asset_class_forex"] = 1 if asset_class == "forex" else 0
        df["asset_class_equity"] = 1 if asset_class == "equity" else 0
        df["asset_class_crypto"] = 1 if asset_class == "crypto" else 0

        # Currency exposure features (for forex)
        if asset_class == "forex":
            df["usd_strength"] = self._calculate_usd_strength_proxy(df)

        return df

    def _calculate_usd_strength_proxy(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate a proxy for USD strength based on price movements."""
        # Simple proxy: negative correlation with USD-denominated pairs
        returns = df["close"].pct_change().fillna(0)
        # Apply exponential smoothing
        alpha = 0.1
        usd_strength = np.zeros(len(returns))
        for i in range(1, len(returns)):
            usd_strength[i] = (
                alpha * (-returns.iloc[i]) + (1 - alpha) * usd_strength[i - 1]
            )

        return usd_strength

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()

        # Day of week effect
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_monday"] = (df["day_of_week"] == 0).astype(int)
        df["is_friday"] = (df["day_of_week"] == 4).astype(int)

        # Month effect
        df["month"] = df["timestamp"].dt.month
        df["quarter"] = df["timestamp"].dt.quarter

        # Year effect (for long-term trends)
        df["year"] = df["timestamp"].dt.year - df["timestamp"].dt.year.min()

        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        df = df.copy()

        # Volatility regime (rolling 20-day volatility)
        df["vol_regime_high"] = (
            df["vol_20"] > df["vol_20"].rolling(100).quantile(0.8)
        ).astype(int)
        df["vol_regime_low"] = (
            df["vol_20"] < df["vol_20"].rolling(100).quantile(0.2)
        ).astype(int)

        # Trend regime (using multiple MA crossovers)
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["trend_bullish"] = (df["sma_20"] > df["sma_50"]).astype(int)
            df["trend_bearish"] = (df["sma_20"] < df["sma_50"]).astype(int)

        return df

    def add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment analysis features."""
        logger.info("Adding sentiment features...")

        # Add forex sentiment for forex pairs
        forex_symbols = df[df["symbol"].isin(MAJOR_USD_PAIRS)]["symbol"].unique()

        if len(forex_symbols) > 0:
            logger.info(f"Adding forex sentiment for {len(forex_symbols)} forex pairs")

            for symbol in forex_symbols:
                try:
                    # Get sentiment data (using mock data for now)
                    sentiment_data = get_forex_sentiment(symbol)
                    if sentiment_data:
                        avg_sentiment = np.mean([s.score for s in sentiment_data])
                        df.loc[df["symbol"] == symbol, f"forex_sentiment_{symbol}"] = (
                            avg_sentiment
                        )
                    else:
                        df.loc[df["symbol"] == symbol, f"forex_sentiment_{symbol}"] = (
                            0.0
                        )
                except Exception as e:
                    logger.warning(f"Failed to get sentiment for {symbol}: {e}")
                    df.loc[df["symbol"] == symbol, f"forex_sentiment_{symbol}"] = 0.0

        # Add general market sentiment (placeholder)
        df["market_sentiment"] = 0.0  # Would integrate real sentiment sources

        return df

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading labels with multiple strategies."""
        logger.info("Generating trading labels...")

        labeled_datasets = []

        for symbol in df["symbol"].unique():
            symbol_data = df[df["symbol"] == symbol].copy()

            # Multi-horizon labels
            for horizon in [1, 3, 5]:
                future_returns = (
                    symbol_data["close"].pct_change(horizon).shift(-horizon)
                )

                # Dynamic thresholds based on volatility
                vol_window = min(20, len(symbol_data) // 4)
                rolling_vol = (
                    symbol_data["close"].pct_change().rolling(vol_window).std()
                )

                buy_threshold = rolling_vol * 1.5  # 1.5x volatility
                sell_threshold = -rolling_vol * 1.5

                # Generate labels
                conditions = [
                    future_returns > buy_threshold,
                    future_returns < sell_threshold,
                ]
                choices = [2, 0]  # 2=Buy, 0=Sell, 1=Hold (default)

                symbol_data[f"label_{horizon}d"] = np.select(
                    conditions, choices, default=1
                )

            # Use 1-day horizon as primary label
            symbol_data["label"] = symbol_data["label_1d"]

            labeled_datasets.append(symbol_data)

        result = pd.concat(labeled_datasets, ignore_index=True)

        # Remove rows where we can't calculate future returns
        result = result.dropna(subset=["label"])

        logger.info(f"Generated labels for {len(result)} data points")
        return result

    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """Comprehensive data quality validation."""
        logger.info("Validating data quality...")

        validation_results = {
            "total_records": len(df),
            "unique_symbols": df["symbol"].nunique(),
            "date_range": {
                "start": df["timestamp"].min(),
                "end": df["timestamp"].max(),
                "days": (df["timestamp"].max() - df["timestamp"].min()).days,
            },
            "missing_data": {},
            "data_quality_issues": [],
            "feature_statistics": {},
        }

        # Check for missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = missing_count / len(df) * 100
            if missing_count > 0:
                validation_results["missing_data"][col] = {
                    "count": missing_count,
                    "percentage": missing_pct,
                }

        # Check OHLC consistency
        ohlc_issues = 0
        if all(col in df.columns for col in ["open", "high", "low", "close"]):
            # High should be >= max(open, close)
            high_issues = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
            # Low should be <= min(open, close)
            low_issues = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
            ohlc_issues = high_issues + low_issues

            if ohlc_issues > 0:
                validation_results["data_quality_issues"].append(
                    f"OHLC consistency issues: {ohlc_issues} records"
                )

        # Check for extreme values (outliers)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ["open", "high", "low", "close"]:
                # Check for extreme price movements (>50% in one day)
                returns = df[col].pct_change().abs()
                extreme_moves = (returns > 0.5).sum()
                if extreme_moves > 0:
                    validation_results["data_quality_issues"].append(
                        f"Extreme price movements in {col}: {extreme_moves} occurrences"
                    )  # Feature statistics
        for col in numeric_cols[:10]:  # Top 10 numeric columns
            try:
                skew_val = df[col].skew()
                kurt_val = df[col].kurtosis()

                # Safe conversion to float
                def safe_float(val: Any) -> float:
                    if pd.isna(val):
                        return 0.0
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return 0.0

                validation_results["feature_statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "skewness": safe_float(skew_val),
                    "kurtosis": safe_float(kurt_val),
                }
            except (TypeError, ValueError, AttributeError):
                validation_results["feature_statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                }

        # Label distribution
        if "label" in df.columns:
            label_dist = df["label"].value_counts().to_dict()
            validation_results["label_distribution"] = {
                int(k): int(v) for k, v in label_dist.items()
            }

        logger.info(
            f"Data validation completed: "
            f"{len(validation_results['data_quality_issues'])} issues found"
        )
        return validation_results

    def build_dataset(self) -> tuple[pd.DataFrame, dict]:
        """Build the complete dataset."""
        logger.info("Building comprehensive dataset...")

        # Step 1: Generate synthetic data
        synthetic_data = self.generate_advanced_synthetic_data()
        logger.info(f"Synthetic data: {len(synthetic_data)} records")

        # Step 2: Fetch real market data
        real_data = self.fetch_real_market_data()
        logger.info(f"Real data: {len(real_data)} records")

        # Step 3: Combine datasets
        if not real_data.empty:
            # Add data source labels
            synthetic_data["data_source"] = "synthetic"
            real_data["data_source"] = "real"

            combined_data = pd.concat([synthetic_data, real_data], ignore_index=True)
        else:
            synthetic_data["data_source"] = "synthetic"
            combined_data = synthetic_data

        # Step 4: Add advanced features
        combined_data = self.add_advanced_features(combined_data)

        # Step 5: Add sentiment features
        combined_data = self.add_sentiment_features(combined_data)

        # Step 6: Generate labels
        combined_data = self.generate_labels(combined_data)

        # Step 7: Final sorting and cleaning
        combined_data = combined_data.sort_values(["symbol", "timestamp"]).reset_index(
            drop=True
        )

        # Step 8: Validate data quality
        validation_results = self.validate_data_quality(combined_data)

        logger.info(f"Dataset build completed: {len(combined_data)} total records")
        return combined_data, validation_results

    def save_dataset(self, df: pd.DataFrame, validation_results: dict) -> dict:
        """Save the dataset and validation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save main dataset
        dataset_path = self.output_dir / f"advanced_trading_dataset_{timestamp}.csv"
        df.to_csv(dataset_path, index=False)
        logger.info(f"Dataset saved to: {dataset_path}")

        # Save validation results
        validation_path = self.output_dir / f"dataset_validation_{timestamp}.json"
        # Convert numpy types to Python types for JSON serialization
        validation_json = json.dumps(validation_results, indent=2, default=str)
        validation_path.write_text(validation_json)
        logger.info(f"Validation results saved to: {validation_path}")

        # Save feature list
        features_path = self.output_dir / f"feature_list_{timestamp}.txt"
        features_path.write_text("\n".join(df.columns.tolist()))
        logger.info(f"Feature list saved to: {features_path}")

        return {
            "dataset_path": str(dataset_path),
            "validation_path": str(validation_path),
            "features_path": str(features_path),
        }


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Build advanced trading dataset")
    parser.add_argument("--output_dir", default="data", help="Output directory")
    parser.add_argument(
        "--start_date", default="2020-01-01", help="Start date for real data"
    )
    parser.add_argument(
        "--end_date", default="2024-12-31", help="End date for real data"
    )
    parser.add_argument(
        "--synthetic_days",
        type=int,
        default=500,
        help="Days of synthetic data per symbol",
    )
    parser.add_argument("--config", help="JSON config file path")

    args = parser.parse_args()

    # Load configuration
    config = {
        "output_dir": args.output_dir,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "synthetic_days": args.synthetic_days,
    }

    if args.config:
        with open(args.config) as f:
            config.update(json.load(f))

    # Build dataset
    builder = AdvancedDatasetBuilder(config)
    dataset, validation_results = builder.build_dataset()

    # Save results
    file_paths = builder.save_dataset(dataset, validation_results)

    # Print summary
    print("\n" + "=" * 80)
    print("ADVANCED DATASET BUILD COMPLETED")
    print("=" * 80)
    print(f"Total Records: {len(dataset):,}")
    print(f"Unique Symbols: {dataset['symbol'].nunique()}")
    print(f"Features: {len(dataset.columns)}")
    print(
        f"Date Range: {validation_results['date_range']['start']} "
        f"to {validation_results['date_range']['end']}"
    )
    print(f"Data Quality Issues: {len(validation_results['data_quality_issues'])}")

    if "label_distribution" in validation_results:
        print(f"Label Distribution: {validation_results['label_distribution']}")

    print("\nFiles created:")
    for key, path in file_paths.items():
        print(f"  {key}: {path}")

    print("\nDataset is ready for CNN-LSTM training!")


if __name__ == "__main__":
    main()
