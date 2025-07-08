"""CNN-LSTM Training Pipeline for Trading Prediction.

This module provides end-to-end training for CNN-LSTM models with integrated
sentiment analysis and technical indicators. It supports both classification
and regression tasks for market prediction.

Example usage:
>>> from trading_rl_agent.training.cnn_lstm import CNNLSTMTrainer, TrainingConfig
>>> trainer = CNNLSTMTrainer()
>>> model = trainer.train_from_config('src/configs/training/cnn_lstm_train.yaml')
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import pickle
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TimeSeriesDataSet
import yaml

from trading_rl_agent.data.features import generate_features
from trading_rl_agent.data.sentiment import SentimentAnalyzer, SentimentConfig
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel


# Define a simple load_data function if not available
def load_data(config: dict[str, Any]) -> pd.DataFrame:
    """Load data from configuration."""
    if "path" in config:
        return pd.read_csv(config["path"])
    else:
        # Return dummy data for testing
        import datetime

        dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
        np.random.seed(42)
        data = {
            "timestamp": dates,
            "open": 100 + np.random.randn(len(dates)).cumsum(),
            "high": 100 + np.random.randn(len(dates)).cumsum() + 1,
            "low": 100 + np.random.randn(len(dates)).cumsum() - 1,
            "close": 100 + np.random.randn(len(dates)).cumsum(),
            "volume": np.random.randint(1000000, 10000000, len(dates)),
        }
        return pd.DataFrame(data)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for CNN-LSTM training."""

    # Data configuration
    sequence_length: int = 60  # Number of timesteps to look back
    prediction_horizon: int = 1  # Steps ahead to predict
    train_split: float = 0.7
    val_split: float = 0.15
    # test_split is automatically 1 - train_split - val_split

    # Model configuration
    model_config: Optional[dict[str, Any]] = None
    use_attention: bool = True

    # Training configuration
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 15

    # Feature configuration
    include_sentiment: bool = True
    sentiment_weight: float = 0.2
    normalize_features: bool = True

    # Output configuration
    save_model: bool = True
    model_save_path: str = "models/cnn_lstm_trained.pth"
    save_scaler: bool = True
    scaler_save_path: str = "models/feature_scaler.pkl"

    # Logging
    log_interval: int = 10
    validate_interval: int = 5


class TimeSeriesDataModule(pl.LightningDataModule):
    """Lightning DataModule using ``TimeSeriesDataSet`` for sequence generation."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, config: TrainingConfig):
        super().__init__()
        self.features = features
        self.targets = targets
        self.config = config

    def setup(self, stage: Optional[str] = None):
        df = pd.DataFrame(
            self.features, columns=[f"f{i}" for i in range(self.features.shape[1])]
        )
        df["target"] = self.targets
        df["time_idx"] = np.arange(len(df))
        df["group"] = 0

        self.dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["group"],
            max_encoder_length=self.config.sequence_length,
            max_prediction_length=self.config.prediction_horizon,
            time_varying_unknown_reals=[f"f{i}" for i in range(self.features.shape[1])],
            target_normalizer=None,
        )

        n = len(df)
        train_end = int(n * self.config.train_split)
        val_end = int(n * (self.config.train_split + self.config.val_split))

        self.train_dataset = TimeSeriesDataSet.from_dataset(
            self.dataset, df[df["time_idx"] < train_end], stop_randomization=True
        )
        self.val_dataset = TimeSeriesDataSet.from_dataset(
            self.dataset,
            df[(df["time_idx"] >= train_end) & (df["time_idx"] < val_end)],
            stop_randomization=True,
        )
        self.test_dataset = TimeSeriesDataSet.from_dataset(
            self.dataset, df[df["time_idx"] >= val_end], stop_randomization=True
        )

    def train_dataloader(self):
        return self.train_dataset.to_dataloader(
            train=True, batch_size=self.config.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return self.val_dataset.to_dataloader(
            train=False, batch_size=self.config.batch_size, shuffle=False
        )

    def test_dataloader(self):
        return self.test_dataset.to_dataloader(
            train=False, batch_size=self.config.batch_size, shuffle=False
        )


class CNNLSTMLightning(pl.LightningModule):
    """Lightning module wrapping :class:`CNNLSTMModel`."""

    def __init__(self, input_dim: int, config: TrainingConfig, model_cfg: Optional[dict[str, Any]] = None):
        super().__init__()
        cfg = model_cfg or config.model_config or {
            "cnn_filters": [64, 128],
            "cnn_kernel_sizes": [3, 5],
            "lstm_units": 256,
            "dropout": 0.2,
        }
        self.model = CNNLSTMModel(
            input_dim=input_dim,
            output_size=cfg.get("output_size", 3),
            config=cfg,
            use_attention=cfg.get("use_attention", config.use_attention),
        )
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        out = self(x)
        loss = self.criterion(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        out = self(x)
        loss = self.criterion(out, y)
        acc = (torch.argmax(out, dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)


class CNNLSTMTrainer:
    """Main trainer class for CNN-LSTM models."""

    def __init__(self, config: Optional[TrainingConfig] = None):
        load_dotenv()
        self.config = config or TrainingConfig()
        self.model = None
        self.scaler = None
        self.sentiment_analyzer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        # Initialize sentiment analyzer if enabled
        if self.config.include_sentiment:
            sentiment_config = SentimentConfig()
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)

    def prepare_data(
        self, df: pd.DataFrame, symbols: Optional[list[str]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare data for training including sentiment features."""
        # Validate input data
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        # Check for required close column early
        if "close" not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # Generate technical features
        df_features = generate_features(df)

        # Check if features were generated successfully
        if df_features.empty:
            # For testing purposes, create a minimal DataFrame with NaN values
            import warnings

            warnings.warn(
                "Feature generation resulted in empty DataFrame; creating minimal DataFrame for testing."
            )
            # Create a minimal dataframe with at least one row for testing
            feature_cols = [
                "close",
                "log_returns",
                "sma_5",
                "rsi",
                "volatility",
                "sentiment",
            ]
            df_features = pd.DataFrame([{col: np.nan for col in feature_cols}])

        # Ensure we have enough data for sequence creation
        if len(df_features) < self.config.sequence_length:
            # Pad with the last available values or NaN
            if len(df_features) > 0:
                # Repeat the last row to meet minimum sequence length
                last_row = df_features.iloc[-1:].copy()
                while len(df_features) < self.config.sequence_length:
                    df_features = pd.concat([df_features, last_row], ignore_index=True)
            else:
                # Create minimum required rows with NaN
                feature_cols = [
                    "close",
                    "log_returns",
                    "sma_5",
                    "rsi",
                    "volatility",
                    "sentiment",
                ]
                df_features = pd.DataFrame(
                    [
                        {col: np.nan for col in feature_cols}
                        for _ in range(self.config.sequence_length)
                    ]
                )

        # Add sentiment features if enabled
        if self.config.include_sentiment and symbols and self.sentiment_analyzer:
            sentiment_features = self._add_sentiment_features(df_features, symbols)
            df_features = pd.concat([df_features, sentiment_features], axis=1)

        # Create target variable (next period return)
        if "close" not in df_features.columns:
            raise ValueError("DataFrame must contain 'close' column")

        # Use existing label column if available, otherwise create target from price changes
        if "label" in df_features.columns:
            # Use existing balanced labels (0=Hold, 1=Buy, 2=Sell)
            target_column = "label"
        elif "target" in df_features.columns:
            # Use existing target column
            target_column = "target"
        else:
            # Create target from price changes as fallback
            if len(df_features) < 2:
                raise ValueError(
                    "Insufficient data for target calculation (need at least 2 rows)"
                )
            df_features["target"] = (
                df_features["close"].pct_change().shift(-self.config.prediction_horizon)
            )
            target_column = "target"

        # Remove rows with NaN values
        df_clean = df_features.dropna()

        # Validate cleaned data
        if df_clean.empty:
            raise ValueError("No data remaining after NaN removal")

        # Separate features and targets
        feature_columns = [
            col
            for col in df_clean.columns
            if col not in ["target", "label", "timestamp"]
        ]
        features = df_clean[feature_columns].values.astype(np.float32)
        targets = df_clean[target_column].values.astype(np.float32)

        # Log feature columns and shape for debugging
        logger.info(f"Feature columns used: {feature_columns}")
        logger.info(f"Feature matrix shape: {features.shape}")

        # Error handling for empty or mismatched features
        if features.size == 0:
            raise ValueError(
                "No features available after preprocessing. Check feature engineering and NaN removal."
            )
        if len(features.shape) != 2:
            raise ValueError(
                f"Features must be 2D (samples, features), got shape {features.shape}"
            )

        # Robust error handling for missing/empty columns (check original input)
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")
            if df[col].isnull().all() or len(df[col].dropna()) == 0:
                raise ValueError(f"Column '{col}' is empty or all NaN")

        # Check if we have enough data for sequence generation after cleaning
        if len(features) < self.config.sequence_length + self.config.prediction_horizon:
            raise ValueError(
                f"Insufficient data for sequence generation: need at least {self.config.sequence_length + self.config.prediction_horizon} rows, got {len(features)} after preprocessing"
            )

        # Normalize features if requested
        if self.config.normalize_features:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        logger.info(
            f"Prepared data shape: features={features.shape}, targets={targets.shape}"
        )
        return features, targets

    def _add_sentiment_features(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        forex_pairs: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Add sentiment features to the dataframe, including forex pairs."""
        import importlib

        sentiment_df = pd.DataFrame(index=df.index)
        # Stock/crypto symbols (existing logic)
        for symbol in symbols:
            try:
                sentiment_scores = []
                for _, row in df.iterrows():
                    score = self.sentiment_analyzer.get_symbol_sentiment(
                        symbol, days_back=1
                    )
                    sentiment_scores.append(score)
                sentiment_df[f"sentiment_{symbol}"] = sentiment_scores
                sentiment_df[f"sentiment_{symbol}_abs"] = np.abs(sentiment_scores)
            except Exception as e:
                logger.warning(f"Failed to add sentiment for {symbol}: {e}")
                sentiment_df[f"sentiment_{symbol}"] = 0.0
                sentiment_df[f"sentiment_{symbol}_abs"] = 0.0
        # Forex pairs (new logic)
        if forex_pairs:
            try:
                forex_mod = importlib.import_module("src.data.forex_sentiment")
                get_forex_sentiment = getattr(forex_mod, "get_forex_sentiment")
                for pair in forex_pairs:
                    sentiment_scores = []
                    sentiment_data = get_forex_sentiment(pair)
                    score = sentiment_data[0].score if sentiment_data else 0.0
                    for _ in df.iterrows():
                        sentiment_scores.append(score)
                    sentiment_df[f"forex_sentiment_{pair}"] = sentiment_scores
                    sentiment_df[f"forex_sentiment_{pair}_abs"] = np.abs(
                        sentiment_scores
                    )
            except Exception as e:
                logger.warning(f"Failed to add forex sentiment: {e}")
                for pair in forex_pairs:
                    sentiment_df[f"forex_sentiment_{pair}"] = 0.0
                    sentiment_df[f"forex_sentiment_{pair}_abs"] = 0.0
        return sentiment_df

    def create_data_module(
        self, features: np.ndarray, targets: np.ndarray
    ) -> TimeSeriesDataModule:
        """Create a :class:`TimeSeriesDataModule` from arrays."""

        module = TimeSeriesDataModule(features, targets, self.config)
        module.setup()
        logger.info(
            "Created datamodule with %d training batches",
            len(module.train_dataloader()),
        )
        return module

    def initialize_model(
        self, input_dim: int, model_config: Optional[dict[str, Any]] = None
    ) -> None:
        """Initialize the CNN-LSTM model."""

        # Create model configuration
        if model_config is None:
            if self.config.model_config is None:
                model_config = {
                    "cnn_filters": [64, 128],
                    "cnn_kernel_sizes": [3, 5],
                    "lstm_units": 256,
                    "dropout": 0.2,
                }
            else:
                model_config = self.config.model_config

        # Create Lightning module
        self.model = CNNLSTMLightning(input_dim, self.config, model_config)

        # Log input_dim for debugging
        logger.info(f"Initializing model with input_dim={input_dim}")
        logger.info(f"Model configuration: {model_config}")

        logger.info(
            f"Initialized model with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def fit(self, datamodule: TimeSeriesDataModule) -> None:
        """Train the model using PyTorch Lightning."""
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.config.early_stopping_patience,
                    mode="min",
                )
            )

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            accelerator="auto",
        )
        trainer.fit(self.model, datamodule=datamodule)

    def _save_model(self):
        """Save the model and scaler."""
        # Create directory if it doesn't exist
        model_dir = Path(self.config.model_save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(
            {
                "model_state_dict": self.model.model.state_dict(),
                "config": self.config,
                "input_dim": self.model.model.input_dim,
            },
            self.config.model_save_path,
        )

        # Save scaler if available
        if self.scaler and self.config.save_scaler:
            scaler_dir = Path(self.config.scaler_save_path).parent
            scaler_dir.mkdir(parents=True, exist_ok=True)

            with open(self.config.scaler_save_path, "wb") as f:
                pickle.dump(self.scaler, f)

        logger.info(f"Model saved to {self.config.model_save_path}")

    def train_from_config(
        self, config_path: str
    ) -> CNNLSTMModel:
        """Train model from YAML configuration file."""
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Update training config
        for key, value in config_dict.get("training", {}).items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Load data
        data_config = config_dict.get("data", {})
        df = load_data(data_config.get("source", {}))

        # Prepare data
        symbols = data_config.get("symbols", ["AAPL"])
        features, targets = self.prepare_data(df, symbols)

        # Create datamodule
        datamodule = self.create_data_module(features, targets)

        # Initialize model with config from YAML
        model_config = config_dict.get("model", {})
        self.initialize_model(features.shape[1], model_config)

        # Train
        self.fit(datamodule)

        return self.model


# Example training configuration
def create_example_config() -> str:
    """Create an example training configuration file."""
    config = {
        "data": {
            "source": {"type": "csv", "path": "data/sample_data.csv"},
            "symbols": ["AAPL", "GOOGL", "TSLA"],
        },
        "training": {
            "sequence_length": 60,
            "prediction_horizon": 1,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "include_sentiment": True,
            "use_attention": True,
            "model_config": {
                "cnn_filters": [64, 128],
                "cnn_kernel_sizes": [3, 5],
                "lstm_units": 256,
                "dropout": 0.2,
            },
        },
    }

    config_path = "src/configs/training/cnn_lstm_train.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    return config_path


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create example config
    config_path = create_example_config()
    print(f"Created example config at {config_path}")

    # Train model
    trainer = CNNLSTMTrainer()
    try:
        model = trainer.train_from_config(config_path)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
