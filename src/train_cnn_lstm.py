"""CNN-LSTM Training Pipeline for Trading Prediction.

This module provides end-to-end training for CNN-LSTM models with integrated
sentiment analysis and technical indicators. It supports both classification
and regression tasks for market prediction.

Example usage:
>>> from src.train_cnn_lstm import CNNLSTMTrainer, TrainingConfig
>>> trainer = CNNLSTMTrainer()
>>> model, history = trainer.train_from_config('src/configs/training/cnn_lstm_train.yaml')
"""

import os
import sys
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
from dotenv import load_dotenv

try:
    # Try relative imports first (when used as module)
    from .models.cnn_lstm import CNNLSTMModel, CNNLSTMConfig
    from .data.sentiment import SentimentAnalyzer, SentimentConfig
    from .data.features import generate_features
except ImportError:
    # Fallback to absolute imports (when run directly)
    import sys
    from pathlib import Path
    
    # Add parent directory to path
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    sys.path.insert(0, str(current_dir))
    
    from src.models.cnn_lstm import CNNLSTMModel, CNNLSTMConfig
    from src.data.sentiment import SentimentAnalyzer, SentimentConfig
    from src.data.features import generate_features

# Define a simple load_data function if not available
def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data from configuration."""
    if 'path' in config:
        return pd.read_csv(config['path'])
    else:
        # Return dummy data for testing
        import datetime
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        data = {
            'timestamp': dates,
            'open': 100 + np.random.randn(len(dates)).cumsum(),
            'high': 100 + np.random.randn(len(dates)).cumsum() + 1,
            'low': 100 + np.random.randn(len(dates)).cumsum() - 1,
            'close': 100 + np.random.randn(len(dates)).cumsum(),
            'volume': np.random.randint(1000000, 10000000, len(dates))
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
    model_config: Optional[Dict[str, Any]] = None
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


class SequenceDataset:
    """Dataset for creating sequences from time series data."""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int, prediction_horizon: int = 1):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Create sequences
        self.sequences, self.sequence_targets = self._create_sequences()
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from the raw data."""
        sequences = []
        targets = []
        
        for i in range(len(self.features) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            seq = self.features[i:i + self.sequence_length]
            
            # Target (could be next value, price change, etc.)
            if self.prediction_horizon == 1:
                target = self.targets[i + self.sequence_length]
            else:
                # For multi-step prediction, take the last value
                target = self.targets[i + self.sequence_length + self.prediction_horizon - 1]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.sequence_targets[idx]


class CNNLSTMTrainer:
    """Main trainer class for CNN-LSTM models."""
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        load_dotenv()
        self.config = config or TrainingConfig()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        self.sentiment_analyzer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        # Initialize sentiment analyzer if enabled
        if self.config.include_sentiment:
            sentiment_config = SentimentConfig()
            self.sentiment_analyzer = SentimentAnalyzer(sentiment_config)
    
    def prepare_data(self, df: pd.DataFrame, symbols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training including sentiment features."""
        # Validate input data
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Check for required close column early
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        # Generate technical features
        df_features = generate_features(df)
        
        # Check if features were generated successfully
        if df_features.empty:
            # For testing purposes, create a minimal DataFrame with NaN values
            import warnings
            warnings.warn("Feature generation resulted in empty DataFrame; creating minimal DataFrame for testing.")
            # Create a minimal dataframe with at least one row for testing
            feature_cols = ['close', 'log_returns', 'sma_5', 'rsi', 'volatility', 'sentiment']
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
                feature_cols = ['close', 'log_returns', 'sma_5', 'rsi', 'volatility', 'sentiment']
                df_features = pd.DataFrame([{col: np.nan for col in feature_cols} for _ in range(self.config.sequence_length)])
        
        # Add sentiment features if enabled
        if self.config.include_sentiment and symbols and self.sentiment_analyzer:
            sentiment_features = self._add_sentiment_features(df_features, symbols)
            df_features = pd.concat([df_features, sentiment_features], axis=1)
            
        # Create target variable (next period return)
        if 'close' not in df_features.columns:
            raise ValueError("DataFrame must contain 'close' column")
            
        # Use existing label column if available, otherwise create target from price changes
        if 'label' in df_features.columns:
            # Use existing balanced labels (0=Hold, 1=Buy, 2=Sell)
            target_column = 'label'
        elif 'target' in df_features.columns:
            # Use existing target column
            target_column = 'target'
        else:
            # Create target from price changes as fallback
            if len(df_features) < 2:
                raise ValueError("Insufficient data for target calculation (need at least 2 rows)")
            df_features['target'] = df_features['close'].pct_change().shift(-self.config.prediction_horizon)
            target_column = 'target'
        
        # Remove rows with NaN values
        df_clean = df_features.dropna()
        
        # Validate cleaned data
        if df_clean.empty:
            raise ValueError("No data remaining after NaN removal")
            
        # Separate features and targets
        feature_columns = [col for col in df_clean.columns if col not in ['target', 'label', 'timestamp']]
        features = df_clean[feature_columns].values.astype(np.float32)
        targets = df_clean[target_column].values.astype(np.float32)
        
        # Log feature columns and shape for debugging
        logger.info(f"Feature columns used: {feature_columns}")
        logger.info(f"Feature matrix shape: {features.shape}")
        
        # Error handling for empty or mismatched features
        if features.size == 0:
            raise ValueError("No features available after preprocessing. Check feature engineering and NaN removal.")
        if len(features.shape) != 2:
            raise ValueError(f"Features must be 2D (samples, features), got shape {features.shape}")
        
        # Robust error handling for missing/empty columns (check original input)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise KeyError(f"Missing required column: {col}")
            if df[col].isnull().all() or len(df[col].dropna()) == 0:
                raise ValueError(f"Column '{col}' is empty or all NaN")
        
        # Check if we have enough data for sequence generation after cleaning
        if len(features) < self.config.sequence_length + self.config.prediction_horizon:
            raise ValueError(f"Insufficient data for sequence generation: need at least {self.config.sequence_length + self.config.prediction_horizon} rows, got {len(features)} after preprocessing")
        
        # Normalize features if requested
        if self.config.normalize_features:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        logger.info(f"Prepared data shape: features={features.shape}, targets={targets.shape}")
        return features, targets
    
    def _add_sentiment_features(self, df: pd.DataFrame, symbols: List[str], forex_pairs: Optional[List[str]] = None) -> pd.DataFrame:
        """Add sentiment features to the dataframe, including forex pairs."""
        import importlib
        sentiment_df = pd.DataFrame(index=df.index)
        # Stock/crypto symbols (existing logic)
        for symbol in symbols:
            try:
                sentiment_scores = []
                for _, row in df.iterrows():
                    score = self.sentiment_analyzer.get_symbol_sentiment(symbol, days_back=1)
                    sentiment_scores.append(score)
                sentiment_df[f'sentiment_{symbol}'] = sentiment_scores
                sentiment_df[f'sentiment_{symbol}_abs'] = np.abs(sentiment_scores)
            except Exception as e:
                logger.warning(f"Failed to add sentiment for {symbol}: {e}")
                sentiment_df[f'sentiment_{symbol}'] = 0.0
                sentiment_df[f'sentiment_{symbol}_abs'] = 0.0
        # Forex pairs (new logic)
        if forex_pairs:
            try:
                forex_mod = importlib.import_module('src.data.forex_sentiment')
                get_forex_sentiment = getattr(forex_mod, 'get_forex_sentiment')
                for pair in forex_pairs:
                    sentiment_scores = []
                    sentiment_data = get_forex_sentiment(pair)
                    score = sentiment_data[0].score if sentiment_data else 0.0
                    for _ in df.iterrows():
                        sentiment_scores.append(score)
                    sentiment_df[f'forex_sentiment_{pair}'] = sentiment_scores
                    sentiment_df[f'forex_sentiment_{pair}_abs'] = np.abs(sentiment_scores)
            except Exception as e:
                logger.warning(f"Failed to add forex sentiment: {e}")
                for pair in forex_pairs:
                    sentiment_df[f'forex_sentiment_{pair}'] = 0.0
                    sentiment_df[f'forex_sentiment_{pair}_abs'] = 0.0
        return sentiment_df
    
    def create_data_loaders(self, features: np.ndarray, targets: np.ndarray) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/validation/test data loaders."""
        
        # Split data chronologically
        n_samples = len(features)
        train_end = int(n_samples * self.config.train_split)
        val_end = int(n_samples * (self.config.train_split + self.config.val_split))
        
        train_features, train_targets = features[:train_end], targets[:train_end]
        val_features, val_targets = features[train_end:val_end], targets[train_end:val_end]
        test_features, test_targets = features[val_end:], targets[val_end:]
        
        # Create sequence datasets
        train_dataset = SequenceDataset(train_features, train_targets, 
                                      self.config.sequence_length, self.config.prediction_horizon)
        val_dataset = SequenceDataset(val_features, val_targets,
                                    self.config.sequence_length, self.config.prediction_horizon)
        test_dataset = SequenceDataset(test_features, test_targets,
                                     self.config.sequence_length, self.config.prediction_horizon)
        
        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_dataset.sequences), 
                         torch.FloatTensor(train_dataset.sequence_targets)),
            batch_size=self.config.batch_size, shuffle=False  # Keep temporal order
        )
        
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_dataset.sequences),
                         torch.FloatTensor(val_dataset.sequence_targets)),
            batch_size=self.config.batch_size, shuffle=False
        )
        
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_dataset.sequences),
                         torch.FloatTensor(test_dataset.sequence_targets)),
            batch_size=self.config.batch_size, shuffle=False
        )
        
        logger.info(f"Created data loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
        return train_loader, val_loader, test_loader
    
    def initialize_model(self, input_dim: int, model_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the CNN-LSTM model."""
        
        # Create model configuration
        if model_config is None:
            if self.config.model_config is None:
                model_config = {
                    'cnn_filters': [64, 128],
                    'cnn_kernel_sizes': [3, 5],
                    'lstm_units': 256,
                    'dropout': 0.2
                }
            else:
                model_config = self.config.model_config
        
        # Create model
        self.model = CNNLSTMModel(
            input_dim=input_dim,
            output_size=model_config.get('output_size', 3),  # 3 classes: Hold, Buy, Sell
            config=model_config,
            use_attention=model_config.get('use_attention', self.config.use_attention)
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Use CrossEntropyLoss for classification (labels 0, 1, 2)
        self.criterion = nn.CrossEntropyLoss()
        
        # Log input_dim for debugging
        logger.info(f"Initializing model with input_dim={input_dim}")
        logger.info(f"Model configuration: {model_config}")
        
        logger.info(f"Initialized model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device).long()  # Convert to long for classification
            
            self.optimizer.zero_grad()
            output = self.model(data)  # Don't squeeze for classification
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % self.config.log_interval == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device).long()  # Convert to long for classification
                output = self.model(data)  # Don't squeeze for classification
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Get predicted classes for classification accuracy
                predicted_classes = torch.argmax(output, dim=1)
                predictions.extend(predicted_classes.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # Calculate accuracy as performance metric for classification
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        accuracy = np.mean(predictions == actuals) if len(predictions) > 0 else 0.0
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """Full training loop with early stopping."""
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []  # Changed from correlation to accuracy
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            if epoch % self.config.validate_interval == 0:
                val_loss, val_acc = self.validate(val_loader)  # Changed from val_corr to val_acc
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)  # Changed from correlation to accuracy
                
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, Val Acc={val_acc:.4f}")  # Updated logging
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    if self.config.save_model:
                        self._save_model()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return history
    
    def _save_model(self):
        """Save the model and scaler."""
        # Create directory if it doesn't exist
        model_dir = Path(self.config.model_save_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'input_dim': self.model.input_dim
        }, self.config.model_save_path)
        
        # Save scaler if available
        if self.scaler and self.config.save_scaler:
            scaler_dir = Path(self.config.scaler_save_path).parent
            scaler_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config.scaler_save_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {self.config.model_save_path}")
    
    def train_from_config(self, config_path: str) -> Tuple[CNNLSTMModel, Dict[str, List[float]]]:
        """Train model from YAML configuration file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update training config
        for key, value in config_dict.get('training', {}).items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Load data
        data_config = config_dict.get('data', {})
        df = load_data(data_config.get('source', {}))
        
        # Prepare data
        symbols = data_config.get('symbols', ['AAPL'])
        features, targets = self.prepare_data(df, symbols)
        
        # Create data loaders
        train_loader, val_loader, test_loader = self.create_data_loaders(features, targets)
        
        # Initialize model with config from YAML
        model_config = config_dict.get('model', {})
        self.initialize_model(features.shape[1], model_config)
        
        # Train
        history = self.train(train_loader, val_loader)
        
        return self.model, history


# Example training configuration
def create_example_config() -> str:
    """Create an example training configuration file."""
    config = {
        'data': {
            'source': {
                'type': 'csv',
                'path': 'data/sample_data.csv'
            },
            'symbols': ['AAPL', 'GOOGL', 'TSLA']
        },
        'training': {
            'sequence_length': 60,
            'prediction_horizon': 1,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'include_sentiment': True,
            'use_attention': True,
            'model_config': {
                'cnn_filters': [64, 128],
                'cnn_kernel_sizes': [3, 5],
                'lstm_units': 256,
                'dropout': 0.2
            }
        }
    }
    
    config_path = 'src/configs/training/cnn_lstm_train.yaml'
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)
    
    return config_path


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create example config
    config_path = create_example_config()
    print(f"Created example config at {config_path}")
    
    # Train model
    trainer = CNNLSTMTrainer()
    try:
        model, history = trainer.train_from_config(config_path)
        print("Training completed successfully!")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        if history['val_loss']:
            print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    except Exception as e:
        print(f"Training failed: {e}")
