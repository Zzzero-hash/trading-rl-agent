# Robust CNN+LSTM Dataset Pipeline Documentation

## Overview

This documentation describes a comprehensive, production-ready dataset generation and training pipeline for CNN+LSTM models in trading applications. The pipeline is designed to be **reproducible**, **real-time replicatable**, and **scalable**.

## Key Features

### 🔄 Reproducible Dataset Generation

- **Version Control**: Every dataset is versioned with timestamps and metadata
- **Deterministic Processing**: Consistent feature engineering across training and inference
- **Configuration Management**: All parameters saved for exact reproduction

### ⚡ Real-Time Compatibility

- **Streaming Processing**: Same pipeline for batch and real-time data
- **Feature Consistency**: Identical feature engineering for training and inference
- **Low Latency**: Optimized for real-time trading applications

### 🎯 CNN+LSTM Optimized

- **Sequence Generation**: Proper temporal sequences with configurable overlap
- **Feature Engineering**: Technical indicators + temporal + regime features
- **Data Quality**: Comprehensive validation and cleaning

## Architecture

```
Data Sources → Feature Engineering → Sequence Creation → Model Training → Real-time Inference
     ↓                ↓                    ↓              ↓                ↓
  Real Data      Technical         CNN+LSTM         PyTorch          Streaming
  Synthetic      Indicators        Sequences        Training         Predictions
  APIs           Temporal          Targets          Evaluation       Signals
                 Regimes
```

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements-cnn-lstm.txt

# Verify installation
python quickstart_cnn_lstm.py --demo
```

### 2. Run Demo

```bash
# Quick demonstration (5-10 minutes)
python quickstart_cnn_lstm.py --demo
```

### 3. Full Training

```bash
# Complete training pipeline
python train_cnn_lstm.py --symbols AAPL GOOGL MSFT TSLA AMZN
```

### 4. Real-time Inference

```bash
# Test real-time predictions
python realtime_cnn_lstm_example.py
```

## Core Components

### 1. RobustDatasetBuilder

**Purpose**: Generate high-quality, reproducible datasets

**Key Features**:

- Multi-source data integration (APIs + synthetic)
- Advanced feature engineering (78+ features)
- Configurable data quality thresholds
- Automatic versioning and metadata

**Usage**:

```python
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder, DatasetConfig

config = DatasetConfig(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    sequence_length=60,
    real_data_ratio=0.8
)

builder = RobustDatasetBuilder(config)
sequences, targets, info = builder.build_dataset()
```

### 2. CNNLSTMTrainer

**Purpose**: Train CNN+LSTM models with comprehensive monitoring

**Key Features**:

- PyTorch-based training with GPU support
- Early stopping and learning rate scheduling
- Comprehensive metrics and visualization
- Model checkpointing

**Usage**:

```python
from train_cnn_lstm import CNNLSTMTrainer, create_model_config

trainer = CNNLSTMTrainer(
    model_config=create_model_config(),
    training_config=create_training_config()
)

summary = trainer.train_from_dataset(sequences, targets)
```

### 3. RealTimeDatasetLoader

**Purpose**: Process real-time data using the same pipeline as training

**Key Features**:

- Identical feature engineering to training
- Fitted scaler for consistent normalization
- Sequence generation for CNN+LSTM input

**Usage**:

```python
from trading_rl_agent.data.robust_dataset_builder import RealTimeDatasetLoader

loader = RealTimeDatasetLoader('path/to/dataset/version')
processed_seq = loader.process_realtime_data(new_market_data)
```

## Configuration Options

### Dataset Configuration

```python
@dataclass
class DatasetConfig:
    # Data sources
    symbols: List[str]                    # Trading symbols to include
    start_date: str                       # Start date (YYYY-MM-DD)
    end_date: str                         # End date (YYYY-MM-DD)
    timeframe: str = "1d"                 # Data frequency (1d, 1h, 5m)

    # Dataset composition
    real_data_ratio: float = 0.7          # Ratio of real vs synthetic data
    min_samples_per_symbol: int = 1000    # Minimum samples required

    # CNN+LSTM specific
    sequence_length: int = 60             # Lookback window
    prediction_horizon: int = 1           # Steps ahead to predict
    overlap_ratio: float = 0.8            # Overlap between sequences

    # Feature engineering
    technical_indicators: bool = True     # Include technical indicators
    sentiment_features: bool = True       # Include sentiment analysis
    market_regime_features: bool = True   # Include regime detection

    # Data quality
    outlier_threshold: float = 3.0        # Standard deviations for outliers
    missing_value_threshold: float = 0.05 # Max missing data ratio
```

### Model Configuration

```python
def create_model_config():
    return {
        "cnn_filters": [64, 128, 256],        # CNN filter sizes
        "cnn_kernel_sizes": [3, 3, 3],        # CNN kernel sizes
        "lstm_units": 128,                    # LSTM hidden units
        "dropout": 0.2                        # Dropout rate
    }
```

### Training Configuration

```python
def create_training_config():
    return {
        "epochs": 100,                        # Maximum epochs
        "batch_size": 64,                     # Training batch size
        "learning_rate": 0.001,               # Initial learning rate
        "weight_decay": 1e-5,                 # L2 regularization
        "val_split": 0.2,                     # Validation split ratio
        "early_stopping_patience": 15         # Early stopping patience
    }
```

## Feature Engineering

The pipeline generates 70+ features across multiple categories:

### Technical Indicators (42 features)

- Moving averages (SMA, EMA): 5, 10, 20, 50 periods
- Momentum indicators: RSI, MACD, ROC
- Volatility indicators: Bollinger Bands, ATR
- Volume indicators: OBV, Volume SMA

### Temporal Features (10 features)

- Cyclical time encoding: hour, day, month (sin/cos)
- Calendar features: weekend, month-end, quarter-end
- Time-based patterns for LSTM learning

### Market Regime Features (8 features)

- Trend regime: Bull/bear market detection
- Volatility regime: High/low volatility periods
- Momentum regime: Strong/weak momentum phases

### Price Action Features (15 features)

- Returns: Log returns, rolling returns
- Volatility: Realized volatility, volatility of volatility
- Intraday patterns: High-low spreads, gaps

## Data Quality Assurance

### Validation Steps

1. **OHLC Consistency**: Ensure High ≥ Open, Close and Low ≤ Open, Close
2. **Outlier Detection**: Remove values beyond 3σ from mean
3. **Missing Data**: Handle missing values with forward fill and interpolation
4. **Duplicates**: Remove duplicate timestamps per symbol
5. **Volume Validation**: Ensure positive volume values

### Quality Metrics

- **Data Completeness**: % of non-missing values
- **Outlier Rate**: % of values flagged as outliers
- **Feature Correlation**: Correlation matrix for feature redundancy
- **Target Distribution**: Statistics on prediction targets

## Real-Time Pipeline

### Architecture

```
Market Data → Feature Engineering → Scaling → Sequence → CNN+LSTM → Signal
     ↓              ↓                ↓         ↓           ↓         ↓
  Live API      Same Pipeline    Fitted     Rolling     Trained   Trading
  OHLCV         as Training      Scaler     Window      Model     Action
```

### Implementation

```python
# 1. Initialize real-time processor
rt_loader = RealTimeDatasetLoader('dataset/version')

# 2. Process incoming data
new_data = get_live_market_data()  # Your data source
processed_seq = rt_loader.process_realtime_data(new_data)

# 3. Make prediction
prediction = model(torch.FloatTensor(processed_seq))

# 4. Generate trading signal
signal = generate_trading_signal(prediction)
```

### Performance Considerations

- **Latency**: < 50ms for feature engineering and prediction
- **Memory**: Efficient sliding window updates
- **Reliability**: Graceful handling of missing or delayed data

## Performance Monitoring

### Training Metrics

- **Loss Curves**: Training and validation loss over epochs
- **Correlation**: Prediction-target correlation
- **RMSE/MAE**: Root mean square error and mean absolute error
- **Sharpe Ratio**: Risk-adjusted returns on predictions

### Real-Time Monitoring

- **Prediction Distribution**: Track prediction statistics
- **Confidence Scores**: Model uncertainty estimation
- **Signal Performance**: Track trading signal accuracy
- **Data Quality**: Monitor incoming data quality

## Best Practices

### Dataset Generation

1. **Start Small**: Begin with 3-5 symbols and shorter time periods
2. **Validate Quality**: Always check data quality metrics
3. **Version Control**: Save dataset versions for reproducibility
4. **Test Pipeline**: Validate feature consistency between train/inference

### Model Training

1. **Cross-Validation**: Use time-series aware splitting
2. **Regularization**: Apply dropout and weight decay
3. **Early Stopping**: Prevent overfitting with patience
4. **Hyperparameter Tuning**: Use systematic grid/random search

### Production Deployment

1. **Model Validation**: Thorough backtesting before deployment
2. **Monitoring**: Continuous monitoring of predictions and performance
3. **Fallback**: Implement fallback strategies for model failures
4. **Updates**: Regular model retraining with new data

## Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Solution: Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:./src"
```

**2. Memory Issues**

```python
# Solution: Reduce batch size or sequence length
config.batch_size = 32  # Instead of 64
config.sequence_length = 30  # Instead of 60
```

**3. Training Divergence**

```python
# Solution: Lower learning rate and add regularization
config.learning_rate = 0.0001
config.weight_decay = 1e-4
```

**4. Poor Real-Time Performance**

```python
# Solution: Optimize feature engineering
config.technical_indicators = True
config.sentiment_features = False  # Disable slow features
```

### Debugging Steps

1. **Check Data**: Verify data quality and shape
2. **Validate Features**: Ensure feature consistency
3. **Monitor Training**: Watch loss curves and metrics
4. **Test Pipeline**: Validate end-to-end workflow

## Examples

### Basic Training

```bash
python train_cnn_lstm.py \
    --symbols AAPL GOOGL MSFT \
    --start-date 2022-01-01 \
    --end-date 2024-01-01 \
    --sequence-length 30 \
    --output-dir outputs/basic_training
```

### Advanced Configuration

```python
config = DatasetConfig(
    symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    timeframe='1h',  # Hourly data
    real_data_ratio=0.9,  # 90% real data
    sequence_length=120,  # Longer sequences
    prediction_horizon=5,  # 5-step ahead prediction
    technical_indicators=True,
    market_regime_features=True,
    output_dir='outputs/advanced_dataset'
)
```

### Real-Time Trading Bot

```python
class TradingBot:
    def __init__(self, model_path, dataset_path):
        self.predictor = RealTimeCNNLSTMPredictor(model_path, dataset_path)

    def on_market_data(self, data):
        result = self.predictor.predict_next_return(data)

        if result['signal'] in ['BUY', 'STRONG_BUY']:
            self.place_buy_order(data['symbol'])
        elif result['signal'] in ['SELL', 'STRONG_SELL']:
            self.place_sell_order(data['symbol'])
```

## Future Enhancements

### Planned Features

1. **Multi-Asset Models**: Train on multiple assets simultaneously
2. **Attention Mechanisms**: Add attention layers to CNN+LSTM
3. **Ensemble Methods**: Combine multiple models for better predictions
4. **Advanced Sentiment**: Integration with news and social media APIs
5. **Risk Management**: Built-in position sizing and risk controls

### Research Directions

1. **Transformer Models**: Experiment with transformer architectures
2. **Reinforcement Learning**: Combine with RL for action optimization
3. **Meta-Learning**: Adapt quickly to new market regimes
4. **Explainable AI**: Understand model predictions and decisions

## Support and Contributions

### Getting Help

1. **Documentation**: Start with this comprehensive guide
2. **Examples**: Review the provided example scripts
3. **Logs**: Check training logs for detailed error messages
4. **Community**: Engage with the trading ML community

### Contributing

1. **Code Quality**: Follow PEP 8 and include tests
2. **Documentation**: Update docs for new features
3. **Performance**: Benchmark improvements
4. **Compatibility**: Ensure backward compatibility

---

_This documentation is part of the Trading RL Agent project. For the latest updates and examples, check the project repository._
