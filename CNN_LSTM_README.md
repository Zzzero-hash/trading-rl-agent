# Robust CNN+LSTM Dataset Pipeline

A production-ready, reproducible dataset generation and training pipeline for CNN+LSTM models in trading applications.

## Core Components

### 1. Dataset Builder

**File**: `src/trading_rl_agent/data/robust_dataset_builder.py`

- Multi-source data integration (real market + synthetic)
- Advanced feature engineering (65+ technical indicators)
- Sequence generation optimized for CNN+LSTM
- Real-time compatible preprocessing

### 2. Training Pipeline

**File**: `train_cnn_lstm.py`

- End-to-end CNN+LSTM model training
- GPU acceleration with early stopping
- Comprehensive metrics and checkpointing
- Configurable architecture

### 3. Real-Time Inference

**File**: `realtime_cnn_lstm_example.py`

- Streaming data processing
- Low-latency predictions
- Trading signal generation
- Live monitoring

## Quick Start

### Basic Training

```bash
python train_cnn_lstm.py \
    --symbols AAPL GOOGL MSFT \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --sequence-length 60
```

### Real-Time Usage

```python
from trading_rl_agent.data.robust_dataset_builder import RealTimeDatasetLoader

# Initialize processor
rt_loader = RealTimeDatasetLoader('dataset/version/path')

# Process new data
processed_seq = rt_loader.process_realtime_data(market_data)

# Make prediction
prediction = model(torch.FloatTensor(processed_seq))
```

## Configuration

### Dataset Configuration

```python
config = DatasetConfig(
    symbols=['AAPL', 'GOOGL', 'MSFT'],
    start_date='2020-01-01',
    end_date='2024-12-31',
    sequence_length=60,
    real_data_ratio=0.8,
    technical_indicators=True,
    market_regime_features=True
)
```

### Model Configuration

```python
model_config = {
    "cnn_filters": [64, 128, 256],
    "cnn_kernel_sizes": [3, 3, 3],
    "lstm_units": 128,
    "dropout": 0.2
}
```

## Features

- **Reproducible**: Versioned datasets with complete metadata
- **Real-Time Compatible**: Same pipeline for training and inference
- **Production Ready**: Comprehensive error handling and validation
- **Scalable**: Optimized for large datasets and multiple symbols

## Dependencies

```bash
pip install torch scikit-learn pandas numpy yfinance matplotlib
```

## Architecture

```
Data Sources → Feature Engineering → Sequence Creation → CNN+LSTM Training → Real-Time Inference
```

The pipeline generates 65+ features including technical indicators, temporal patterns, and market regime features, all optimized for CNN+LSTM architectures.
