# Data Standardization Guide

## Overview

The Data Standardization system ensures **consistent feature engineering** between training and live inference. This is critical for production trading systems where model inputs must be exactly the same to avoid compilation errors and incorrect predictions.

## 🎯 Key Benefits

1. **Consistent Input Dimensions**: Model always receives exactly the same number of features
2. **Feature Order Consistency**: Features are always in the same order
3. **Missing Data Handling**: Consistent strategies for handling missing values
4. **Data Quality**: Automatic cleaning of invalid values (negative prices, infinite values)
5. **Scalability**: Same preprocessing pipeline for training and live data
6. **Reproducibility**: Deterministic feature engineering

## 📊 Feature Configuration

The system uses a standardized set of **78 features** organized into categories:

### Core Price Features (5)

- `open`, `high`, `low`, `close`, `volume`

### Technical Indicators (20)

- `log_return`, `sma_5`, `sma_10`, `sma_20`, `sma_50`
- `rsi_14`, `vol_20`, `ema_20`, `macd_line`, `macd_signal`
- `macd_hist`, `atr_14`, `bb_mavg_20`, `bb_upper_20`, `bb_lower_20`
- `stoch_k`, `stoch_d`, `adx_14`, `wr_14`, `obv`

### Candlestick Patterns (18) - Binary Features

- `doji`, `hammer`, `hanging_man`, `bullish_engulfing`, `bearish_engulfing`
- `shooting_star`, `morning_star`, `evening_star`, `inside_bar`, `outside_bar`
- `tweezer_top`, `tweezer_bottom`, `three_white_soldiers`, `three_black_crows`
- `bullish_harami`, `bearish_harami`, `dark_cloud_cover`, `piercing_line`

### Candlestick Characteristics (9)

- `body_size`, `range_size`, `rel_body_size`, `upper_shadow`, `lower_shadow`
- `rel_upper_shadow`, `rel_lower_shadow`, `body_position`, `body_type`

### Rolling Candlestick Features (15)

- 5-period, 10-period, and 20-period averages of candlestick characteristics
- `avg_rel_body_5`, `avg_upper_shadow_5`, `avg_lower_shadow_5`, etc.

### Sentiment Features (2)

- `sentiment`, `sentiment_magnitude`

### Time Features (4)

- `hour`, `day_of_week`, `month`, `quarter`

### Market Regime Features (5)

- `price_change_pct`, `high_low_pct`, `volume_ma_20`, `volume_ratio`, `volume_change`

## 🔧 Usage

### 1. Training Pipeline Integration

The standardization is automatically integrated into the training pipeline:

```bash
# Build dataset with standardization
python train.py build-dataset --forex-focused

# This will:
# 1. Build the dataset with all features
# 2. Create standardized dataset with 78 features
# 3. Save the standardizer to outputs/data_standardizer.pkl
# 4. Save standardized dataset to outputs/standardized_dataset.csv
```

### 2. Manual Standardization

```python
from src.trading_rl_agent.data.data_standardizer import create_standardized_dataset

# Create standardized dataset
standardized_df, standardizer = create_standardized_dataset(
    df=your_dataframe,
    save_path="outputs/data_standardizer.pkl"
)

print(f"Feature count: {standardizer.get_feature_count()}")  # 78
print(f"Feature names: {standardizer.get_feature_names()}")
```

### 3. Live Data Processing

```python
from src.trading_rl_agent.data.data_standardizer import DataStandardizer, LiveDataProcessor

# Load the standardizer
standardizer = DataStandardizer.load("outputs/data_standardizer.pkl")
live_processor = LiveDataProcessor(standardizer)

# Process live data (can be missing features, different order, etc.)
live_data = {
    "open": 100.0,
    "high": 102.0,
    "low": 99.0,
    "close": 101.0,
    "volume": 1000000,
    # Missing features will be filled with defaults
}

# Get standardized data with exactly 78 features
processed_data = live_processor.process_single_row(live_data)
print(f"Processed shape: {processed_data.shape}")  # (1, 78)
```

### 4. Model Integration

```python
import torch
from src.trading_rl_agent.models.cnn_lstm import CNNLSTMModel

# Load standardizer
standardizer = DataStandardizer.load("outputs/data_standardizer.pkl")
live_processor = LiveDataProcessor(standardizer)

# Create model with correct input dimension
model = CNNLSTMModel(input_dim=standardizer.get_feature_count())  # 78
model.load_state_dict(torch.load("outputs/best_model.pth"))

# Process live data
live_data = {"open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 1000000}
processed_data = live_processor.process_single_row(live_data)

# Make prediction
with torch.no_grad():
    input_tensor = torch.FloatTensor(processed_data.values)
    prediction = model(input_tensor)
    print(f"Prediction: {prediction.item()}")
```

## 🛡️ Data Quality Features

### Missing Value Handling

- **Price features**: Forward fill, then backward fill
- **Technical indicators**: Forward fill
- **Candlestick patterns**: Fill with 0
- **Sentiment features**: Fill with 0
- **Time features**: Fill with 0

### Invalid Value Cleaning

- **Negative prices**: Clipped to 0
- **Infinite values**: Replaced with 0
- **Negative volumes**: Clipped to 0
- **Binary features**: Clipped to [0, 1]

### Feature Validation

- Ensures all 78 features are present
- Validates feature order
- Checks for NaN and infinite values
- Warns about data quality issues

## 📁 Generated Files

When you run the standardization, these files are created:

```
outputs/
├── data_standardizer.pkl          # Standardizer object (for loading)
├── data_standardizer.json         # Human-readable config
├── standardized_dataset.csv       # Training data with 78 features
└── model_input_template.csv       # Template for live data structure
```

## 🔄 Workflow

### Training Phase

1. Build dataset with all features
2. Create standardized dataset (78 features)
3. Train model with standardized data
4. Save standardizer for live inference

### Live Inference Phase

1. Receive live market data
2. Process with LiveDataProcessor
3. Ensure exactly 78 features
4. Feed to trained model
5. Get prediction

## ⚠️ Important Notes

### Model Input Dimension

- **Always use `standardizer.get_feature_count()`** for model input dimension
- This ensures the model expects exactly 78 features
- Never hardcode the feature count

### Feature Order

- Features are always in the same order
- Use `standardizer.get_feature_names()` to get the correct order
- Never assume feature order from raw data

### Missing Features

- If live data is missing features, they're filled with sensible defaults
- Binary features (patterns) default to 0
- Numeric features default to 0.0
- This ensures the model always gets 78 features

### Data Quality

- The standardizer automatically cleans data quality issues
- Monitor warnings about missing values, infinite values, etc.
- Use the standardized dataset for training to ensure consistency

## 🚀 Production Deployment

For production deployment:

1. **Save the standardizer** during training
2. **Load the standardizer** in production
3. **Use LiveDataProcessor** for all live data
4. **Validate input dimensions** before model inference
5. **Monitor data quality** warnings

```python
# Production code example
class TradingSystem:
    def __init__(self, model_path: str, standardizer_path: str):
        self.standardizer = DataStandardizer.load(standardizer_path)
        self.live_processor = LiveDataProcessor(self.standardizer)
        self.model = CNNLSTMModel(input_dim=self.standardizer.get_feature_count())
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, market_data: dict) -> float:
        # Process live data
        processed_data = self.live_processor.process_single_row(market_data)

        # Validate dimensions
        if processed_data.shape[1] != self.standardizer.get_feature_count():
            raise ValueError(f"Expected {self.standardizer.get_feature_count()} features, got {processed_data.shape[1]}")

        # Make prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(processed_data.values)
            prediction = self.model(input_tensor)
            return prediction.item()
```

## 📈 Benefits for Trading Systems

1. **Reliability**: No more dimension mismatch errors
2. **Consistency**: Same preprocessing for training and live data
3. **Maintainability**: Centralized feature engineering logic
4. **Scalability**: Easy to add new data sources
5. **Quality**: Automatic data cleaning and validation
6. **Reproducibility**: Deterministic feature engineering

This standardization system ensures your trading model will work reliably in production with consistent, high-quality inputs.
