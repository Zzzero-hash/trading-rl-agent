# Examples

This page provides working examples for common use cases with the Trading RL Agent system.

## 📊 **Data Pipeline Examples**

### **Basic Data Loading**

```python
from trading_rl_agent import ConfigManager
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder

# Initialize configuration
config = ConfigManager("configs/development.yaml")

# Build dataset
builder = RobustDatasetBuilder()
dataset = builder.build_dataset(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2023-01-01",
    end_date="2024-01-01"
)

print(f"Dataset shape: {dataset.shape}")
print(f"Columns: {dataset.columns.tolist()}")
```

### **Custom Feature Engineering**

```python
from trading_rl_agent.data.features import FeatureEngineer
import pandas as pd

# Sample data
data = pd.DataFrame({
    'close': [100, 101, 99, 102, 98, 103],
    'volume': [1000, 1100, 900, 1200, 800, 1300]
})

# Initialize feature engineer
engineer = FeatureEngineer()

# Add technical indicators
features = engineer.add_technical_indicators(
    data=data,
    indicators=["sma", "ema", "rsi", "macd"]
)

# Add temporal features
features = engineer.add_temporal_features(features)

print(f"Enhanced features: {features.columns.tolist()}")
```

### **Multi-Source Data Loading**

```python
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder, DatasetConfig

# Configure multiple data sources
config = DatasetConfig(
    symbols=["AAPL", "GOOGL"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    data_sources=["yfinance", "alpha_vantage"],
    features=["sma", "ema", "rsi", "macd"],
    window_size=50,
    normalize=True
)

# Build dataset
builder = RobustDatasetBuilder()
dataset = builder.build_dataset(config)

print(f"Multi-source dataset: {dataset.shape}")
```

## 🧠 **CNN+LSTM Model Examples**

### **Basic Model Creation**

```python
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
import torch

# Initialize model
model = CNNLSTMModel(
    input_dim=50,  # Number of features
    config={
        "cnn_filters": [32, 64, 128],
        "lstm_units": 256,
        "dropout_rate": 0.2,
        "uncertainty_estimation": True
    }
)

# Create sample input
batch_size, sequence_length, features = 32, 50, 50
x = torch.randn(batch_size, sequence_length, features)

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")
```

### **Model Training**

```python
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
import torch
import torch.nn as nn
import torch.optim as optim

# Initialize model
model = CNNLSTMModel(input_dim=50, config={
    "cnn_filters": [32, 64],
    "lstm_units": 128,
    "dropout_rate": 0.1
})

# Setup training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Sample training data
x = torch.randn(64, 50, 50)  # batch, sequence, features
y = torch.randn(64, 1)       # target values

# Training step
optimizer.zero_grad()
output = model(x)
loss = criterion(output, y)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
```

## ⚙️ **Configuration Examples**

### **Basic Configuration**

```python
from trading_rl_agent import ConfigManager

# Load configuration
config = ConfigManager("configs/development.yaml")

# Access configuration values
print(f"Environment: {config.environment}")
print(f"Debug mode: {config.debug}")
print(f"Data sources: {config.data.data_sources}")
```

### **Custom Configuration**

```python
from trading_rl_agent import ConfigManager

# Create custom configuration
custom_config = {
    "environment": "development",
    "debug": True,
    "data": {
        "data_sources": {
            "primary": "yfinance",
            "backup": "alpha_vantage"
        },
        "feature_window": 50,
        "symbols": ["AAPL", "GOOGL", "MSFT"]
    },
    "training": {
        "cnn_lstm": {
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        }
    }
}

# Initialize with custom config
config = ConfigManager(config_dict=custom_config)
print(f"Custom config loaded: {config.data.symbols}")
```

## 🧪 **Testing Examples**

### **Unit Test Example**

```python
import pytest
from trading_rl_agent.data.features import FeatureEngineer
import pandas as pd

def test_feature_engineering():
    # Setup test data
    data = pd.DataFrame({
        'close': [100, 101, 99, 102, 98, 103],
        'volume': [1000, 1100, 900, 1200, 800, 1300]
    })

    # Initialize feature engineer
    engineer = FeatureEngineer()

    # Add features
    features = engineer.add_technical_indicators(
        data=data,
        indicators=["sma", "rsi"]
    )

    # Assertions
    assert "sma" in features.columns
    assert "rsi" in features.columns
    assert len(features) == len(data)
    assert not features.isnull().all().any()

# Run with: pytest test_example.py -v
```

### **Integration Test Example**

```python
import pytest
from trading_rl_agent import ConfigManager
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder

def test_data_pipeline_integration():
    # Initialize components
    config = ConfigManager("configs/development.yaml")
    builder = RobustDatasetBuilder()

    # Build dataset
    dataset = builder.build_dataset(
        symbols=["AAPL"],
        start_date="2023-01-01",
        end_date="2023-01-31"
    )

    # Assertions
    assert not dataset.empty
    assert "close" in dataset.columns
    assert len(dataset) > 0
    assert not dataset.isnull().all().any()

# Run with: pytest test_integration.py -v
```

## 🖥️ **CLI Examples**

### **Data Generation**

```bash
# Generate synthetic data
python cli.py generate-data configs/finrl_synthetic_data.yaml --synthetic

# Generate real data
python cli.py generate-data configs/finrl_real_data.yaml
```

### **Model Training**

```bash
# Train CNN+LSTM model
python cli.py train cnn-lstm configs/cnn_lstm_training.yaml

# Train with custom parameters
python cli.py train cnn-lstm configs/cnn_lstm_training.yaml --num-workers 4
```

### **Backtesting**

```bash
# Simple backtest
python cli.py backtest data/price_data.csv --policy "lambda p: 'buy' if p > 100 else 'sell'"

# Backtest with slippage
python cli.py backtest data/price_data.csv \
    --policy "lambda p: 'buy' if p > 100 else 'sell'" \
    --slippage-pct 0.001 \
    --latency 0.1
```

### **Model Evaluation**

```bash
# Evaluate trained agent
python cli.py evaluate data/test_data.csv checkpoints/agent.zip --agent sac

# Evaluate with custom parameters
python cli.py evaluate data/test_data.csv checkpoints/agent.zip \
    --agent sac \
    --window-size 100 \
    --output results/evaluation.json
```

## 📈 **Advanced Examples**

### **Custom Feature Engineering Pipeline**

```python
from trading_rl_agent.data.features import FeatureEngineer
from trading_rl_agent.data.preprocessing import DataPreprocessor
import pandas as pd

class CustomFeaturePipeline:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor()

    def process(self, data):
        # Preprocess data
        clean_data = self.preprocessor.clean_data(data)

        # Add technical indicators
        features = self.feature_engineer.add_technical_indicators(
            clean_data,
            indicators=["sma", "ema", "rsi", "macd", "bollinger_bands"]
        )

        # Add custom features
        features = self.add_custom_features(features)

        # Normalize features
        normalized = self.preprocessor.normalize_features(features)

        return normalized

    def add_custom_features(self, data):
        # Add your custom features here
        data['price_momentum'] = data['close'].pct_change(5)
        data['volume_momentum'] = data['volume'].pct_change(5)
        return data

# Usage
pipeline = CustomFeaturePipeline()
processed_data = pipeline.process(raw_data)
```

### **Model Ensemble**

```python
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel
import torch
import torch.nn as nn

class ModelEnsemble(nn.Module):
    def __init__(self, input_dim, num_models=3):
        super().__init__()
        self.models = nn.ModuleList([
            CNNLSTMModel(input_dim, config={
                "cnn_filters": [32, 64],
                "lstm_units": 128,
                "dropout_rate": 0.1
            }) for _ in range(num_models)
        ])

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Usage
ensemble = ModelEnsemble(input_dim=50)
x = torch.randn(32, 50, 50)
output = ensemble(x)
print(f"Ensemble output shape: {output.shape}")
```

## 🎯 **Next Steps**

These examples demonstrate the core functionality of the Trading RL Agent system. To explore further:

1. **Experiment with different configurations** in the `configs/` directory
2. **Try different feature combinations** in the feature engineering pipeline
3. **Modify model architectures** in the CNN+LSTM models
4. **Add custom components** following the established patterns
5. **Run the test suite** to verify your implementations

For more detailed information, check out the [API Reference](../src/) and [Development Guide](DEVELOPMENT_GUIDE.md).

---

**Happy coding! 🚀**
