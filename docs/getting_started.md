# Getting Started

Welcome to the Trading RL Agent! This guide will help you get up and running with the hybrid reinforcement learning trading system.

## 🚀 **Quick Installation**

### **Prerequisites**

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### **Installation Steps**

```bash
# Clone the repository
git clone https://github.com/your-org/trading-rl-agent.git
cd trading-rl-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development (optional)
pip install -r requirements.dev.txt
```

## 🧪 **First Steps**

### **1. Basic Data Loading**

Start by loading some market data:

```python
from trading_rl_agent import ConfigManager
from trading_rl_agent.data.robust_dataset_builder import RobustDatasetBuilder

# Initialize configuration
config = ConfigManager("configs/development.yaml")

# Build dataset with features
builder = RobustDatasetBuilder()
dataset = builder.build_dataset(
    symbols=["AAPL", "GOOGL", "MSFT"],
    start_date="2023-01-01",
    end_date="2024-01-01"
)

print(f"Dataset shape: {dataset.shape}")
print(f"Features: {dataset.columns.tolist()}")
```

### **2. Using the CLI**

The project provides a unified CLI interface:

```bash
# Generate synthetic data
python cli.py generate-data configs/finrl_synthetic_data.yaml --synthetic

# Train CNN+LSTM model
python cli.py train cnn-lstm configs/cnn_lstm_training.yaml

# Run backtest
python cli.py backtest data/price_data.csv --policy "lambda p: 'buy' if p > 100 else 'sell'"
```

### **3. Feature Engineering**

Explore the comprehensive feature engineering capabilities:

```python
from trading_rl_agent.data.features import FeatureEngineer

# Initialize feature engineer
engineer = FeatureEngineer()

# Add technical indicators
features = engineer.add_technical_indicators(
    data=dataset,
    indicators=["sma", "ema", "rsi", "macd", "bollinger_bands"]
)

# Add temporal features
features = engineer.add_temporal_features(features)

print(f"Enhanced dataset shape: {features.shape}")
```

## 📊 **Configuration**

The system uses YAML-based configuration files:

```yaml
# configs/development.yaml
environment: development
debug: true

data:
  data_sources:
    primary: yfinance
    backup: alpha_vantage
  feature_window: 50
  symbols: ["AAPL", "GOOGL", "MSFT"]

training:
  cnn_lstm:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    early_stopping_patience: 10
```

## 🧠 **CNN+LSTM Models**

Work with the hybrid neural network models:

```python
from trading_rl_agent.models.cnn_lstm import CNNLSTMModel

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

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 🧪 **Testing Your Setup**

Run the test suite to verify your installation:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit          # Unit tests
pytest -m integration   # Integration tests

# Check code quality
ruff check src/ tests/
mypy src/
```

## 📚 **Next Steps**

### **For Data Scientists**
- Explore the [Feature Engineering](../src/trading_rl_agent/features/) module
- Study the [CNN+LSTM Models](../src/trading_rl_agent/models/) architecture
- Review the [Data Pipeline](../src/trading_rl_agent/data/) implementation

### **For Developers**
- Check out the [Development Guide](DEVELOPMENT_GUIDE.md)
- Review the [Contributing Guide](../CONTRIBUTING.md)
- Explore the [Test Suite](../tests/)

### **For Researchers**
- Examine the [Model Architectures](../src/trading_rl_agent/models/)
- Review the [Training Pipeline](../src/training/)
- Study the [Evaluation Framework](../src/evaluation/)

## 🆘 **Troubleshooting**

### **Common Issues**

**Import Errors**: Make sure you're in the virtual environment and have installed all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Configuration Errors**: Check that your config files are valid YAML:
```bash
python -c "import yaml; yaml.safe_load(open('configs/development.yaml'))"
```

**Data Loading Issues**: Verify your data sources are accessible:
```python
import yfinance as yf
data = yf.download("AAPL", start="2023-01-01", end="2024-01-01")
print(data.head())
```

### **Getting Help**

- **Documentation**: Check the [API Reference](../src/) for detailed information
- **Issues**: Report bugs on [GitHub Issues](https://github.com/your-org/trading-rl-agent/issues)
- **Examples**: See [examples.md](examples.md) for working code examples

## 🎯 **What's Next?**

Now that you're set up, you can:

1. **Explore the Data Pipeline**: Build custom datasets with different features
2. **Train CNN+LSTM Models**: Experiment with different architectures
3. **Develop RL Agents**: Implement and train reinforcement learning agents
4. **Contribute**: Help improve the project by contributing code or documentation

Check out the [TODO.md](../TODO.md) file to see what features are planned and in progress.

---

**Happy trading! 🚀**
