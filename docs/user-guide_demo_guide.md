# Trading RL Agent - Demo Guide

This directory contains demo scripts and commands to showcase the capabilities of the Trading RL Agent software.

## 🚀 Quick Start

### Option 1: Quick Demo (Recommended for first-time users)

```bash
./quick_demo.sh
```

This runs a focused 5-minute demo showing the key capabilities.

### Option 2: Comprehensive Demo (Full feature showcase)

```bash
./demo_showcase.sh
```

This runs a complete demo that takes 15-30 minutes and shows all features.

### Option 3: Individual Commands

See `demo_commands.md` for individual command examples you can run manually.

## 📁 Demo Files

- **`quick_demo.sh`** - Fast demo showing essential features
- **`demo_showcase.sh`** - Comprehensive demo with all capabilities
- **`demo_commands.md`** - Individual command examples and workflows
- **`DEMO_README.md`** - This file

## 🎯 What the Demos Show

### Quick Demo (5 minutes)

1. ✅ System information and health check
2. ✅ CLI interface overview
3. ✅ Data pipeline demonstration
4. ✅ Feature engineering capabilities
5. ✅ Configuration management
6. ✅ Summary of all capabilities

### Comprehensive Demo (15-30 minutes)

1. ✅ System information and health check
2. ✅ Complete data pipeline (download, process, build datasets)
3. ✅ Feature engineering with market pattern generation
4. ✅ CNN+LSTM model training (demo version)
5. ✅ RL agent training (SAC, demo version)
6. ✅ Model evaluation and metrics
7. ✅ Backtesting with trained models
8. ✅ Risk management and scenario analysis
9. ✅ Portfolio management and ensemble trading
10. ✅ Configuration management examples
11. ✅ Performance monitoring and testing
12. ✅ CLI interface exploration
13. ✅ Complete workflow demonstration

## 🛠️ Prerequisites

Before running the demos, ensure you have:

1. **Python 3.9+** installed
2. **Dependencies** installed:
   ```bash
   pip install -r requirements.txt
   ```
3. **Working directory**: Run from the project root directory
4. **Internet connection**: For downloading market data

## 📊 Demo Outputs

The demos will create:

- **`demo_outputs/`** - All generated files and results
  - `data/` - Downloaded and processed market data
  - `models/` - Trained CNN+LSTM and RL models
  - `backtest/` - Backtesting results and reports
  - `evaluation/` - Model evaluation metrics
  - `logs/` - System logs and monitoring data

- **`*.png`** - Generated visualizations (market patterns, performance charts)

## 🎯 Key Features Demonstrated

### Core Capabilities

- **Hybrid AI Models**: CNN+LSTM for pattern recognition + RL for decision optimization
- **Data Pipeline**: Multi-source data ingestion with parallel processing (10-50x speedup)
- **Feature Engineering**: 150+ technical indicators with robust implementation
- **Risk Management**: VaR, CVaR, position sizing, and portfolio optimization
- **Real-time Processing**: Live data feeds and sentiment analysis

### Infrastructure

- **Unified CLI**: Single command interface for all operations
- **Configuration Management**: YAML-based with validation
- **Logging & Monitoring**: Structured logging with MLflow/TensorBoard
- **Testing**: Comprehensive test suite with pytest
- **Docker Support**: Containerized deployment ready

### Performance Optimizations

- **Parallel Data Fetching**: Ray-based parallel processing
- **Mixed Precision Training**: 2-3x faster training with 30-50% memory reduction
- **Memory-Mapped Datasets**: 60-80% memory reduction for large datasets
- **Advanced LR Scheduling**: 1.5-2x faster convergence

## 🔧 Customization

### Modify Demo Parameters

Edit the demo scripts to:

- Change symbols (e.g., from AAPL to TSLA)
- Adjust time periods
- Modify training parameters
- Add custom configurations

### Add Your Own Data

```bash
# Use your own data files
python main.py data prepare --input-path your_data.csv --output-dir custom_output/
```

### Custom Configurations

```bash
# Use custom config files
python main.py train cnn-lstm --config your_config.yaml
```

## 🚨 Troubleshooting

### Common Issues

1. **Permission denied**: Make scripts executable

   ```bash
   chmod +x *.sh
   ```

2. **Module not found**: Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

3. **Data download fails**: Check internet connection and API limits

4. **Training fails**: Ensure sufficient disk space and memory

### Getting Help

- **Documentation**: See `README.md` and `README_CLI_USAGE.md`
- **Examples**: Check the `examples/` directory
- **Issues**: Review error messages and logs in `demo_outputs/logs/`

## 📈 Next Steps

After running the demos:

1. **Review outputs**: Examine generated models, data, and results
2. **Customize**: Modify configurations for your use case
3. **Scale up**: Increase training epochs and data size
4. **Paper trading**: Start with paper trading before live trading
5. **Production**: Deploy with proper risk management

## 🎉 Success Metrics

A successful demo should show:

- ✅ All commands execute without errors
- ✅ Data is downloaded and processed
- ✅ Models are trained (even if small)
- ✅ Backtesting produces results
- ✅ Visualizations are generated
- ✅ System information is displayed correctly

## 📚 Additional Resources

- **Main Documentation**: `README.md`
- **CLI Usage**: `README_CLI_USAGE.md`
- **Contributing**: `CONTRIBUTING.md`
- **Project Status**: `PROJECT_STATUS.md`
- **Example Scripts**: `examples/` directory
