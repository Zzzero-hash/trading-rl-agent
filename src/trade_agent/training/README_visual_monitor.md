# CNN-LSTM Visual Training Monitor

A comprehensive real-time visual monitoring system for CNN-LSTM training that automatically starts when training begins. Provides live dashboards, Optuna trial tracking, and comprehensive training analytics.

## ✨ Key Features

- **🎬 Automatic Activation**: Visual monitoring starts automatically when training begins
- **📊 Real-time Dashboards**: Live training metrics, loss curves, and progress bars
- **🔬 Optuna Integration**: Trial progress tracking and hyperparameter analysis
- **📈 Interactive Plots**: Plotly-based interactive charts and matplotlib fallbacks
- **💾 Auto-save**: All visualizations are automatically saved to disk
- **🎯 Multi-format**: Supports both synthetic data and CSV datasets

## 🚀 Quick Start

### Basic Training with Visual Monitor

```python
from trade_agent.training.train_cnn_lstm_enhanced import EnhancedCNNLSTMTrainer

# Create trainer with visual monitoring (enabled by default)
trainer = EnhancedCNNLSTMTrainer(
    model_config=model_config,
    training_config=training_config,
    enable_visual_monitor=True  # 🎨 This starts the visual monitor!
)

# Start training - visual monitor activates automatically!
results = trainer.train_from_dataset(sequences, targets)
```

### Optuna Optimization with Visual Tracking

```python
from trade_agent.training.train_cnn_lstm_enhanced import HyperparameterOptimizer

# Create optimizer with visual monitoring
optimizer = HyperparameterOptimizer(
    sequences=sequences,
    targets=targets,
    n_trials=50,
    enable_visual_monitor=True  # 🔬 Tracks all trials visually!
)

# Run optimization - trial progress visualized in real-time!
results = optimizer.optimize()
```

## 🖥️ CLI Commands

### Train with Visual Monitoring

```bash
# Basic training with automatic visual monitoring
python -m trade_agent.cli_train cnn-lstm-visual

# With custom parameters
python -m trade_agent.cli_train cnn-lstm-visual --epochs 50 --lr 0.001 --batch 64

# With your own CSV data
python -m trade_agent.cli_train cnn-lstm-visual --data ./my_data.csv

# Disable visual monitoring
python -m trade_agent.cli_train cnn-lstm-visual --no-visual
```

### Hyperparameter Optimization

```bash
# Optimize hyperparameters with visual trial tracking
python -m trade_agent.cli_train cnn-lstm-optimize --trials 30

# With timeout and custom data
python -m trade_agent.cli_train cnn-lstm-optimize --data ./data.csv --trials 50 --timeout 3600
```

### Run Demonstrations

```bash
# Complete visual monitoring demo suite
python -m trade_agent.cli_train visual-demo
```

## 📊 Visual Components

### Training Dashboard

- **Loss Curves**: Real-time training and validation loss
- **Metrics Tracking**: MAE, RMSE, R² scores
- **Learning Rate**: Schedule visualization
- **Progress Bar**: Current epoch progress
- **Model Info**: Architecture details and parameters

### Optuna Trial Monitor

- **Trial Progress**: Best score progression over trials
- **Parameter Importance**: Which hyperparameters matter most
- **Trial States**: Success/failure distribution
- **Best Trials**: Comparison of top-performing configurations
- **Interactive Analysis**: Plotly-based exploration tools

## 📁 Output Files

All visualizations are automatically saved to `./training_visualizations/`:

```
training_visualizations/
├── training_dashboard_*.html          # Interactive training dashboards
├── optuna_dashboard_*.html            # Optuna trial visualizations
├── training_summary_*/                # Complete training summaries
│   ├── training_history.csv          # Raw metrics data
│   ├── model_config.csv              # Model configuration
│   ├── final_summary.html/.png       # Summary visualizations
└── ...
```

## 🎛️ Configuration Options

### TrainingVisualMonitor Parameters

```python
monitor = TrainingVisualMonitor(
    save_dir="./my_visualizations",     # Output directory
    use_plotly=True,                    # Use interactive Plotly plots
    update_interval=1.0,                # Update frequency (seconds)
    figsize=(15, 10)                    # Matplotlib figure size
)
```

### Integration Options

```python
# Enhanced trainer with visual monitoring
trainer = EnhancedCNNLSTMTrainer(
    model_config=config,
    training_config=training_config,
    enable_visual_monitor=True,         # Enable visual monitoring
    enable_mlflow=False,                # MLflow tracking
    enable_tensorboard=False            # TensorBoard logging
)

# Hyperparameter optimizer with visual tracking
optimizer = HyperparameterOptimizer(
    sequences=data,
    targets=labels,
    n_trials=100,
    enable_visual_monitor=True          # Enable trial visualization
)
```

## 🔧 Advanced Usage

### Custom Visual Monitor Setup

```python
from trade_agent.training.visual_monitor import TrainingVisualMonitor

# Create custom monitor
monitor = TrainingVisualMonitor(
    save_dir="./custom_viz",
    use_plotly=True,
    update_interval=0.5
)

# Start monitoring manually
monitor.start_training_monitor(trainer_instance)

# Save summary when done
monitor.save_training_summary(trainer_instance)
monitor.stop_monitoring()
```

### Programmatic Demo

```python
from trade_agent.training.auto_visual_demo import run_all_demos

# Run comprehensive demonstration
success = run_all_demos()

# Run specific demos
from trade_agent.training.auto_visual_demo import (
    demo_basic_training_with_visual_monitor,
    demo_optuna_optimization_with_visual_monitor,
    demo_csv_data_training
)

demo_basic_training_with_visual_monitor()
```

## 🛠️ Dependencies

The visual monitor requires:

- `matplotlib >= 3.7.0` (core plotting)
- `seaborn >= 0.12.0` (styling)
- `plotly >= 5.17.0` (interactive plots)
- `pandas >= 2.0.0` (data handling)
- `numpy >= 1.24.0` (numerical operations)

All dependencies are included in the main `requirements.txt`.

## 🎯 Best Practices

1. **Always Enable by Default**: Visual monitoring provides valuable insights
2. **Monitor Resource Usage**: Visual updates can consume memory during long training
3. **Check Output Directory**: Visualizations accumulate over multiple runs
4. **Use Interactive Mode**: Plotly charts provide better exploration capabilities
5. **Save Summaries**: Call `save_training_summary()` for comprehensive reports

## 🔍 Troubleshooting

### Common Issues

**Q: Visual monitor doesn't start automatically**

```python
# Check if properly enabled
trainer = EnhancedCNNLSTMTrainer(..., enable_visual_monitor=True)
```

**Q: Plotly plots not appearing**

```python
# Check Plotly installation
pip install plotly>=5.17.0

# Fallback to matplotlib
monitor = TrainingVisualMonitor(use_plotly=False)
```

**Q: Too many visualization files**

```bash
# Clean up old visualizations
rm -rf ./training_visualizations/
```

**Q: Memory usage during long training**

```python
# Reduce update frequency
monitor = TrainingVisualMonitor(update_interval=5.0)
```

## 📝 Examples

See `auto_visual_demo.py` for complete working examples of:

- Basic training with visual monitoring
- Optuna optimization with trial tracking
- CSV data processing with visualizations
- Custom configuration and setup

## 🎊 Demo Mode

Run the complete demonstration suite:

```bash
cd src/trade_agent/training
python auto_visual_demo.py
```

This will show all visual monitoring features with synthetic data, creating comprehensive examples of the system in action.
