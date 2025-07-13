# Enhanced CNN+LSTM Training Pipeline Guide

This guide covers the production-ready CNN+LSTM training pipeline with comprehensive monitoring, hyperparameter optimization, and evaluation capabilities.

## 🎯 Overview

The enhanced training pipeline addresses all the TODO items from the original training script:

- ✅ **MLflow/TensorBoard Integration** - Complete experiment tracking and visualization
- ✅ **Model Checkpointing and Early Stopping** - Robust model saving and training control
- ✅ **Hyperparameter Optimization Framework** - Automated hyperparameter tuning with Optuna
- ✅ **Comprehensive Training Validation Metrics** - Extensive metrics and evaluation
- ✅ **Training CLI with Argument Parsing** - Flexible command-line interface
- ✅ **Training Progress Visualization** - Advanced plotting and monitoring

## 🚀 Quick Start

### Installation

```bash
# Install enhanced training dependencies
pip install -r requirements_enhanced_training.txt

# Or install specific components
pip install mlflow tensorboard optuna torch torchvision
```

### Basic Training

```bash
# Train with default settings
python train_cnn_lstm_enhanced.py --symbols AAPL GOOGL MSFT --epochs 50

# Train with GPU acceleration
python train_cnn_lstm_enhanced.py --symbols AAPL GOOGL MSFT --gpu --epochs 100

# Train with hyperparameter optimization
python train_cnn_lstm_enhanced.py --symbols AAPL GOOGL MSFT --optimize-hyperparams --n-trials 30
```

## 📊 Features

### 1. MLflow Experiment Tracking

The enhanced pipeline automatically tracks all experiments with MLflow:

```python
# Training automatically logs to MLflow
trainer = EnhancedCNNLSTMTrainer(
    model_config=model_config,
    training_config=training_config,
    enable_mlflow=True,  # Default: True
    experiment_name="cnn_lstm_training"
)

# View experiments
mlflow ui  # Start MLflow UI
```

**Tracked Information:**
- Model hyperparameters
- Training hyperparameters
- Training metrics (loss, MAE, RMSE, R²)
- Validation metrics
- Model artifacts
- Training time and resources

### 2. TensorBoard Integration

Real-time training visualization with TensorBoard:

```python
# TensorBoard logging is enabled by default
trainer = EnhancedCNNLSTMTrainer(
    enable_tensorboard=True  # Default: True
)

# View TensorBoard
tensorboard --logdir runs/
```

**Visualizations:**
- Loss curves (train/validation)
- Learning rate scheduling
- Gradient norms
- Model architecture graph
- Training metrics over time

### 3. Hyperparameter Optimization

Automated hyperparameter tuning with Optuna:

```bash
# Run optimization
python train_cnn_lstm_enhanced.py --optimize-hyperparams --n-trials 50

# Use optimized parameters for full training
python train_cnn_lstm_enhanced.py --optimize-hyperparams --epochs 100
```

**Optimized Parameters:**
- CNN architecture (coordinated filters and kernel sizes)
- LSTM configuration (units, layers)
- Training hyperparameters (learning rate, batch size)
- Regularization (dropout, weight decay)

**CNN Architecture Optimization:**
The optimizer selects from pre-defined coordinated architectures to ensure matching filter and kernel sizes:
- 2-layer architectures: `([16, 32], [3, 3])`, `([32, 64], [3, 3])`, `([64, 128], [3, 3])`
- 3-layer architectures: `([32, 64, 128], [3, 3, 3])`, `([16, 32, 64], [3, 3, 3])`
- 4-layer architectures: `([32, 64, 128, 256], [3, 3, 3, 3])`, `([16, 32, 64, 128], [5, 5, 5, 5])`
- Mixed kernel sizes: `([16, 32, 64], [3, 5, 3])`, `([32, 64, 128], [5, 3, 5])`

### 4. Comprehensive Metrics

Enhanced evaluation with multiple metrics:

```python
# Training automatically calculates:
final_metrics = {
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "rmse": root_mean_square_error,
    "r2": r2_score,
    "explained_variance": explained_variance_score,
    "correlation": correlation_coefficient,
    "std_predictions": prediction_std,
    "std_targets": target_std,
    "mean_predictions": prediction_mean,
    "mean_targets": target_mean,
}
```

### 5. Advanced Visualization

Comprehensive training history plots:

```python
# Automatic plotting after training
trainer.plot_training_history(save_path="training_history.png")
```

**Generated Plots:**
- Loss curves (train/validation)
- MAE and RMSE progression
- Learning rate scheduling
- Gradient norm monitoring
- Train/validation loss ratio

## 🔧 Configuration

### Model Configuration

```python
model_config = {
    "cnn_filters": [32, 64, 128],      # CNN filter sizes
    "cnn_kernel_sizes": [3, 3, 3],     # CNN kernel sizes
    "lstm_units": 128,                 # LSTM hidden units
    "lstm_layers": 2,                  # Number of LSTM layers
    "dropout_rate": 0.2,               # Dropout rate
    "output_size": 1,                  # Output dimension
}
```

### Training Configuration

```python
training_config = {
    "learning_rate": 0.001,            # Initial learning rate
    "batch_size": 32,                  # Batch size
    "epochs": 100,                     # Maximum epochs
    "weight_decay": 1e-5,              # L2 regularization
    "val_split": 0.2,                  # Validation split
    "early_stopping_patience": 15,     # Early stopping patience
    "lr_patience": 5,                  # LR scheduler patience
    "max_grad_norm": 1.0,              # Gradient clipping
}
```

## 📈 Usage Examples

### Example 1: Basic Training with Monitoring

```python
from train_cnn_lstm_enhanced import EnhancedCNNLSTMTrainer, create_enhanced_model_config, create_enhanced_training_config

# Load your data
sequences, targets = load_your_data()

# Create configurations
model_config = create_enhanced_model_config()
training_config = create_enhanced_training_config()
training_config["epochs"] = 50

# Initialize trainer
trainer = EnhancedCNNLSTMTrainer(
    model_config=model_config,
    training_config=training_config,
    experiment_name="my_experiment"
)

# Train with monitoring
result = trainer.train_from_dataset(
    sequences=sequences,
    targets=targets,
    save_path="models/best_model.pth"
)

print(f"Best validation loss: {result['best_val_loss']:.6f}")
print(f"Final R² score: {result['final_metrics']['r2']:.4f}")
```

### Example 2: Hyperparameter Optimization

```python
from train_cnn_lstm_enhanced import HyperparameterOptimizer

# Load your data
sequences, targets = load_your_data()

# Run optimization
optimizer = HyperparameterOptimizer(sequences, targets, n_trials=30)
opt_result = optimizer.optimize()

print(f"Best validation loss: {opt_result['best_score']:.6f}")
print(f"Best parameters: {opt_result['best_params']}")

# Use optimized parameters for full training
best_model_config = opt_result['best_params']['model_config']
best_training_config = opt_result['best_params']['training_config']
best_training_config['epochs'] = 100  # Full training

trainer = EnhancedCNNLSTMTrainer(
    model_config=best_model_config,
    training_config=best_training_config
)

result = trainer.train_from_dataset(sequences, targets)
```

### Example 3: Custom Model Architecture

```python
# Custom model configuration
custom_model_config = {
    "cnn_filters": [64, 128, 256],     # Larger CNN
    "cnn_kernel_sizes": [5, 5, 5],     # Larger kernels
    "lstm_units": 256,                 # Larger LSTM
    "lstm_layers": 3,                  # More layers
    "dropout_rate": 0.3,               # Higher dropout
    "output_size": 1,
}

# Custom training configuration
custom_training_config = {
    "learning_rate": 0.0005,           # Lower learning rate
    "batch_size": 64,                  # Larger batch
    "epochs": 200,                     # More epochs
    "weight_decay": 1e-4,              # Higher regularization
    "val_split": 0.15,                 # Smaller validation
    "early_stopping_patience": 20,     # More patience
    "lr_patience": 8,                  # More LR patience
    "max_grad_norm": 0.5,              # Tighter gradient clipping
}

trainer = EnhancedCNNLSTMTrainer(
    model_config=custom_model_config,
    training_config=custom_training_config
)
```

## 🎛️ Command Line Interface

### Basic Options

```bash
# Required arguments
--symbols SYMBOLS [SYMBOLS ...]    # Stock symbols to include
--output-dir OUTPUT_DIR            # Output directory

# Data options
--start-date START_DATE            # Start date (default: 2020-01-01)
--end-date END_DATE                # End date (default: 2024-12-31)
--sequence-length SEQUENCE_LENGTH  # Sequence length (default: 60)
--load-dataset LOAD_DATASET        # Load existing dataset

# Training options
--epochs EPOCHS                    # Number of epochs (default: 100)
--gpu                             # Use GPU if available
--no-mlflow                       # Disable MLflow logging
--no-tensorboard                  # Disable TensorBoard logging

# Optimization options
--optimize-hyperparams            # Run hyperparameter optimization
--n-trials N_TRIALS               # Number of optimization trials (default: 50)
```

### Example Commands

```bash
# Quick test run
python train_cnn_lstm_enhanced.py \
    --symbols AAPL GOOGL \
    --epochs 10 \
    --output-dir outputs/test_run

# Full training with optimization
python train_cnn_lstm_enhanced.py \
    --symbols AAPL GOOGL MSFT TSLA AMZN \
    --epochs 100 \
    --gpu \
    --optimize-hyperparams \
    --n-trials 30 \
    --output-dir outputs/full_training

# Training with existing dataset
python train_cnn_lstm_enhanced.py \
    --load-dataset outputs/previous_run/dataset \
    --epochs 50 \
    --output-dir outputs/continued_training

# Minimal logging for production
python train_cnn_lstm_enhanced.py \
    --symbols AAPL GOOGL MSFT \
    --epochs 100 \
    --no-mlflow \
    --no-tensorboard \
    --output-dir outputs/production
```

## 📊 Output Structure

After training, the pipeline creates a comprehensive output structure:

```
outputs/enhanced_cnn_lstm_training/
├── best_model.pth                 # Best model checkpoint
├── model_config.json             # Model configuration
├── training_config.json          # Training configuration
├── training_summary.json         # Training results summary
├── training_history.png          # Training visualization
├── optimization_results.json     # Hyperparameter optimization results (if enabled)
├── dataset/                      # Built dataset (if not loaded)
│   ├── sequences.npy
│   ├── targets.npy
│   └── metadata.json
└── mlruns/                       # MLflow experiment tracking
    └── ...
```

## 🔍 Monitoring and Debugging

### MLflow UI

```bash
# Start MLflow UI to view experiments
mlflow ui --port 5000

# Access at http://localhost:5000
```

### TensorBoard

```bash
# Start TensorBoard to view training progress
tensorboard --logdir runs/ --port 6006

# Access at http://localhost:6006
```

### Log Analysis

```python
# Load training results
import json

with open("outputs/training_summary.json", "r") as f:
    results = json.load(f)

print(f"Best validation loss: {results['best_val_loss']}")
print(f"Training time: {results['training_time'] / 60:.1f} minutes")
print(f"Final metrics: {results['final_metrics']}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/integration/test_enhanced_cnn_lstm_training.py -v

# Run specific test
pytest tests/integration/test_enhanced_cnn_lstm_training.py::TestEnhancedCNNLSTMTraining::test_training_workflow -v

# Run with coverage
pytest tests/integration/test_enhanced_cnn_lstm_training.py --cov=train_cnn_lstm_enhanced --cov-report=html
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements_enhanced_training.txt
   ```

2. **GPU Memory Issues**
   ```python
   # Reduce batch size
   training_config["batch_size"] = 16
   
   # Use gradient accumulation
   training_config["gradient_accumulation_steps"] = 4
   ```

3. **MLflow Connection Issues**
   ```bash
   # Disable MLflow for local testing
   python train_cnn_lstm_enhanced.py --no-mlflow
   ```

4. **Slow Training**
   ```python
   # Use smaller model
   model_config["lstm_units"] = 64
   model_config["cnn_filters"] = [16, 32]
   
   # Reduce sequence length
   --sequence-length 30
   ```

### Performance Optimization

1. **GPU Utilization**
   ```python
   # Ensure GPU is being used
   trainer = EnhancedCNNLSTMTrainer(device="cuda")
   ```

2. **Memory Efficiency**
   ```python
   # Use mixed precision training
   training_config["use_amp"] = True
   ```

3. **Data Loading**
   ```python
   # Increase number of workers
   training_config["num_workers"] = 4
   ```

## 📚 Next Steps

After training your CNN+LSTM model:

1. **Model Evaluation**: Use the trained model for backtesting
2. **RL Integration**: Use the model as a feature extractor for RL agents
3. **Production Deployment**: Deploy the model for real-time inference
4. **Model Monitoring**: Set up continuous model performance monitoring

## 🤝 Contributing

To contribute to the enhanced training pipeline:

1. Add new metrics to the evaluation framework
2. Implement additional hyperparameter optimization strategies
3. Add support for new model architectures
4. Improve visualization capabilities
5. Add distributed training support

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.