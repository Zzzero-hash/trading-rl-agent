# Development Guide

## 📚 ML Workflow Best Practices

### Notebook Organization

```markdown
# Notebook Title and Objectives

## 1. Environment Setup

## 2. Data Loading

## 3. Model Training

## 4. Evaluation & Results

## 5. Cleanup & Export
```

### Memory Management

```python
# Clear GPU memory between experiments
torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Monitor resource usage
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Experiment Tracking

```python
# Save experiment results
experiment_config = {
    "name": "cnn_lstm_v2",
    "notes": "Testing LSTM size impact"
}

# Track with timestamps
results = {
    "timestamp": datetime.now().isoformat(),
    "config": experiment_config,
    "metrics": metrics
}
```

## 🧹 Output Management

### Directory Structure

```bash
optimization_results/     # Keep best configs only
ray_results/             # Clean up old trials
data/                   # Archive old datasets
*.ipynb                 # Always clear outputs before commit
```

### Cleanup Commands

```bash
# Clear notebook outputs
find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} +

# Clean experiment outputs
find optimization_results/ -name "hparam_opt_*" -type d -mtime +7 -exec rm -rf {} +
find ray_results/ -name "*_hparam_*" -type d -mtime +7 -exec rm -rf {} +

# Archive important results
mkdir -p archive/$(date +%Y%m%d)
cp optimization_results/best_*.json archive/$(date +%Y%m%d)/
```

### Git Pre-commit Setup

```bash
# Install pre-commit hooks
pre-commit install

# Manual cleanup before commit
jupyter nbconvert --clear-output --inplace *.ipynb
```

## 🔧 Development Tools

### Required Extensions (VS Code)

- Python
- Pylance
- Black Formatter
- GitLens

### Configuration

```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true
}
```

## 📊 Performance Guidelines

- Clear outputs between notebook runs
- Archive results weekly
- Keep optimization_results/ < 500MB
- Monitor GPU memory usage
- Use progress bars for long operations

## 🚀 Optimization Workflow

```python
# CNN-LSTM optimization
from src.optimization.cnn_lstm_optimization import optimize_cnn_lstm
results = optimize_cnn_lstm(features, targets, num_samples=20)

# RL optimization
from src.optimization.rl_optimization import optimize_sac_hyperparams
results = optimize_sac_hyperparams(env_config, num_samples=10)
```

## ✅ Progress Checklist

The following tasks are tracked from the codebase TODOs:

- [ ] Migrate training scripts into `src/training` and integrate Ray Tune sweeps.
- [ ] Provide sentiment and risk scoring APIs in `src/nlp`.
