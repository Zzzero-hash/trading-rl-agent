# Jupyter Notebook Best Practices for ML Training

## 📚 Overview

This guide provides best practices for using Jupyter notebooks in machine learning training workflows, particularly for hyperparameter optimization and model experimentation. This project includes automated tools for managing experiment outputs and maintaining clean development workflows.

## 🎯 Development Workflow

### 1. Notebook Structure
```
# Notebook Title and Objectives
## Step 1: Environment Setup
## Step 2: Data Loading
## Step 3: Model Architecture
## Step 4: Training Pipeline
## Step 5: Evaluation & Results
## Step 6: Cleanup & Export
```

### 2. Code Organization
- **Import cells**: Group all imports at the beginning
- **Configuration cells**: Define hyperparameters and settings
- **Function definitions**: Keep reusable functions in separate cells
- **Experiment cells**: One major experiment per cell
- **Output cells**: Clear visualizations and results

### 3. Documentation Standards
- Use markdown cells to explain each section
- Document experiment parameters and expected outcomes
- Include links to relevant papers or documentation
- Add timestamps for long-running experiments

## 🔧 Technical Best Practices

### Memory Management
```python
# Clear variables between experiments
def clear_memory():
    """Clear large variables to free memory."""
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Use context managers for large operations
with torch.no_grad():
    # inference code here
    pass
```

### Progress Tracking
```python
from tqdm.notebook import tqdm
import time

# For long-running experiments
for epoch in tqdm(range(num_epochs), desc="Training"):
    # training code
    if epoch % 10 == 0:
        tqdm.write(f"Epoch {epoch}: Loss = {loss:.4f}")
```

### Resource Monitoring
```python
# Monitor GPU usage
def check_gpu_usage():
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Monitor disk usage in output directories
def check_disk_usage():
    import shutil
    dirs = ["optimization_results", "ray_results", "models"]
    for d in dirs:
        if Path(d).exists():
            size = shutil.disk_usage(d).used / (1024**3)
            print(f"{d}: {size:.2f} GB")
```

## 🚀 Hyperparameter Optimization

### Experiment Configuration
```python
# Define experiment metadata
experiment_config = {
    "name": "cnn_lstm_hparam_v2",
    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    "objective": "minimize validation loss for price prediction",
    "search_space": {
        "learning_rate": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "lstm_units": tune.choice([32, 64, 128])
    },
    "notes": "Testing impact of LSTM size on convergence"
}
```

### Results Tracking
```python
# Save experiment results
def save_experiment_results(analysis, config):
    results_dir = Path("optimization_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = config["timestamp"]
    
    # Save best configuration
    best_config = analysis.get_best_config()
    with open(results_dir / f"best_config_{timestamp}.json", "w") as f:
        json.dump(best_config, f, indent=2)
    
    # Save experiment metadata
    with open(results_dir / f"experiment_{timestamp}.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Save trial results as CSV
    df = analysis.dataframe()
    df.to_csv(results_dir / f"trials_{timestamp}.csv", index=False)
```

## 📊 Visualization Best Practices

### Interactive Plots
```python
import plotly.express as px
import plotly.graph_objects as go

# Create interactive training curves
def plot_training_history(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history['train_loss'], 
        name='Training Loss',
        mode='lines'
    ))
    fig.add_trace(go.Scatter(
        y=history['val_loss'], 
        name='Validation Loss',
        mode='lines'
    ))
    fig.update_layout(
        title="Training Progress",
        xaxis_title="Epoch",
        yaxis_title="Loss"
    )
    return fig
```

### Results Comparison
```python
# Compare multiple experiments
def compare_experiments(results_dir):
    experiments = []
    for config_file in Path(results_dir).glob("best_config_*.json"):
        with open(config_file) as f:
            config = json.load(f)
            experiments.append(config)
    
    df = pd.DataFrame(experiments)
    return df.sort_values('val_loss').head(10)
```

## 🧹 Cleanup and Maintenance

### Pre-commit Checklist
- [ ] Clear all cell outputs
- [ ] Remove temporary variables
- [ ] Check for hardcoded paths
- [ ] Verify external dependencies
- [ ] Update documentation strings

### Automated Cleanup Tools

The project includes automated cleanup tools to manage experiment outputs:

```bash
# Check storage usage
python scripts/cleanup_experiments.py --status-only

# Dry run to see what would be cleaned
python scripts/cleanup_experiments.py --dry-run --all

# Clean up old results (keeps last 7 days)
python scripts/cleanup_experiments.py --all

# Archive important results before cleanup
python scripts/cleanup_experiments.py --archive --all
```

### Output Management
```python
# Clear notebook outputs before saving
def clear_all_outputs():
    from IPython.core.display import display, Javascript
    display(Javascript('IPython.notebook.clear_all_output();'))

# Save important results before cleanup
def archive_results():
    important_files = [
        "best_config.json",
        "final_model.pth", 
        "training_plots.png"
    ]
    
    archive_dir = Path("results_archive")
    archive_dir.mkdir(exist_ok=True)
    
    for file in important_files:
        if Path(file).exists():
            shutil.copy2(file, archive_dir)
```

### Storage Management Guidelines

- **optimization_results/**: Keep < 500MB, archive older results
- **ray_results/**: Keep < 1GB, clean up after experiments  
- **Notebook outputs**: Always clear before commits
- **Python cache**: Clean regularly with cleanup script

See `docs/EXPERIMENT_OUTPUTS_MANAGEMENT.md` for detailed cleanup procedures.

## 🔄 Reproducibility

### Environment Setup
```python
# Set random seeds
def set_random_seeds(seed=42):
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Log environment information
def log_environment():
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }
    return info
```

### Configuration Versioning
```python
# Version control for configurations
def save_config_version(config, version="v1"):
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    
    filename = f"experiment_config_{version}.json"
    with open(config_dir / filename, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved as {filename}")
```

## 🚨 Common Pitfalls to Avoid

### Memory Issues
- Don't accumulate large objects in loops
- Clear variables after heavy computations
- Use `del` and `gc.collect()` strategically
- Monitor GPU memory usage

### Reproducibility Issues
- Always set random seeds
- Pin dependency versions
- Document hardware specifications
- Save model architectures with weights

### Performance Issues
- Use appropriate batch sizes for your hardware
- Implement early stopping
- Profile code to identify bottlenecks
- Use mixed precision training when appropriate

### Organization Issues
- Don't mix exploration and production code
- Keep notebooks focused on single objectives
- Extract reusable code to modules
- Use meaningful variable names

## 📈 Advanced Techniques

### Distributed Training Integration
```python
# Ray Tune integration
def distributed_training_function(config):
    # Your training code here
    model = create_model(config)
    train_loader, val_loader = create_data_loaders(config)
    
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate_epoch(model, val_loader)
        
        # Report to Ray Tune
        tune.report(
            train_loss=train_loss,
            val_loss=val_loss,
            epoch=epoch
        )
```

### Experiment Tracking
```python
# Integration with experiment tracking tools
import wandb  # or mlflow, tensorboard

def setup_experiment_tracking(config):
    wandb.init(
        project="trading-rl-agent",
        config=config,
        name=f"experiment_{config['timestamp']}"
    )
    
def log_metrics(metrics, step):
    wandb.log(metrics, step=step)
```

This guide should be followed alongside the project's specific requirements and constraints.
