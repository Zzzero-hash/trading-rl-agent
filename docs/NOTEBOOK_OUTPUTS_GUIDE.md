# Jupyter Notebook Outputs Management Guide

## 🎯 Overview

This guide outlines best practices for managing outputs from Jupyter notebooks in the trading RL agent project, particularly for hyperparameter optimization and model training sessions.

## 📁 Output Directory Structure

```
trading-rl-agent/
├── optimization_results/          # Hyperparameter optimization results
│   ├── best_*_config_*.json       # Best configurations found
│   ├── *_hparam_results_*.json    # Detailed results with metrics
│   ├── *_trials_*.csv             # Trial data for analysis
│   └── hparam_opt_*/              # Individual experiment directories
├── ray_results/                   # Ray Tune experiment outputs
│   └── experiment_name_*/         # Timestamped experiment folders
├── models/                        # Trained model checkpoints
│   ├── *.pth                      # PyTorch model files
│   ├── *.pkl                      # Pickled objects (scalers, etc.)
│   └── config.json               # Model configuration
├── logs/                          # Training and execution logs
└── temp_outputs/                  # Temporary files (auto-cleaned)
```

## 🧹 Cleanup Strategies

### Automated Cleanup Script

Use the provided cleanup script to manage old outputs:

```bash
# Clean outputs older than 7 days
python scripts/cleanup_outputs.py --days 7

# Clean specific experiment type
python scripts/cleanup_outputs.py --type hyperparameter --days 3

# Dry run to see what would be deleted
python scripts/cleanup_outputs.py --dry-run
```

### Manual Cleanup Guidelines

1. **Keep Recent Results**: Preserve results from the last 7-14 days
2. **Archive Important Configs**: Move best configurations to `configs/archived/`
3. **Clean Ray Results**: Ray Tune creates large directories - clean regularly
4. **Backup Best Models**: Save important model checkpoints to `models/production/`

## 📊 Output File Types

### Hyperparameter Optimization
- `best_*_config_*.json` - Best hyperparameter configurations
- `*_hparam_results_*.json` - Complete optimization results
- `*_trials_*.csv` - Trial-by-trial data for analysis
- `hparam_opt_*/` - Ray Tune experiment directories

### Model Training
- `*.pth` - PyTorch model state dictionaries  
- `*.pkl` - Preprocessors, scalers, and other objects
- `training_history_*.json` - Loss curves and metrics
- `model_summary_*.txt` - Architecture summaries

### Visualizations
- `*.png`, `*.jpg` - Generated plots and charts
- `training_plots_*/` - Training visualization directories

## 🔒 Git Integration

The following patterns are automatically ignored by Git:

```gitignore
# ML/AI Training Outputs
models/
optimization_results/
ray_results/
*.pth
*.pt
*.pkl

# Training logs and metrics  
logs/
*_trials_*.csv
best_*_config_*.json
training_*.json
```

## 💡 Best Practices

### For Notebook Users
1. **Use Timestamps**: Include timestamps in output filenames
2. **Clear Outputs**: Clear notebook outputs before committing
3. **Document Important Runs**: Add markdown cells describing significant experiments
4. **Export Key Results**: Save important configurations to version control

### For Automated Training
1. **Structured Naming**: Use consistent naming conventions
2. **Metadata Files**: Include experiment metadata in JSON format
3. **Checkpoint Management**: Implement automatic checkpoint rotation
4. **Resource Monitoring**: Log GPU/CPU usage for optimization

## 🛠️ Utility Functions

### Notebook Helper Functions

```python
# Clear all notebook outputs
def clear_notebook_outputs():
    from IPython.core.display import display, Javascript
    display(Javascript('IPython.notebook.clear_all_output();'))

# Save experiment configuration
def save_experiment_config(config, experiment_name):
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_results/{experiment_name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Configuration saved to: {filename}")
```

## 📈 Monitoring and Analysis

### Tracking Experiment Progress
- Use consistent naming for easy filtering
- Include validation metrics in filenames when possible
- Create summary dashboards for experiment comparison

### Storage Management
- Monitor disk usage in output directories
- Set up alerts for large directory sizes
- Implement automatic archiving for old experiments

## 🚨 Troubleshooting

### Common Issues
1. **Disk Space**: Monitor `optimization_results/` and `ray_results/` sizes
2. **Permission Issues**: Ensure write permissions in output directories
3. **Ray Cleanup**: Ray may leave behind large temporary files
4. **Notebook Memory**: Clear variables and outputs between large experiments

### Recovery Procedures
- Incomplete experiments: Check for partial results in Ray directories
- Corrupted files: Look for backup configurations in experiment metadata
- Lost configurations: Parse trial CSV files to reconstruct parameters
