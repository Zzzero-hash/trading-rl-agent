# Experiment Outputs Management Guide

## 📁 Directory Structure Overview

This project generates several types of outputs during ML training and hyperparameter optimization:

```
trading-rl-agent/
├── optimization_results/     # Hyperparameter optimization results
│   ├── best_*_config_*.json # Best configurations found
│   ├── *_hparam_results_*.json # Full optimization results
│   ├── *_trials_*.csv       # Trial data for analysis
│   └── hparam_opt_*/        # Individual optimization run directories
├── ray_results/             # Ray Tune experiment outputs
│   └── *_hparam_*/          # Ray Tune trial directories
├── data/                    # Training datasets (sample files)
└── *.ipynb                  # Jupyter notebooks with cell outputs
```

## 🧹 Cleanup Recommendations

### Before Each Commit

1. **Clear Notebook Outputs**: Always clear notebook outputs before committing

   ```bash
   # Use VS Code command palette: "Notebook: Clear All Outputs"
   # Or programmatically in notebook:
   from IPython.core.display import display, Javascript
   display(Javascript('IPython.notebook.clear_all_output();'))
   ```

2. **Archive Important Results**: Save significant results before cleanup

   ```bash
   # Create archive directory
   mkdir -p results_archive/$(date +%Y%m%d)

   # Copy best configurations
   cp optimization_results/best_*_config_*.json results_archive/$(date +%Y%m%d)/

   # Copy summary results
   cp optimization_results/*_hparam_results_*.json results_archive/$(date +%Y%m%d)/
   ```

### Weekly Cleanup

1. **Remove Old Experiment Directories**:

   ```bash
   # Remove Ray Tune results older than 7 days
   find ray_results/ -type d -name "*_hparam_*" -mtime +7 -exec rm -rf {} +

   # Remove old optimization run directories
   find optimization_results/ -type d -name "hparam_opt_*" -mtime +7 -exec rm -rf {} +
   ```

2. **Consolidate Results**:
   ```bash
   # Keep only the best results from each day
   python scripts/consolidate_results.py --keep-best-per-day
   ```

## 📋 What to Keep vs. What to Clean

### ✅ Keep (Archive)

- Best configuration files (`best_*_config_*.json`)
- Final experiment summaries (`*_hparam_results_*.json`)
- Key trial data for analysis
- Notebooks with cleared outputs (code only)
- Performance plots and visualizations

### 🗑️ Clean Up Regularly

- Individual Ray Tune trial directories
- Temporary optimization directories (`hparam_opt_*`)
- Large CSV files with all trial data (after extracting insights)
- Notebook cell outputs
- Intermediate model checkpoints
- Debug logs and temporary files

## 🔄 Automated Cleanup Scripts

### 1. Pre-commit Hook Setup

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Clear Jupyter notebook outputs before commit
jupyter nbconvert --clear-output --inplace *.ipynb 2>/dev/null || true

# Check for large files
find . -size +50M -not -path "./.git/*" -not -path "./ray_results/*" -not -path "./optimization_results/*" | head -5
```

### 2. Experiment Cleanup Script

```python
# scripts/cleanup_experiments.py
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_old_experiments(days_to_keep=7):
    """Remove experiment results older than specified days."""
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)

    # Cleanup Ray results
    ray_results = Path("ray_results")
    if ray_results.exists():
        for experiment_dir in ray_results.iterdir():
            if experiment_dir.is_dir():
                dir_time = datetime.fromtimestamp(experiment_dir.stat().st_mtime)
                if dir_time < cutoff_date:
                    print(f"Removing old experiment: {experiment_dir}")
                    shutil.rmtree(experiment_dir)

    # Cleanup optimization results subdirectories
    opt_results = Path("optimization_results")
    if opt_results.exists():
        for item in opt_results.iterdir():
            if item.is_dir() and item.name.startswith("hparam_opt_"):
                dir_time = datetime.fromtimestamp(item.stat().st_mtime)
                if dir_time < cutoff_date:
                    print(f"Removing old optimization run: {item}")
                    shutil.rmtree(item)

if __name__ == "__main__":
    cleanup_old_experiments()
```

## 📊 Storage Management

### Current Storage Usage

Monitor disk usage regularly:

```bash
# Check size of output directories
du -sh optimization_results/ ray_results/ data/

# List largest files
find . -type f -size +10M -not -path "./.git/*" | xargs ls -lh | sort -k5 -h
```

### Recommended Limits

- **optimization_results/**: Keep < 500MB (archive older results)
- **ray_results/**: Keep < 1GB (clean up after each experiment)
- **Notebook outputs**: Always clear before commit
- **data/**: Keep sample files < 50MB (use data versioning for larger datasets)

## 🔧 VS Code Integration

### Settings for Notebook Management

Add to `.vscode/settings.json`:

```json
{
  "notebook.output.textLineLimit": 30,
  "notebook.output.wordWrap": true,
  "notebook.clearOutputBeforeSave": true,
  "files.watcherExclude": {
    "**/ray_results/**": true,
    "**/optimization_results/hparam_opt_*/**": true
  }
}
```

### Recommended Extensions

- **Jupyter**: Core notebook support
- **Python**: Language support
- **GitLens**: Better git integration
- **File Utils**: Bulk file operations

## 📈 Best Practices Summary

1. **Always clear notebook outputs** before committing
2. **Archive important results** regularly
3. **Set up automated cleanup** for old experiments
4. **Monitor disk usage** and set limits
5. **Use meaningful naming conventions** for experiments
6. **Document significant findings** before cleanup
7. **Keep only essential data** in version control

## 🚨 Emergency Cleanup

If disk space is critically low:

```bash
# Emergency cleanup (BE CAREFUL!)
# 1. Clear all notebook outputs
find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

# 2. Remove all Ray results
rm -rf ray_results/*

# 3. Keep only best configs from optimization results
cd optimization_results
mkdir ../temp_best
cp best_*_config_*.json ../temp_best/
rm -rf *
mv ../temp_best/* .
rmdir ../temp_best

# 4. Clear Python cache
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
```

Remember: Always backup important results before aggressive cleanup!
