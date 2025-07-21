# Development Guide

This guide outlines best practices for ML workflows, output management, and development tools to ensure consistency and efficiency.

## 📚 ML Workflow Best Practices

### Notebook Organization

A clear and organized notebook is crucial for reproducibility. Structure your notebooks as follows:

- **Environment Setup**: Import libraries, set up logging, and define constants.
- **Data Loading**: Load and preprocess data, including any feature engineering.
- **Model Training**: Define and train your models, and log experiments.
- **Evaluation & Results**: Visualize and analyze results, and save key metrics.
- **Cleanup & Export**: Clear memory and export artifacts.

### Memory Management

Efficient memory usage is critical, especially when working with large datasets and models.

- **Clear GPU Memory**: Use `torch.cuda.empty_cache()` between experiments to free up unused GPU memory.
- **Monitor Resources**: Keep an eye on GPU and CPU usage to prevent bottlenecks.

### Experiment Tracking

Track all experiments to ensure you can reproduce and compare results.

- **Save Configurations**: Store the exact configuration used for each experiment.
- **Timestamp Results**: Log results with timestamps for easy tracking.

## 🧹 Output Management

### Directory Structure

Maintain a clean directory structure to avoid clutter and confusion.

- `optimization_results/`: Store only the best-performing configurations.
- `ray_results/`: Regularly clean up old and failed trials.
- `data/`: Archive outdated datasets to keep the working directory clean.
- `*.ipynb`: Always clear outputs from notebooks before committing to Git.

### Cleanup Commands

Use these commands to keep your workspace tidy:

- **Clear Notebook Outputs**: `find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} +`
- **Clean Experiment Outputs**: `find ray_results/ -type d -mtime +7 -exec rm -rf {} +`

### Git Pre-commit Setup

Automate code quality checks with pre-commit hooks.

- **Install Hooks**: Run `pre-commit install` to set up the hooks.
- **Manual Cleanup**: Before committing, manually clear notebook outputs to avoid storing large files in Git.

## 🔧 Development Tools

### Required Extensions (VS Code)

- **Python**: Core Python support.
- **Pylance**: Advanced language server for Python.
- **Ruff**: Code formatting and linting.
- **GitLens**: Enhanced Git capabilities.

### VS Code Configuration

```json
{
  "python.formatting.provider": "charliermarsh.ruff",
  "editor.formatOnSave": true,
  "python.linting.enabled": true
}
```

---

For legal and safety notes see the [project disclaimer](disclaimer.md).
