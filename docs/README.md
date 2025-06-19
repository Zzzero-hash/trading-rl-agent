# Documentation Index

This directory contains comprehensive documentation for the Trading RL Agent project.

## 📚 Available Documentation

### Core Development Guides

#### [`NOTEBOOK_BEST_PRACTICES.md`](NOTEBOOK_BEST_PRACTICES.md)

**Complete guide for Jupyter notebook development in ML workflows**

- Development workflow and code organization
- Technical best practices (memory management, progress tracking)
- Hyperparameter optimization with Ray Tune
- Visualization and results comparison
- Automated cleanup integration
- Reproducibility and configuration versioning
- Common pitfalls and advanced techniques

#### [`EXPERIMENT_OUTPUTS_MANAGEMENT.md`](EXPERIMENT_OUTPUTS_MANAGEMENT.md)

**Comprehensive guide for managing ML experiment outputs and storage**

- Directory structure overview and output types
- Cleanup recommendations and automation
- Storage management and monitoring
- VS Code integration and settings
- Emergency cleanup procedures

- Git integration and pre-commit hooks

#### [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md)

**High level architecture and RL agent interactions**

- Data pipeline, `TradingEnv`, SAC/TD3, and ensemble overview

#### [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md)

**How to evaluate trained agents and interpret metrics**

## 🎯 Quick Navigation

### For New Developers

1. Start with [`NOTEBOOK_BEST_PRACTICES.md`](NOTEBOOK_BEST_PRACTICES.md) - Learn the development workflow
2. Review [`EXPERIMENT_OUTPUTS_MANAGEMENT.md`](EXPERIMENT_OUTPUTS_MANAGEMENT.md) - Understand output management

### For Experiment Management

- **Daily**: Use `python scripts/cleanup_experiments.py --status-only` to monitor storage
- **Weekly**: Run `python scripts/cleanup_experiments.py --archive --all` for cleanup
- **Pre-commit**: Outputs are automatically cleared by git hooks

### For Production Deployment

- All documentation includes production-ready practices
- Automated tooling is included for experiment lifecycle management
- Storage limits and monitoring guidelines are provided

## 🔧 Related Tools

All documentation references the automated tools in [`scripts/`](../scripts/):

- `cleanup_experiments.py` - Main experiment management tool
- `pre-commit-hook.sh` - Automatic cleanup before git commits
- See [`scripts/README.md`](../scripts/README.md) for detailed tool documentation

## 📈 Documentation Philosophy

Our documentation follows these principles:

1. **Actionable**: Every guide includes concrete commands and examples
2. **Automated**: Manual processes are complemented by automated tools
3. **Scalable**: Practices work for both individual development and team collaboration
4. **Production-Ready**: All recommendations are suitable for production environments

## 🚀 Getting Started

```bash
# Set up experiment management (one-time setup)
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Daily development workflow
python scripts/cleanup_experiments.py --status-only

# Weekly maintenance
python scripts/cleanup_experiments.py --archive --all
```

---

**📋 Note**: This documentation is actively maintained and reflects the current state of the project. All tools and practices have been tested in the production environment.
