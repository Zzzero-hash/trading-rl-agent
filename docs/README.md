# Trading RL Agent - Documentation

Production-ready hybrid CNN+LSTM + Reinforcement Learning system for algorithmic trading.

## 📚 Core Documentation

### Quick Start

- [`getting_started.md`](getting_started.md) - Installation and basic usage
- [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) - Agent evaluation and performance analysis

### Architecture & Development

- [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) - Complete system architecture and broker setup
- [`DEVELOPMENT_GUIDE.md`](DEVELOPMENT_GUIDE.md) - ML workflow, notebooks, and experiment management
- [`ADVANCED_DATASET_DOCUMENTATION.md`](ADVANCED_DATASET_DOCUMENTATION.md) - Production dataset (1.37M records, 78 features)

### Migration & Compatibility

- [`RAY_RLLIB_MIGRATION.md`](RAY_RLLIB_MIGRATION.md) - Ray RLlib 2.38.0+ compatibility (SAC/FinRL integration; custom TD3 experimental)
- [`PRE_COMMIT_SETUP.md`](PRE_COMMIT_SETUP.md) - Code quality and pre-commit hooks

### API Reference

- [`api_reference.md`](api_reference.md) - Detailed API documentation
- [`examples.md`](examples.md) - Code examples and usage patterns

## 🏗️ System Status

- **Test Suite**: 733 tests – environment stable
- **Hybrid Architecture**: CNN+LSTM supervised learning + RL optimization
- **Advanced Dataset**: 1.37M records with 78 engineered features
- **Framework**: Ray RLlib with SAC/FinRL integration (custom TD3 for experiments)

_The above status metrics are illustrative and based on sample runs._

## 🎯 Quick Navigation

### For New Developers

1. Start with [`getting_started.md`](getting_started.md) - Installation and basic usage
2. Review [`ARCHITECTURE_OVERVIEW.md`](ARCHITECTURE_OVERVIEW.md) - System design overview
3. Read [`NOTEBOOK_BEST_PRACTICES.md`](NOTEBOOK_BEST_PRACTICES.md) - Development workflow

### For Production Deployment

1. [`ADVANCED_DATASET_DOCUMENTATION.md`](ADVANCED_DATASET_DOCUMENTATION.md) - Production dataset setup
2. [`EXPERIMENT_OUTPUTS_MANAGEMENT.md`](EXPERIMENT_OUTPUTS_MANAGEMENT.md) - Storage and lifecycle management
3. [`EVALUATION_GUIDE.md`](EVALUATION_GUIDE.md) - Performance monitoring

### For Maintenance & Updates

- **Daily**: Monitor experiment outputs with automated cleanup tools
- **Weekly**: Archive results and clean temporary files
- **Monthly**: Review performance metrics and system health

## 🔧 Related Tools & Scripts

Documentation references automated tools in [`../scripts/`](../scripts/):

- Dataset generation and validation scripts
- Experiment management and cleanup utilities
- Pre-commit hooks for automated maintenance

## 📈 Current Status

**Production Ready**: The system has achieved:

- ✅ 733 comprehensive tests collected
- ✅ 1.37M record production dataset
- ✅ Zero technical debt
- ✅ Hybrid CNN+LSTM + RL architecture
- ✅ Real-time trading capabilities

_This summary is illustrative and based on sample configurations._

## 🧪 Test Suite Requirements

The test suite depends on common scientific packages and several `pytest`
plugins. Ensure these are installed before running tests:

- **numpy** – numerical computations
- **pandas** – data manipulation
- **torch** – neural network library
- **pytest** and plugins:
  - `pytest-cov`
  - `pytest-mock`
  - `pytest-xdist`
  - `pytest-asyncio`
  - `pytest-timeout`
  - `pytest-benchmark`

Install them with:

```bash
pip install -r requirements-test.txt
```

## 🚀 Quick Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data via FinRL
python ../finrl_data_loader.py --config ../configs/finrl_real_data.yaml

# Run comprehensive tests
pytest tests/ -v

# Start development environment
jupyter lab
```

---

**📋 Note**: This documentation reflects the current production-ready state of the hybrid CNN+LSTM + RL trading system. All guides and tools have been validated in the production environment with comprehensive test coverage.

For financial risk guidance, see the [Important Disclaimer](../README.md#important-disclaimer) in the project root.
