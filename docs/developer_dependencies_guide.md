# Dependency Management Guide

This guide explains how to efficiently manage dependencies for the Trading RL Agent project, keeping your environment clean and only installing what you need.

## 🎯 Overview

The project now uses a **tiered dependency system** that allows you to install only the packages you need:

- **Core** (~50MB): Basic functionality only
- **ML** (~2.1GB): Core + Machine Learning capabilities
- **Full** (~2.6GB): Complete production setup
- **Dev** (~500MB): Development and testing tools
- **Production** (~2.6GB): Optimized for deployment

## 📦 Dependency Profiles

### Core Profile (`requirements-core.txt`)

**Size**: ~50MB
**Use case**: Basic trading functionality, data manipulation, CLI tools

```bash
pip install -r requirements-core.txt
```

**Includes**:

- `numpy`, `pandas` - Data manipulation
- `pyyaml`, `python-dotenv` - Configuration
- `requests` - HTTP requests
- `typer`, `rich` - CLI interface
- `structlog` - Logging
- `tqdm` - Progress bars

### ML Profile (`requirements-ml.txt`)

**Size**: ~2.1GB
**Use case**: Machine learning, neural networks, reinforcement learning

```bash
pip install -r requirements-ml.txt
```

**Includes**: Core +

- `torch` - Deep learning framework
- `scikit-learn` - Machine learning
- `scipy` - Scientific computing
- `ta`, `pandas-ta` - Technical analysis
- `statsmodels`, `empyrical` - Statistical analysis
- `optuna` - Hyperparameter optimization
- `matplotlib`, `seaborn` - Visualization
- `gymnasium`, `stable-baselines3` - Reinforcement learning

### Full Profile (`requirements-full.txt`)

**Size**: ~2.6GB
**Use case**: Complete production system with all features

```bash
pip install -r requirements-full.txt
```

**Includes**: ML +

- `yfinance`, `alpha-vantage` - Data sources
- `ray[rllib,tune]` - Distributed RL
- `vaderSentiment` - Sentiment analysis
- `arch`, `quantstats` - Advanced statistics
- `mlflow`, `tensorboard`, `wandb` - Experiment tracking
- `fastapi`, `uvicorn`, `gunicorn` - Web framework
- `pydantic` - Data validation

### Dev Profile (`requirements-dev.txt`)

**Size**: ~500MB
**Use case**: Development, testing, code quality

```bash
pip install -r requirements-dev.txt
```

**Includes**: Core +

- `pytest*` - Testing framework
- `black`, `isort`, `ruff` - Code formatting
- `mypy` - Type checking
- `pre-commit` - Git hooks
- `jupyter`, `notebook` - Development tools

### Production Profile (`requirements-production.txt`)

**Size**: ~2.6GB
**Use case**: Production deployment with pinned versions

```bash
pip install -r requirements-production.txt
```

**Includes**: Full dependencies with specific version pins for stability

## 🚀 Quick Start

### Option 1: Interactive Installer

```bash
python install-deps.py
```

### Option 2: Shell Script

```bash
./setup-env.sh [core|ml|full|dev|production]
```

### Option 3: Direct pip install

```bash
# Core only
pip install -r requirements-core.txt

# ML capabilities
pip install -r requirements-ml.txt

# Full production
pip install -r requirements-full.txt

# Development tools
pip install -r requirements-dev.txt
```

### Option 4: Using pip install with extras

```bash
# Install package with specific extras
pip install -e .[ml]        # ML dependencies
pip install -e .[dev]       # Development tools
pip install -e .[production] # Production tools
pip install -e .[full]      # Everything
```

## 🔍 Dependency Analysis

Analyze dependencies and their sizes:

```bash
# Show all profiles
python analyze-deps.py

# Analyze specific profile
python analyze-deps.py ml
```

This will show:

- Package counts and estimated sizes
- Installation commands
- Current installation status

## 💡 Best Practices

### 1. Start Small

Begin with the **core** profile and add more as needed:

```bash
pip install -r requirements-core.txt
# Test basic functionality
python minimal_test.py
```

### 2. Use Virtual Environments

Always use virtual environments to avoid conflicts:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Pin Versions for Production

Use the production profile for deployment:

```bash
pip install -r requirements-production.txt
```

### 4. Clean Up Unused Dependencies

Regularly audit your environment:

```bash
pip list  # See what's installed
pip uninstall package_name  # Remove unused packages
```

## 🔧 Advanced Usage

### Custom Requirements

Create your own requirements file:

```bash
# Start with core
cp requirements-core.txt my-requirements.txt

# Add specific packages
echo "my-special-package>=1.0.0" >> my-requirements.txt
pip install -r my-requirements.txt
```

### Minimal Docker Images

For Docker deployments, use only core dependencies:

```dockerfile
COPY requirements-core.txt .
RUN pip install -r requirements-core.txt
```

### Development Workflow

1. Install core for basic development
2. Add dev tools when needed
3. Add ML capabilities for model development
4. Use full profile for testing complete system

## 📊 Size Comparison

| Profile    | Size   | Packages | Use Case            |
| ---------- | ------ | -------- | ------------------- |
| Core       | ~50MB  | 9        | Basic functionality |
| Dev        | ~500MB | 15       | Development tools   |
| ML         | ~2.1GB | 20       | Machine learning    |
| Full       | ~2.6GB | 30       | Complete system     |
| Production | ~2.6GB | 30       | Deployed system     |

## 🛠️ Troubleshooting

### Common Issues

**1. Version Conflicts**

```bash
# Use specific versions
pip install package==1.2.3

# Or upgrade all
pip install --upgrade -r requirements-core.txt
```

**2. Large Downloads**

```bash
# Use mirrors or local cache
pip install -r requirements-ml.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**3. Memory Issues**

```bash
# Install packages one by one
pip install numpy pandas
pip install torch  # Install heavy packages separately
```

### Getting Help

1. Check the analysis tool: `python analyze-deps.py`
2. Test your installation: `python minimal_test.py`
3. Review the requirements files for specific versions
4. Use `pip check` to verify dependencies

## 📝 Migration Guide

### From Old Requirements

If you were using the old `requirements.txt`:

1. **Backup your environment**:

   ```bash
   pip freeze > old-requirements.txt
   ```

2. **Choose your new profile**:

   ```bash
   python install-deps.py
   ```

3. **Verify installation**:
   ```bash
   python minimal_test.py
   ```

### Clean Installation

For a completely fresh start:

```bash
# Remove old environment
rm -rf venv/

# Create new environment
python -m venv venv
source venv/bin/activate

# Install minimal dependencies
pip install -r requirements-core.txt
```

## 🎯 Recommendations by Use Case

| Use Case              | Recommended Profile | Alternative |
| --------------------- | ------------------- | ----------- |
| Learning/Exploration  | Core                | ML          |
| Development           | Core + Dev          | Full        |
| Model Training        | ML                  | Full        |
| Production Deployment | Production          | Full        |
| Testing/CI            | Core + Dev          | ML          |
| Research              | ML                  | Full        |

This tiered approach ensures you only install what you need, keeping your environment clean and efficient! 🚀
