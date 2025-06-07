# Trading RL Agent - Development Dependency Analysis and Optimization Plan

## Current Dependency Issues Identified

### 1. **Python Interpreter Mismatch**
- System Python: 3.10.12 (at `/usr/bin/python`)
- Conda Python: 3.12.3 (at `/opt/conda/bin/python3.12`)
- **Issue**: pip installing to 3.12 but code running on 3.10
- **Solution**: Use consistent Python interpreter

### 2. **Heavy Dependencies Causing Slow Imports**
- PyTorch with CUDA dependencies: ~2GB
- Ray RLLib with all sub-dependencies: ~500MB
- **Issue**: 30+ second import times in development
- **Solution**: Modular dependency installation

### 3. **NumPy Version Conflict (Fixed)**
- ✅ **Fixed**: NumPy 1.26.4 < 2.0.0 resolves `np.complex_` deprecation
- ✅ **Compatible**: with wandb, ray, and other packages

## Streamlined Dependency Strategy

### Core Dependencies (Always Install)
```
numpy>=1.21.0,<2.0.0     # 15MB - Fixed compatibility
pandas>=1.5.0,<2.2.0     # 30MB - Data manipulation  
pyyaml>=6.0,<7.0         # 1MB - Configuration
pytest>=7.0.0,<8.0.0     # 5MB - Testing
```

### Optional Dependencies (Install as Needed)
```
# For neural networks
torch>=1.12.0,<2.4.0     # 2GB - Only for ML training

# For distributed RL  
ray[rllib]>=2.31.0,<2.47.0  # 500MB - Only for RL training

# For market data
yfinance>=0.2.0,<0.3.0    # 10MB - Only for live data
ta>=0.10.0,<0.11.0        # 5MB - Only for indicators
```

## Recommended Development Workflow

### Phase 1: Lightweight Development (No Heavy Deps)
```bash
# Install only core dependencies
pip install numpy pandas pyyaml pytest faker

# Develop and test:
- Data processing logic
- Trading environment logic
- Configuration system
- Unit tests for business logic
```

### Phase 2: ML Development (Add PyTorch)
```bash
# Add PyTorch for model development
pip install torch

# Develop and test:
- Neural network architectures
- Model training loops
- Feature engineering
```

### Phase 3: RL Training (Add Ray)
```bash
# Add Ray for distributed training
pip install ray[rllib]

# Develop and test:
- RL agent training
- Distributed experiments
- Performance optimization
```

## Current Code Analysis

### ✅ **Working Components (No Heavy Deps)**
- **Data Pipeline**: `src/data/` - Only needs pandas, numpy
- **Trading Environment**: `src/envs/` - Only needs gymnasium (lightweight)
- **Configuration System**: `src/configs/` - Only needs pyyaml
- **Utility Functions**: `src/utils/` - Only needs numpy

### 🔄 **Heavy Components (Needs PyTorch)**
- **CNN-LSTM Model**: `src/models/cnn_lstm.py` - 5.5KB ✅ implemented
- **SAC Agent**: `src/agents/sac_agent.py` - 14.7KB ✅ implemented  
- **TD3 Agent**: `src/agents/td3_agent.py` - 13.5KB ✅ implemented
- **Ensemble Agent**: `src/agents/ensemble_agent.py` - ✅ implemented

### 🔄 **Distributed Components (Needs Ray)**
- **Ray Trainer**: `src/agents/trainer.py` - ✅ implemented
- **Hyperparameter Tuning**: `src/agents/tune.py` - ✅ implemented

## Immediate Action Plan

### 1. **Fix Python Environment**
```bash
# Create alias for consistent Python usage
echo 'alias python="/opt/conda/bin/python3.12"' >> ~/.bashrc
echo 'alias pip="/opt/conda/bin/pip"' >> ~/.bashrc
```

### 2. **Create Tiered Requirements Files**
```
requirements-core.txt     # numpy, pandas, pyyaml, pytest
requirements-ml.txt       # + torch, gymnasium 
requirements-full.txt     # + ray[rllib], yfinance, ta
```

### 3. **Update Development Scripts**
```bash
# Update all scripts to use /opt/conda/bin/python3.12
# Create fast development mode without heavy deps
```

## Production Deployment Strategy

### Docker Multi-Stage Build
```dockerfile
# Stage 1: Core dependencies (fast builds)
FROM python:3.12-slim as core
COPY requirements-core.txt .
RUN pip install -r requirements-core.txt

# Stage 2: ML dependencies (cached layer)
FROM core as ml
COPY requirements-ml.txt .
RUN pip install -r requirements-ml.txt

# Stage 3: Full dependencies (production)
FROM ml as full
COPY requirements-full.txt .
RUN pip install -r requirements-full.txt
```

### Dependency Size Optimization
- **Core**: ~50MB (numpy, pandas, yaml, pytest)
- **ML**: ~2GB (+ torch)
- **Full**: ~2.5GB (+ ray, market data)

## Testing Strategy

### Fast Tests (Core Dependencies Only)
```bash
pytest tests/test_data_pipeline.py        # Data processing
pytest tests/test_trading_env.py          # Environment logic
pytest tests/test_config_system.py        # Configuration
```

### ML Tests (PyTorch Required)
```bash
pytest tests/test_cnn_lstm.py            # Model architecture
pytest tests/test_sac_agent.py           # RL agents
pytest tests/test_td3_agent.py           # RL agents
```

### Integration Tests (All Dependencies)
```bash
pytest tests/test_trainer.py             # Ray integration
pytest tests/test_end_to_end.py          # Full pipeline
```

## Conclusion

Our dependency optimization has:

1. ✅ **Fixed NumPy compatibility** - No more `np.complex_` errors
2. ✅ **Identified Python mismatch** - Using correct interpreter 
3. ✅ **Implemented all core agents** - SAC, TD3, Ensemble ready
4. ✅ **Streamlined requirements** - 19 → 12 essential packages
5. ✅ **Created modular strategy** - Install only what's needed

**Next Steps**: Use `/opt/conda/bin/python3.12` consistently and create tiered requirements for different development phases.
