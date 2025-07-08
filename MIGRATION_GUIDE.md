# Migration Guide - Trading RL Agent Architecture Restructure

This guide helps you migrate from the old structure to the new production-grade architecture.

## 📋 **Migration Overview**

The restructuring maintains **backward compatibility** while providing a clear **upgrade path** to the new modular architecture. You can migrate incrementally without breaking existing functionality.

---

## 🔄 **Phase 1: Immediate Changes (Already Completed)**

### **✅ New Directory Structure**

```bash
# OLD STRUCTURE (moved to new locations)
src/agents/          → src/trading_rl_agent/agents/
src/core/            → src/trading_rl_agent/core/
src/data/            → src/trading_rl_agent/data/
src/envs/            → src/trading_rl_agent/envs/
src/models/          → src/trading_rl_agent/models/
src/utils/           → src/trading_rl_agent/utils/

# NEW MODULES (added)
src/trading_rl_agent/features/      # Feature engineering
src/trading_rl_agent/portfolio/     # Portfolio management
src/trading_rl_agent/risk/          # Risk management
src/trading_rl_agent/execution/     # Order execution
src/trading_rl_agent/monitoring/    # Performance monitoring
```

### **✅ Updated Import Structure**

```python
# OLD IMPORTS (still work via compatibility layer)
from src.agents import SACAgent
from src.envs import TradingEnv

# NEW IMPORTS (recommended)
from trading_rl_agent.agents import SACAgent
from trading_rl_agent.envs import TradingEnv
from trading_rl_agent.portfolio import PortfolioManager  # NEW
from trading_rl_agent.risk import RiskManager            # NEW
```

---

## 🔄 **Phase 2: Code Migration (Recommended)**

### **Step 1: Update Import Statements**

**Old Code:**

```python
from src.agents.sac_agent import SACAgent
from src.envs.trading_env import TradingEnv
from src.models.cnn_lstm import CNNLSTMModel
```

**New Code:**

```python
from trading_rl_agent.agents import SACAgent
from trading_rl_agent.envs import TradingEnv
from trading_rl_agent.models import CNNLSTMModel
```

### **Step 2: Adopt New Configuration System**

**Old Configuration:**

```python
# Manual configuration
config = {
    'initial_balance': 100000,
    'window_size': 50,
    'batch_size': 32
}
```

**New Configuration:**

```python
from trading_rl_agent import ConfigManager

# Hierarchical configuration with validation
config = ConfigManager("configs/production.yaml")
```

### **Step 3: Integrate Risk Management**

**Old Code (No Risk Management):**

```python
# Basic trading without risk controls
agent = SACAgent(state_dim, action_dim)
action = agent.select_action(state)
```

**New Code (With Risk Management):**

```python
from trading_rl_agent.risk import RiskManager

# Risk-adjusted trading
risk_manager = RiskManager()
agent = SACAgent(state_dim, action_dim)
raw_action = agent.select_action(state)

# Apply risk controls
portfolio_weights = get_current_weights()
risk_metrics = risk_manager.check_limits(portfolio_weights)
if not risk_metrics.violations:
    action = raw_action
else:
    action = risk_manager.adjust_action(raw_action, risk_metrics)
```

### **Step 4: Add Portfolio Management**

**Old Code (Single Asset):**

```python
# Single asset trading
env = TradingEnv(symbol="AAPL")
```

**New Code (Multi-Asset Portfolio):**

```python
from trading_rl_agent.portfolio import PortfolioManager
from trading_rl_agent.envs import PortfolioEnv

# Multi-asset portfolio optimization
symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
portfolio = PortfolioManager(initial_capital=100000)
env = PortfolioEnv(symbols=symbols, portfolio_manager=portfolio)
```

---

## 🔄 **Phase 3: Advanced Features (Optional)**

### **Feature Engineering Pipeline**

```python
from trading_rl_agent.features import FeaturePipeline

# Advanced feature engineering
feature_pipeline = FeaturePipeline([
    'technical_indicators',
    'market_microstructure',
    'cross_asset_correlations',
    'alternative_data'
])

enhanced_data = feature_pipeline.transform(raw_data)
```

### **Real-Time Execution**

```python
from trading_rl_agent.execution import ExecutionEngine

# Production execution engine
execution_engine = ExecutionEngine(
    broker="alpaca",
    paper_trading=True,  # Start with paper trading
    max_slippage=0.001
)

# Execute trades with smart routing
execution_engine.execute_order(
    symbol="AAPL",
    quantity=100,
    order_type="market"
)
```

### **Performance Monitoring**

```python
from trading_rl_agent.monitoring import MetricsCollector

# Real-time performance monitoring
metrics = MetricsCollector()
metrics.start_monitoring(portfolio, frequency=60)  # Every minute

# Access real-time metrics
current_metrics = metrics.get_current_metrics()
print(f"Sharpe Ratio: {current_metrics['sharpe_ratio']}")
print(f"Max Drawdown: {current_metrics['max_drawdown']}")
```

---

## 🧪 **Testing Migration**

### **Update Test Imports**

**Old Test Code:**

```python
from src.agents.sac_agent import SACAgent
from src.envs.trading_env import TradingEnv

def test_agent_training():
    agent = SACAgent(state_dim=10, action_dim=3)
    # ... test code
```

**New Test Code:**

```python
from trading_rl_agent.agents import SACAgent
from trading_rl_agent.envs import TradingEnv

def test_agent_training():
    agent = SACAgent(state_dim=10, action_dim=3)
    # ... test code (same functionality)
```

### **Run Migration Tests**

```bash
# Test backward compatibility
pytest tests/migration/ -v

# Test new features
pytest tests/unit/test_portfolio.py -v
pytest tests/unit/test_risk.py -v
pytest tests/integration/test_new_architecture.py -v
```

---

## 📊 **Configuration Migration**

### **Convert Old Config Files**

**Old Configuration (config.json):**

```json
{
  "initial_balance": 100000,
  "window_size": 50,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

**New Configuration (configs/config.yaml):**

```yaml
environment: development
debug: false

data:
  feature_window: 50
  cache_enabled: true

model:
  batch_size: 32
  learning_rate: 0.001

agent:
  agent_type: sac
  total_timesteps: 1000000

risk:
  max_position_size: 0.1
  max_leverage: 1.0

execution:
  broker: alpaca
  paper_trading: true
```

### **Migration Script**

```python
# scripts/migrate_config.py
import json
import yaml
from pathlib import Path

def migrate_config(old_config_path, new_config_path):
    """Migrate old JSON config to new YAML structure."""

    # Load old config
    with open(old_config_path) as f:
        old_config = json.load(f)

    # Convert to new structure
    new_config = {
        'environment': 'development',
        'debug': False,
        'data': {
            'feature_window': old_config.get('window_size', 50),
            'cache_enabled': True
        },
        'model': {
            'batch_size': old_config.get('batch_size', 32),
            'learning_rate': old_config.get('learning_rate', 0.001)
        },
        'agent': {
            'agent_type': 'sac',
            'total_timesteps': 1000000
        },
        'risk': {
            'max_position_size': 0.1,
            'max_leverage': 1.0
        },
        'execution': {
            'broker': 'alpaca',
            'paper_trading': True
        }
    }

    # Save new config
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, indent=2)

    print(f"✅ Migrated {old_config_path} → {new_config_path}")

# Usage
migrate_config("config.json", "configs/config.yaml")
```

---

## 🔧 **Common Migration Issues & Solutions**

### **Issue 1: Import Errors**

**Problem:**

```python
ImportError: No module named 'src.agents'
```

**Solution:**

```python
# Add compatibility import at the top of your files
try:
    from trading_rl_agent.agents import SACAgent
except ImportError:
    from src.agents.sac_agent import SACAgent  # Fallback
```

### **Issue 2: Configuration Loading**

**Problem:**

```python
# Old code doesn't work with new config system
config = load_json_config("config.json")
```

**Solution:**

```python
from trading_rl_agent import ConfigManager

# Use new config manager with fallback
try:
    config = ConfigManager("configs/config.yaml")
except FileNotFoundError:
    # Fallback to old system during migration
    config = load_json_config("config.json")
```

### **Issue 3: Test Failures**

**Problem:**

```bash
ModuleNotFoundError: No module named 'src'
```

**Solution:**

```bash
# Update PYTHONPATH temporarily
export PYTHONPATH="/workspaces/trading-rl-agent/src:$PYTHONPATH"
pytest tests/

# Or update test imports
sed -i 's/from src\./from trading_rl_agent\./g' tests/**/*.py
```

---

## ✅ **Migration Checklist**

### **Phase 1: Basic Migration**

- [ ] Update import statements
- [ ] Test existing functionality
- [ ] Verify all tests pass
- [ ] Update configuration files

### **Phase 2: Enhanced Features**

- [ ] Integrate risk management
- [ ] Add portfolio management
- [ ] Implement feature engineering
- [ ] Setup monitoring

### **Phase 3: Production Readiness**

- [ ] Configure real-time data feeds
- [ ] Setup execution engine
- [ ] Deploy monitoring dashboard
- [ ] Test with paper trading

### **Phase 4: Advanced Features**

- [ ] Multi-asset portfolio optimization
- [ ] Alternative data integration
- [ ] Advanced risk models
- [ ] Production deployment

---

## 📚 **Resources**

### **Documentation**

- [Architecture Overview](ARCHITECTURE_RESTRUCTURE.md) - Complete system design
- [API Reference](docs/api_reference.md) - Detailed API documentation
- [Configuration Guide](docs/configuration.md) - Configuration management

### **Examples**

- [Basic Migration](examples/migration_basic.py) - Simple migration example
- [Advanced Migration](examples/migration_advanced.py) - Full feature migration
- [Testing Migration](examples/migration_testing.py) - Test update examples

### **Support**

- **GitHub Issues**: Report migration problems
- **Documentation**: Comprehensive migration guides
- **Examples**: Working migration examples

---

## 🎯 **Migration Timeline**

### **Week 1: Planning & Assessment**

- Review current codebase
- Identify migration scope
- Plan migration strategy

### **Week 2-3: Core Migration**

- Update imports and structure
- Migrate configuration system
- Update and run tests

### **Week 4-5: Feature Integration**

- Add risk management
- Integrate portfolio management
- Setup monitoring

### **Week 6: Production Preparation**

- Test with paper trading
- Performance optimization
- Documentation updates

---

**💡 Pro Tip**: Start with paper trading when testing the new features. The production-grade architecture includes comprehensive safety mechanisms, but it's always best to validate thoroughly before using real capital.

**🚀 Ready to migrate to production-grade trading systems!**
