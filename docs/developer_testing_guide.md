# Testing Guide

This document provides comprehensive information about the testing strategy, current status, and guidelines for improving test coverage in the Trading RL Agent project.

## 📊 **Current Testing Status**

### **Overall Coverage: 3.91%**

**Target Coverage: 90%**

### **Coverage Breakdown by Module**

| Module               | Coverage | Status       | Priority |
| -------------------- | -------- | ------------ | -------- |
| Core Configuration   | 82.32%   | ✅ Good      | Low      |
| Agent Configurations | 88.06%   | ✅ Good      | Low      |
| Exception Handling   | 100%     | ✅ Excellent | Low      |
| Risk Management      | 13.14%   | 🔄 Critical  | **High** |
| CLI Interface        | 0%       | 🔄 Critical  | **High** |
| Data Pipeline        | 0%       | 🔄 Critical  | **High** |
| Model Training       | 0%       | 🔄 Critical  | **High** |
| Portfolio Management | 0%       | 🔄 Important | Medium   |
| Feature Engineering  | 0%       | 🔄 Important | Medium   |
| Live Trading         | 0%       | 🔄 Important | Medium   |

## 🧪 **Test Structure**

### **Test Organization**

```
tests/
├── unit/                    # Unit tests for individual components
│   ├── test_core_config.py  # Configuration system tests
│   ├── test_features.py     # Feature engineering tests
│   ├── test_data_utils.py   # Data utility tests
│   └── ...                  # Other unit tests
├── integration/             # Integration tests for workflows
├── smoke/                   # Smoke tests for basic functionality
├── performance/             # Performance and benchmark tests
└── conftest.py             # Shared test fixtures
```

### **Test Categories**

1. **Unit Tests**: Test individual functions and classes in isolation
2. **Integration Tests**: Test interactions between multiple components
3. **Smoke Tests**: Basic functionality tests for quick validation
4. **Performance Tests**: Benchmark and performance regression tests

## 🎯 **Testing Priorities**

### **Phase 1: Critical Components (Target: 50% coverage)**

#### **1. CLI Interface Testing (Priority 1)**

- **Current Coverage**: 0%
- **Target Coverage**: 80%
- **Files to Test**:
  - `src/trading_rl_agent/cli.py`
  - `src/trading_rl_agent/cli_backtest.py`
  - `src/trading_rl_agent/cli_trade.py`
  - `src/trading_rl_agent/cli_train.py`

**Testing Strategy**:

```python
# Example CLI test structure
def test_cli_version():
    """Test CLI version command"""
    result = runner.invoke(cli, ["version"])
    assert result.exit_code == 0
    assert "Trading RL Agent" in result.output

def test_cli_data_download():
    """Test data download command"""
    result = runner.invoke(cli, ["data", "download", "--symbols", "AAPL"])
    assert result.exit_code == 0
```

#### **2. Data Pipeline Testing (Priority 2)**

- **Current Coverage**: 0%
- **Target Coverage**: 70%
- **Files to Test**:
  - `src/trading_rl_agent/data/pipeline.py`
  - `src/trading_rl_agent/data/loaders/`
  - `src/trading_rl_agent/data/preprocessing.py`
  - `src/trading_rl_agent/data/optimized_dataset_builder.py`

**Testing Strategy**:

```python
# Example data pipeline test
def test_data_pipeline_end_to_end():
    """Test complete data pipeline workflow"""
    # Test data loading
    # Test preprocessing
    # Test feature engineering
    # Test dataset creation
    pass
```

#### **3. Model Training Testing (Priority 3)**

- **Current Coverage**: 0%
- **Target Coverage**: 60%
- **Files to Test**:
  - `src/trading_rl_agent/training/`
  - `src/trading_rl_agent/models/`
  - `src/trading_rl_agent/agents/trainer.py`

### **Phase 2: Important Components (Target: 75% coverage)**

#### **4. Risk Management Testing**

- **Current Coverage**: 13.14%
- **Target Coverage**: 80%
- **Files to Test**:
  - `src/trading_rl_agent/risk/manager.py`
  - `src/trading_rl_agent/risk/position_sizer.py`

#### **5. Portfolio Management Testing**

- **Current Coverage**: 0%
- **Target Coverage**: 70%
- **Files to Test**:
  - `src/trading_rl_agent/portfolio/manager.py`

### **Phase 3: Advanced Components (Target: 90% coverage)**

#### **6. Live Trading Testing**

- **Current Coverage**: 0%
- **Target Coverage**: 60%
- **Files to Test**:
  - `src/trading_rl_agent/core/live_trading.py`

#### **7. Monitoring and Alerting Testing**

- **Current Coverage**: 0%
- **Target Coverage**: 70%
- **Files to Test**:
  - `src/trading_rl_agent/monitoring/`

## 🛠️ **Testing Tools and Setup**

### **Required Dependencies**

```bash
# Install testing dependencies
pip install pytest pytest-cov pytest-mock pytest-asyncio
pip install pytest-benchmark pytest-xdist
```

### **Running Tests**

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=trading_rl_agent --cov-report=html

# Run specific test categories
python -m pytest tests/unit/
python -m pytest tests/integration/

# Run tests in parallel
python -m pytest -n auto

# Run performance tests
python -m pytest tests/performance/
```

### **Coverage Reporting**

```bash
# Generate coverage report
python -m pytest --cov=trading_rl_agent --cov-report=term-missing

# Generate HTML coverage report
python -m pytest --cov=trading_rl_agent --cov-report=html

# Generate XML coverage report (for CI/CD)
python -m pytest --cov=trading_rl_agent --cov-report=xml
```

## 📝 **Testing Guidelines**

### **Test Naming Conventions**

```python
# Use descriptive test names
def test_feature_engineering_pipeline_handles_missing_data():
    """Test that feature engineering handles missing data gracefully"""
    pass

def test_risk_manager_calculates_var_correctly():
    """Test VaR calculation accuracy"""
    pass
```

### **Test Structure**

```python
def test_component_functionality():
    """Test description"""
    # Arrange - Set up test data and conditions
    test_data = create_test_data()

    # Act - Execute the function being tested
    result = component_function(test_data)

    # Assert - Verify the results
    assert result is not None
    assert len(result) > 0
```

### **Mocking Guidelines**

```python
# Use mocks for external dependencies
@patch('trading_rl_agent.data.loaders.yfinance_loader.yf.download')
def test_data_loader_handles_api_errors(mock_download):
    """Test data loader handles API errors gracefully"""
    mock_download.side_effect = Exception("API Error")

    with pytest.raises(DataLoadError):
        load_market_data("AAPL")
```

### **Fixtures**

```python
# Create reusable test fixtures
@pytest.fixture
def sample_market_data():
    """Provide sample market data for tests"""
    return pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [95, 96, 97],
        'close': [103, 104, 105],
        'volume': [1000, 1100, 1200]
    })

@pytest.fixture
def mock_config():
    """Provide mock configuration for tests"""
    return {
        'data': {'symbols': ['AAPL'], 'start_date': '2023-01-01'},
        'features': {'technical_indicators': True},
        'model': {'type': 'cnn_lstm'}
    }
```

## 🚀 **Implementation Plan**

### **Week 1-2: CLI Testing**

- [ ] Create CLI test framework
- [ ] Test version command
- [ ] Test data download command
- [ ] Test data process command
- [ ] Test training commands
- [ ] Test evaluation commands

### **Week 3-4: Data Pipeline Testing**

- [ ] Test data loaders (yfinance, Alpha Vantage)
- [ ] Test data preprocessing
- [ ] Test feature engineering pipeline
- [ ] Test dataset builder
- [ ] Test parallel data fetching

### **Week 5-6: Model Training Testing**

- [ ] Test CNN+LSTM model architecture
- [ ] Test training pipeline
- [ ] Test model evaluation
- [ ] Test hyperparameter optimization

### **Week 7-8: Integration Testing**

- [ ] Test end-to-end workflows
- [ ] Test cross-module interactions
- [ ] Test error handling and recovery
- [ ] Test performance under load

## 📊 **Success Metrics**

### **Coverage Targets**

| Phase   | Target Coverage | Timeline |
| ------- | --------------- | -------- |
| Phase 1 | 50%             | 4 weeks  |
| Phase 2 | 75%             | 8 weeks  |
| Phase 3 | 90%             | 12 weeks |

### **Quality Metrics**

- **Test Execution Time**: < 5 minutes for full suite
- **Test Reliability**: > 99% pass rate
- **Code Quality**: Maintain A+ rating with ruff/mypy
- **Documentation**: 100% of public APIs documented

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Issues**: Check mock setup and teardown
3. **Data Issues**: Use fixtures for consistent test data
4. **Performance Issues**: Use pytest-benchmark for performance tests

### **Getting Help**

- Check existing test examples in `tests/` directory
- Review pytest documentation
- Ask questions in GitHub issues
- Review test coverage reports

---

**Remember**: Good tests are an investment in code quality and maintainability. Focus on testing the most critical paths and error conditions first.
