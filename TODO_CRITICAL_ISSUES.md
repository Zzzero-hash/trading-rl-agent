# 🔧 CRITICAL ISSUES & TODOs
*Generated from Testing Session - June 11, 2025*

## 🚨 **IMMEDIATE PRIORITY - Phase 2 Blockers**

### **Issue #1: Trading Environment Action Space Configuration**
**Status: CRITICAL BLOCKER** ⚠️
**Affects: 5 integration tests**

**Problem**:
```python
# In tests/test_td3_integration.py (multiple tests)
action_dim = trading_env.action_space.shape[0]  # IndexError: tuple index out of range
```

**Root Cause**: 
- `TradingEnv.action_space` likely returns a scalar/Discrete space instead of Box space
- TD3 agent expects continuous action space with shape attribute
- Action space configuration inconsistent between environment definition and agent expectations

**Investigation Required**:
```bash
# Check current action space configuration
python3 -c "
from src.envs.trading_env import TradingEnv
import numpy as np
env = TradingEnv({'dataset_paths': ['data/sample_training_data_simple_20250607_192034.csv']})
print(f'Action space: {env.action_space}')
print(f'Action space type: {type(env.action_space)}')
print(f'Has shape: {hasattr(env.action_space, \"shape\")}')
if hasattr(env.action_space, 'shape'):
    print(f'Shape: {env.action_space.shape}')
"
```

**TODO - Fix Action Space**:
1. [ ] Examine `src/envs/trading_env.py` action space definition
2. [ ] Ensure action space is `gymnasium.spaces.Box` for continuous actions
3. [ ] Verify action space shape matches TD3 agent expectations (should be `(n,)` tuple)
4. [ ] Update action space configuration if using discrete actions
5. [ ] Test TD3 integration after fix

**Files to Examine**:
- `src/envs/trading_env.py` - Action space definition
- `tests/test_td3_integration.py` - Integration test expectations
- `src/agents/td3_agent.py` - Action space requirements

---

### **Issue #2: CNN Model Input Dimension Mismatch**
**Status: CRITICAL BLOCKER** ⚠️  
**Affects: 2 integration tests**

**Problem**:
```python
RuntimeError: Given groups=1, weight of size [1, 2, 1], expected input[1, 5, 5] to have 2 channels, but got 5 channels instead
```

**Root Cause**:
- CNN model expects 2 input channels but receives 5 channels
- Model was trained/configured for different input dimensions than environment provides
- Feature dimension mismatch between model expectations and environment observations

**Investigation Required**:
```python
# Check model and environment dimensions
from src.envs.trading_env import TradingEnv
from src.supervised_model import load_model

# Check environment observation shape
env = TradingEnv({'dataset_paths': ['data/sample_training_data_simple_20250607_192034.csv']})
obs, _ = env.reset()
print(f"Environment obs shape: {obs.shape}")

# Check model expected input dimensions  
# Look at model configuration and input_dim parameter
```

**TODO - Fix Model-Environment Integration**:
1. [ ] Examine environment observation shape in `src/envs/trading_env.py`
2. [ ] Check CNN model input dimension configuration in `src/supervised_model.py`
3. [ ] Verify feature engineering pipeline produces consistent dimensions
4. [ ] Either:
   - [ ] **Option A**: Retrain model with correct input dimensions
   - [ ] **Option B**: Modify environment to match model expectations
   - [ ] **Option C**: Add dimension adapter/transformer layer
5. [ ] Test model predictions with environment observations

**Files to Examine**:
- `src/envs/trading_env.py` - Observation space and feature generation
- `src/supervised_model.py` - CNN input dimension configuration
- `tests/test_trading_env_model_integration.py` - Integration test details

---

## 🔧 **MEDIUM PRIORITY - Code Quality**

### **Issue #3: Test File Imports and Dependencies**
**Status: RESOLVED** ✅ (but document for future)

**What Was Fixed**:
- Updated `gym` imports to `gymnasium` in test files
- Fixed indentation errors in test methods
- Improved test assertions for better reliability

**TODO - Prevent Future Issues**:
1. [ ] Add pre-commit hooks to check for deprecated imports
2. [ ] Create dependency migration checklist
3. [ ] Add linting rules for import statements

---

### **Issue #4: Error Handling in Integration Tests**
**Status: IMPROVEMENT NEEDED** ⚠️

**Problem**: Integration tests fail silently or with unclear error messages

**TODO - Improve Test Diagnostics**:
1. [ ] Add better error messages to integration tests
2. [ ] Include environment and model state information in test failures
3. [ ] Add debug logging to integration test setup
4. [ ] Create test utilities for common debugging operations

---

## 📋 **TESTING STRATEGY IMPROVEMENTS**

### **Issue #5: Test Coverage Gaps**
**Current Coverage**:
- ✅ Unit tests: Excellent (Fast: 13/13, ML: 18/18, TD3: 21/21)
- ⚠️ Integration tests: Poor (1/8 passing)

**TODO - Improve Integration Testing**:
1. [ ] Create isolated integration tests for each component pair:
   - [ ] Model ↔ Environment integration
   - [ ] Agent ↔ Environment integration  
   - [ ] Model ↔ Agent integration
   - [ ] Full pipeline integration
2. [ ] Add integration test setup utilities
3. [ ] Create mock/stub versions for faster integration testing
4. [ ] Add performance benchmarks to integration tests

---

### **Issue #6: Test Data Management**
**Current Issue**: Tests use hardcoded sample data paths

**TODO - Improve Test Data**:
1. [ ] Create test data generation utilities
2. [ ] Add test data validation
3. [ ] Create smaller, faster test datasets
4. [ ] Add test data cleanup procedures

---

## ✅ Sentiment Analysis: Robust Testing Coverage
- All sentiment data sources (robust Yahoo Finance scraping and mock fallback) are now covered by unit tests.
- No NewsAPI.org API key is required; all news sentiment is scraped.
- Tests ensure correct source labeling (`news_scrape`, `news_mock`).
- Run `pytest tests/test_sentiment.py -v` to verify all sentiment integration and fallback logic.

---

## 🎯 **IMMEDIATE ACTION PLAN**

### **Day 1 (Next Session)**:
1. [ ] **Priority 1**: Fix trading environment action space configuration
2. [ ] **Priority 2**: Investigate CNN model input dimension mismatch
3. [ ] **Priority 3**: Run integration tests after fixes

### **Day 2**:
1. [ ] Fix model-environment integration issues
2. [ ] Improve integration test error handling
3. [ ] Validate full pipeline functionality

### **Day 3**:
1. [ ] Complete Phase 2 integration testing
2. [ ] Document fixes and improvements
3. [ ] Prepare for Phase 3 development

---

## 🏆 **SUCCESS CRITERIA FOR ISSUE RESOLUTION**

### **Integration Tests Target**:
- [ ] All TD3 integration tests passing (5/5)
- [ ] All model-environment integration tests passing (2/2)  
- [ ] All Ray integration tests passing (1/1)
- [ ] **Target: 8/8 integration tests passing**

### **Code Quality Target**:
- [ ] No critical dependencies issues
- [ ] Clear error messages in all test failures
- [ ] Comprehensive test coverage documentation
- [ ] Clean, maintainable integration test suite

---

## 📝 **NOTES FROM TESTING SESSION**

### **What Worked Well**:
✅ Core component testing strategy
✅ Systematic approach to dependency issues  
✅ Comprehensive unit test coverage
✅ Clear error identification and documentation

### **What Needs Improvement**:
⚠️ Integration test design and setup
⚠️ Model-environment compatibility validation
⚠️ Error handling and debugging capabilities
⚠️ Test data management and generation

### **Key Learnings**:
1. **Unit tests alone are insufficient** - Need robust integration testing
2. **Environment-model compatibility is critical** - Must validate dimensions early
3. **Dependency migrations require systematic checking** - gym→gymnasium affected multiple files
4. **Clear error messages are essential** - Poor error handling masks root causes

---

*This document will be updated as issues are resolved and new issues are discovered.*
