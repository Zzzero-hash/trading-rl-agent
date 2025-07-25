# Trading RL Agent Test Suite

This comprehensive test suite provides robust testing coverage for the Trading RL Agent CLI with focus on CLI functionality, reliability, and user experience.

## Test Organization

### Unit Tests (`tests/unit/`)

- **`test_cli_data_complete.py`** - Complete CLI data command coverage with all parameter combinations
- **`test_cli_train_advanced.py`** - Advanced training command tests including hyperparameter optimization
- **`test_cli_trade_sessions.py`** - Trading session lifecycle management tests
- **`test_cli_error_handling.py`** - Comprehensive error scenario testing
- **`test_cli_argument_validation.py`** - Parameter validation testing for all CLI arguments
- **`test_cli_state_management.py`** - Configuration state persistence and session management
- **`test_cli_user_experience.py`** - Help text quality, error messages, and progress indication

### Integration Tests (`tests/integration/`)

- **`test_cli_real_data.py`** - Real data integration with actual market data sources
- **`test_training_pipeline_integration.py`** - End-to-end training pipeline integration

### Performance Tests (`tests/performance/`)

- **`test_cli_performance.py`** - CLI performance benchmarking and memory usage monitoring

### Security Tests (`tests/security/`)

- **`test_cli_security.py`** - Security testing including credential handling, input sanitization, and injection prevention

### Cross-Platform Tests (`tests/cross_platform/`)

- **`test_cli_cross_platform.py`** - Windows/Linux/macOS compatibility testing

### Property Tests (`tests/property/`)

- **`test_trading_properties.py`** - Property-based testing for trading logic

### Smoke Tests (`tests/smoke/`)

- **`test_comprehensive_smoke.py`** - Quick smoke tests for basic functionality

## Test Coverage Areas

### 1. CLI Command Coverage ✅

- Complete testing of all CLI commands and subcommands
- Parameter validation and edge cases
- Error handling for invalid inputs
- Help text and documentation quality

### 2. Data Pipeline Testing ✅

- Symbol validation and processing
- Configuration file integration
- Multi-format export testing (CSV, Parquet, Feather)
- Cache management and performance
- Real data integration with rate limiting

### 3. Training Pipeline Testing ✅

- CNN-LSTM model training with various configurations
- Hyperparameter optimization testing
- GPU/CPU compatibility
- Model persistence and loading
- Training progress monitoring

### 4. Trading Session Management ✅

- Session start/stop/monitor operations
- State persistence across sessions
- Real-time monitoring capabilities
- Error recovery and cleanup

### 5. Performance & Scalability ✅

- Response time benchmarking
- Memory usage monitoring
- Concurrent operation testing
- Load testing with increasing data sizes

### 6. Security Testing ✅

- API key and credential protection
- Input sanitization and injection prevention
- File path traversal protection
- Configuration security validation

### 7. Cross-Platform Compatibility ✅

- Windows/Linux/macOS path handling
- Environment variable management
- File system operation compatibility
- Character encoding handling

### 8. User Experience ✅

- Help text clarity and completeness
- Error message quality and actionability
- Progress indication and feedback
- Command discoverability

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires network)
pytest tests/integration/ -v -m integration

# Performance tests
pytest tests/performance/ -v -m performance

# Security tests
pytest tests/security/ -v

# Cross-platform tests
pytest tests/cross_platform/ -v
```

### Run Tests by Functionality

```bash
# CLI-specific tests
pytest tests/unit/test_cli_*.py -v

# Data pipeline tests
pytest tests/unit/test_cli_data_complete.py -v

# Training tests
pytest tests/unit/test_cli_train_advanced.py -v
```

## Test Markers

Key markers defined in `pytest.ini`:

- `@pytest.mark.integration` - Integration tests requiring external resources
- `@pytest.mark.slow` - Tests that take longer to execute
- `@pytest.mark.performance` - Performance and benchmarking tests
- `@pytest.mark.network` - Tests requiring network connectivity
- `@pytest.mark.skipif` - Platform-specific tests
- `e2e` - End-to-end tests
- `gpu` - GPU-specific tests
- `memory` - Memory usage tests
- `ml` - Machine learning tests
- `ray` - Ray framework tests
- `regression` - Regression testing
- `security` - Security tests
- `smoke` - Smoke tests
- `unit` - Unit tests

## Mock Strategy

The test suite uses comprehensive mocking to:

- Isolate CLI logic from external dependencies
- Enable fast, reliable test execution
- Test error conditions and edge cases
- Avoid rate limits and API costs
- Ensure reproducible test results

Key mocked components:

- Data pipeline and market data fetching
- Model training and GPU operations
- File system operations
- Network requests and API calls
- Trading session management

## Quality Assurance

All test files pass:

- **mypy** type checking with `--ignore-missing-imports`
- **ruff** linting and formatting
- **pytest** execution with comprehensive assertions

## Test Data Management

- Uses temporary directories for file operations
- Proper cleanup in teardown methods
- Mock data generation for consistent testing
- Small dataset sizes for fast execution

## Coverage Goals

The test suite aims for:

- **High CLI coverage** - All commands, arguments, and workflows
- **Comprehensive error handling** - All error conditions and edge cases
- **Real-world scenarios** - Integration with actual data sources
- **Performance validation** - Response times and resource usage
- **Security compliance** - Protection against common vulnerabilities
- **Cross-platform support** - Windows, Linux, and macOS compatibility

## Contributing

When adding new CLI functionality:

1. Add corresponding unit tests in `tests/unit/`
2. Include integration tests if external resources are involved
3. Test error conditions and edge cases
4. Ensure cross-platform compatibility
5. Add performance tests for resource-intensive operations
6. Include security tests for sensitive operations
7. Update this README with new test descriptions

## Utility Scripts

Utility scripts are available at the repository root:

```bash
./test-fast.sh    # core functionality
./test-ml.sh      # ML specific tests
./test-all.sh     # full suite
```

## Coverage Reporting

```bash
# Generate coverage report
pytest --cov=src --cov-report=html

# Run with coverage thresholds
pytest --cov=src --cov-fail-under=80
```

Expected coverage thresholds are defined in the centralized configuration file (`config.py`). Please refer to that file for the latest coverage expectations.

## Test Maintenance

- Tests are designed to be maintainable and readable
- Mock configurations are reusable across test classes
- Helper methods reduce code duplication
- Clear test names describe the scenario being tested
- Comprehensive docstrings explain test purpose and scope
