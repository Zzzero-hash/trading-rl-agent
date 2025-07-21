# Advanced Testing Framework

This document describes the comprehensive advanced testing framework implemented for the Trading RL Agent project. The framework includes property-based testing, chaos engineering, load testing, contract testing, and data quality validation.

## 🎯 Overview

The advanced testing framework provides multiple layers of testing to ensure system reliability, performance, and data quality:

- **Property-based Testing**: Mathematical invariants and edge case discovery
- **Chaos Engineering**: System resilience under failure conditions
- **Load Testing**: Performance validation under various load scenarios
- **Contract Testing**: API compatibility and service integration
- **Data Quality Testing**: Data integrity and validation

## 📋 Quick Start

### Installation

Install the advanced testing dependencies:

```bash
pip install -r requirements-dev.txt
```

### Running Tests

#### All Advanced Tests

```bash
# Using the test runner script
python scripts/run_advanced_tests.py

# Using Makefile
make advanced-tests
```

#### Individual Test Types

```bash
# Property-based tests
make property

# Chaos engineering tests
make chaos

# Load tests
make load

# Contract tests
make contract

# Data quality tests
make data-quality
```

## 🧪 Property-Based Testing

Property-based testing uses Hypothesis to automatically generate test cases and discover edge cases that traditional unit tests might miss.

### Key Features

- **Mathematical Invariants**: Tests mathematical properties that should always hold
- **Edge Case Discovery**: Automatically finds boundary conditions and edge cases
- **Data Generation**: Generates realistic test data for complex scenarios
- **Shrinking**: Automatically simplifies failing test cases for easier debugging

### Example Tests

```python
@given(
    initial_cash=st.floats(min_value=1000, max_value=1000000),
    num_assets=st.integers(min_value=1, max_value=10),
    num_days=st.integers(min_value=10, max_value=100)
)
@settings(max_examples=50, deadline=None)
def test_portfolio_value_never_negative(self, initial_cash, num_assets, num_days):
    """Property: Portfolio value should never go negative."""
    # Test implementation
```

### Running Property Tests

```bash
pytest tests/property/ -v -m property
```

## 🌪️ Chaos Engineering

Chaos engineering tests verify system resilience by simulating various failure scenarios.

### Test Categories

1. **Network Failures**: Data feed interruptions, API rate limits
2. **Resource Issues**: Memory pressure, CPU intensive operations
3. **Data Quality**: Missing data, outliers, inconsistent data
4. **Trading Engine**: Order execution failures, market data delays
5. **Concurrent Operations**: Race conditions, thread safety
6. **Recovery**: System recovery after failures

### Example Tests

```python
@pytest.mark.chaos
def test_data_feed_interruption(self):
    """Test system behavior when data feed is interrupted."""
    # Simulate network failure
    with patch.object(data_manager, '_fetch_market_data',
                     side_effect=Exception("Network error")):
        # Verify system handles error gracefully
        assert data_manager.is_running
```

### Running Chaos Tests

```bash
pytest tests/chaos/ -v -m chaos
```

## 📊 Load Testing

Load testing uses Locust to simulate various user scenarios and measure system performance.

### User Types

1. **TradingSystemUser**: Regular trading operations
2. **HighFrequencyTrader**: High-frequency trading scenarios
3. **RiskManagerUser**: Risk management operations
4. **DataAnalystUser**: Data analysis and backtesting

### Load Scenarios

- **Normal Load**: 50 users, typical trading hours
- **High Load**: 200 users, market open/close
- **Stress Test**: 500 users, peak load testing
- **Spike Test**: 1000 users, sudden load spikes

### Running Load Tests

```bash
# Basic load test
locust -f tests/load/locustfile.py --headless -u 10 -r 1 --run-time 30s

# Interactive mode
locust -f tests/load/locustfile.py
```

### Load Test Configuration

```python
class LoadTestConfig:
    @staticmethod
    def get_normal_load_config():
        return {
            "users": 50,
            "spawn_rate": 5,
            "run_time": "10m"
        }
```

## 📋 Contract Testing

Contract testing ensures API compatibility between services using consumer-driven contracts.

### Test Categories

1. **Market Data API**: Data retrieval contracts
2. **Trading API**: Order placement and management
3. **Portfolio API**: Portfolio status and updates
4. **Risk API**: Risk metrics and calculations
5. **Error Handling**: Error response contracts

### Example Contracts

```python
@pytest.mark.contract
def test_get_market_data_contract(self, market_data_consumer):
    """Test contract for getting market data."""
    expected_response = {
        'symbol': 'AAPL',
        'price': Like(150.0),
        'volume': Like(1000000),
        'timestamp': Term(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z',
                         '2024-01-01T00:00:00Z')
    }

    (market_data_consumer
     .given('AAPL market data is available')
     .upon_receiving('a request for AAPL market data')
     .with_request('GET', '/api/v1/market-data/AAPL')
     .will_respond_with(200, body=expected_response))
```

### Running Contract Tests

```bash
pytest tests/contract/ -v -m contract
```

## 🔍 Data Quality Testing

Data quality testing validates data integrity using Great Expectations and Pandera.

### Validation Types

1. **Schema Validation**: Data structure and types
2. **Business Rules**: Domain-specific constraints
3. **Statistical Properties**: Data distribution and quality metrics
4. **Real-time Validation**: Data freshness and consistency

### Example Validations

```python
@pytest.mark.data_quality
def test_market_data_schema_validation(self, sample_market_data):
    """Test market data schema using Pandera."""
    class MarketDataSchema(pa.SchemaModel):
        date: Series[datetime] = pa.Field(ge=datetime(2020, 1, 1))
        symbol: Series[str] = pa.Field(str_length=pa.Field(ge=1, le=10))
        open: Series[float] = pa.Field(ge=0)
        high: Series[float] = pa.Field(ge=0)

        @pa.check("high >= open")
        def high_greater_than_open(self, df: DataFrame) -> Series[bool]:
            return df["high"] >= df["open"]
```

### Running Data Quality Tests

```bash
pytest tests/data_quality/ -v -m data_quality
```

## 🚀 Test Runner

The advanced test runner provides a unified interface for running all test types.

### Usage

```bash
# Run all tests
python scripts/run_advanced_tests.py

# Run specific test type
python scripts/run_advanced_tests.py --test-type property

# Save detailed report
python scripts/run_advanced_tests.py --save-report

# Verbose output
python scripts/run_advanced_tests.py --verbose
```

### Test Types

- `property`: Property-based tests
- `chaos`: Chaos engineering tests
- `load`: Load tests
- `contract`: Contract tests
- `data-quality`: Data quality tests
- `all`: All test types (default)

### Report Generation

The test runner generates comprehensive reports including:

- Test execution results
- Performance metrics
- Failure analysis
- Recommendations for improvement

## 📈 Continuous Integration

### GitHub Actions Integration

The advanced testing framework integrates with GitHub Actions for automated testing:

```yaml
- name: Run Advanced Tests
  run: |
    python scripts/run_advanced_tests.py --save-report
    python scripts/run_advanced_tests.py --test-type load
```

### Pre-commit Hooks

Add advanced testing to pre-commit hooks:

```yaml
- repo: local
  hooks:
    - id: advanced-tests
      name: Run Advanced Tests
      entry: python scripts/run_advanced_tests.py --test-type property
      language: system
      pass_filenames: false
      stages: [manual]
```

## 🔧 Configuration

### Hypothesis Configuration

Configure Hypothesis behavior in `pytest.ini`:

```ini
[hypothesis]
hypothesis_profile = ci
```

### Locust Configuration

Configure load testing parameters:

```python
# tests/load/locustfile.py
class LoadTestConfig:
    @staticmethod
    def get_stress_test_config():
        return {
            "users": 500,
            "spawn_rate": 50,
            "run_time": "3m"
        }
```

### Great Expectations Configuration

Configure data quality expectations:

```python
# tests/data_quality/test_data_quality.py
ge_df = ge.from_pandas(sample_market_data)
ge_df.expect_column_values_to_be_between("price", 1, 10000)
```

## 📊 Monitoring and Metrics

### Test Metrics

Track key metrics for each test type:

- **Property Tests**: Number of examples, shrinking efficiency
- **Chaos Tests**: Failure recovery time, system stability
- **Load Tests**: Response times, throughput, error rates
- **Contract Tests**: API compatibility, service integration
- **Data Quality Tests**: Data completeness, accuracy, consistency

### Performance Baselines

Establish performance baselines:

```python
# Load test performance targets
PERFORMANCE_TARGETS = {
    'response_time_p95': 200,  # ms
    'throughput': 1000,        # requests/second
    'error_rate': 0.01,        # 1%
}
```

## 🛠️ Troubleshooting

### Common Issues

1. **Property Test Failures**
   - Check mathematical invariants
   - Review test data generation
   - Examine shrinking output

2. **Chaos Test Failures**
   - Verify error handling
   - Check system recovery mechanisms
   - Review resource management

3. **Load Test Failures**
   - Monitor system resources
   - Check network connectivity
   - Review performance bottlenecks

4. **Contract Test Failures**
   - Verify API specifications
   - Check service compatibility
   - Review error handling

5. **Data Quality Failures**
   - Validate data sources
   - Check data processing pipelines
   - Review business rules

### Debugging Tips

- Use `--verbose` flag for detailed output
- Check test logs for specific failure details
- Review generated reports for insights
- Use interactive debugging for complex issues

## 📚 Best Practices

### Test Design

1. **Property Tests**
   - Focus on mathematical invariants
   - Use realistic data generation
   - Test edge cases and boundaries

2. **Chaos Tests**
   - Simulate realistic failure scenarios
   - Test recovery mechanisms
   - Verify graceful degradation

3. **Load Tests**
   - Use realistic user scenarios
   - Test various load patterns
   - Monitor system resources

4. **Contract Tests**
   - Define clear API contracts
   - Test error scenarios
   - Verify service integration

5. **Data Quality Tests**
   - Validate data schemas
   - Test business rules
   - Monitor data freshness

### Maintenance

1. **Regular Updates**
   - Update test dependencies
   - Review and update test cases
   - Monitor test performance

2. **Documentation**
   - Keep test documentation current
   - Document test scenarios
   - Maintain troubleshooting guides

3. **Monitoring**
   - Track test execution metrics
   - Monitor test reliability
   - Review failure patterns

## 🔮 Future Enhancements

### Planned Features

1. **AI-Powered Test Generation**
   - Automated test case generation
   - Intelligent test prioritization
   - Adaptive test strategies

2. **Enhanced Monitoring**
   - Real-time test metrics
   - Predictive failure detection
   - Automated alerting

3. **Integration Testing**
   - End-to-end test scenarios
   - Multi-service integration
   - Performance regression testing

4. **Security Testing**
   - Vulnerability scanning
   - Penetration testing
   - Security compliance validation

## 📞 Support

For questions or issues with the advanced testing framework:

1. Check the troubleshooting section
2. Review test documentation
3. Examine test logs and reports
4. Consult the development team

## 📄 License

This advanced testing framework is part of the Trading RL Agent project and follows the same MIT license.
