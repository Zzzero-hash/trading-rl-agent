# Test Fixing Prompts for Agentic AI

A comprehensive set of prompts, scripts, and tools designed to help agentic AI systems systematically get all pytest, unit, and integration tests passing.

## 📋 Overview

This collection provides a structured approach to test debugging and fixing, covering:

- **9 Main Test Fixing Prompts** - Systematic approach to fixing different types of test failures
- **Specialized Domain Prompts** - Specific prompts for database, API, async, frontend, and other testing scenarios
- **Implementation Scripts** - Bash and Python scripts for automated test analysis and fixing
- **Configuration Examples** - Pytest and CI/CD configurations
- **Best Practices** - Guidelines for effective test maintenance

## 🚀 Quick Start

### 1. Initial Assessment
```bash
# Run the assessment script to understand current test state
./initial_assessment.sh
```

### 2. Environment Setup
```bash
# Set up testing environment and dependencies
./setup_test_env.sh
```

### 3. Progressive Test Fixing
```bash
# Use systematic approach to fix tests
./fix_tests_progressive.sh
```

### 4. Detailed Analysis
```bash
# Get comprehensive test analysis
python test_analyzer.py
```

## 📁 File Structure

```
├── test_fixing_prompts.md          # Main test fixing prompts (9 categories)
├── specialized_test_prompts.md     # Domain-specific prompts
├── test_fixing_implementation_guide.md  # Scripts and implementation details
├── README.md                       # This file
├── initial_assessment.sh           # Test assessment script
├── setup_test_env.sh              # Environment setup script
├── fix_tests_progressive.sh       # Progressive fixing script
├── test_analyzer.py               # Python test analysis tool
└── test_fixer.py                  # Python test fixing helper
```

## 🎯 Main Test Fixing Prompts

### 1. Initial Assessment Prompt
- Run all tests and capture output
- Identify failing tests and error messages
- Categorize failures by type
- Analyze dependencies and execution order

### 2. Environment Setup Prompt
- Check required dependencies
- Verify configuration files
- Set up test databases and external services
- Check file permissions and paths

### 3. Syntax and Import Error Fixing Prompt
- Fix Python syntax errors
- Resolve import errors
- Fix module structure issues
- Address dependency issues

### 4. Unit Test Fixing Prompt
- Fix assertion failures
- Resolve mock and stub issues
- Fix test isolation problems
- Address timing and async issues
- Fix test data issues

### 5. Integration Test Fixing Prompt
- Fix database integration issues
- Resolve API integration problems
- Fix external service integration
- Address file system integration
- Fix environment-specific issues

### 6. Performance and Flaky Test Fixing Prompt
- Fix flaky tests
- Resolve performance bottlenecks
- Fix resource management
- Address test isolation

### 7. Test Configuration and Setup Prompt
- Fix pytest configuration
- Resolve fixture issues
- Fix test environment setup
- Address test execution issues

### 8. Final Verification Prompt
- Run all tests multiple times
- Validate test coverage
- Performance validation
- Documentation and reporting

### 9. Maintenance and Prevention Prompt
- Set up continuous integration
- Implement test quality checks
- Create test maintenance procedures
- Establish best practices

## 🔧 Specialized Domain Prompts

### Database Testing
- PostgreSQL/MySQL integration test fixing
- SQLite test environment setup
- Database connection and authentication
- Schema and migration issues
- Transaction management

### API Testing
- REST API integration test fixing
- GraphQL API test fixing
- Authentication and authorization
- Request/response handling
- API mocking

### Async/Concurrent Testing
- Async test fixing
- Threading and multiprocessing
- Race conditions and timing
- Resource management

### Frontend Testing
- Selenium/Playwright test fixing
- React/Vue component testing
- Browser automation
- Cross-browser compatibility

### Performance Testing
- Load testing fixing
- Memory and resource testing
- Performance profiling
- Resource cleanup

### Security Testing
- Authentication testing
- Input validation testing
- Authorization testing
- Security scanning

### Microservices Testing
- Service integration test fixing
- Service discovery
- Inter-service communication
- Distributed system testing

### Data Pipeline Testing
- ETL/Data pipeline test fixing
- Data transformation testing
- Pipeline orchestration
- Data streaming testing

## 🛠️ Implementation Tools

### Bash Scripts

#### `initial_assessment.sh`
```bash
# Comprehensive test assessment
./initial_assessment.sh
```
- Runs all tests and captures output
- Parses results and categorizes failures
- Identifies common issues
- Generates detailed report

#### `setup_test_env.sh`
```bash
# Environment setup
./setup_test_env.sh
```
- Installs dependencies
- Configures pytest
- Sets up test databases
- Creates test directories

#### `fix_tests_progressive.sh`
```bash
# Progressive test fixing
./fix_tests_progressive.sh
```
- Fixes syntax errors first
- Runs unit tests
- Runs integration tests
- Final verification

### Python Tools

#### `test_analyzer.py`
```python
# Detailed test analysis
python test_analyzer.py
```
- Comprehensive test analysis
- Failure categorization
- Performance monitoring
- Detailed reporting

#### `test_fixer.py`
```python
# Automated test fixing
python test_fixer.py
```
- Common issue resolution
- Import error fixing
- Syntax error detection
- Database error handling

## ⚙️ Configuration Examples

### Basic pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

### Advanced pyproject.toml
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",
    "--tb=short",
    "--cov=src",
    "--cov-report=html:htmlcov"
]
markers = [
    "unit: Unit tests",
    "integration: Integration tests"
]
```

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run tests
      run: pytest --verbose --cov=src
```

### GitLab CI
```yaml
test:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest --verbose --cov=src
```

## 📊 Usage Workflow

### For Agentic AI Systems

1. **Start with Assessment**
   ```bash
   ./initial_assessment.sh
   ```

2. **Use Main Prompts Sequentially**
   - Apply prompts 1-9 in order
   - Wait for confirmation before moving to next
   - Use specialized prompts as needed

3. **Apply Specialized Prompts**
   - Identify domain-specific issues
   - Use appropriate specialized prompt
   - Validate fixes with domain-specific tools

4. **Verify and Document**
   - Run final verification
   - Document all changes
   - Set up monitoring

### For Human Developers

1. **Quick Assessment**
   ```bash
   python test_analyzer.py
   ```

2. **Targeted Fixing**
   - Use specific prompts for identified issues
   - Run individual test files for isolation
   - Apply fixes incrementally

3. **Continuous Integration**
   - Set up automated testing
   - Monitor test performance
   - Maintain test quality

## 🎯 Best Practices

### Test Organization
- Organize tests by type (unit, integration, etc.)
- Use descriptive test names
- Group related tests in classes
- Use appropriate test markers

### Test Isolation
- Ensure tests don't share state
- Use fresh data for each test
- Clean up resources properly
- Mock external dependencies

### Performance
- Keep tests fast
- Use parallel execution where appropriate
- Monitor test execution time
- Optimize slow tests

### Maintenance
- Regular test review and cleanup
- Update test dependencies
- Refactor outdated patterns
- Maintain test documentation

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Install missing dependencies
pip install -r requirements.txt
```

#### Database Connection Issues
```bash
# Check database configuration
python -c "from your_app import db; print(db.engine.url)"

# Test database connection
python -c "from your_app import db; db.engine.connect()"
```

#### Test Timeout Issues
```bash
# Increase timeout
pytest --timeout=300

# Run specific slow tests
pytest -m slow --durations=10
```

#### Memory Issues
```bash
# Monitor memory usage
python -m memory_profiler test_file.py

# Check for memory leaks
pytest --memray
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://realpython.com/python-testing/)
- [Test-Driven Development](https://en.wikipedia.org/wiki/Test-driven_development)
- [Continuous Integration Best Practices](https://martinfowler.com/articles/continuousIntegration.html)

## 🤝 Contributing

To contribute to this collection:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the best practices
3. Open an issue with detailed information
4. Provide test output and error messages

---

**Remember**: The goal is not just to make tests pass, but to ensure they provide reliable, meaningful feedback about your code's correctness and quality.
