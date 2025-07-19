# Test Fixing Prompts for Agentic AI

## 1. Initial Assessment Prompt

```
You are a test debugging expert. Your goal is to get all pytest, unit, and integration tests passing in this codebase.

First, perform a comprehensive assessment:
1. Run all tests and capture the output
2. Identify all failing tests and their error messages
3. Categorize failures by type (syntax errors, assertion failures, import errors, etc.)
4. Analyze test dependencies and execution order
5. Check for missing dependencies or configuration issues

Provide a detailed report of:
- Total number of tests
- Number of passing/failing tests
- Failure patterns and common issues
- Test execution time and performance bottlenecks
- Dependencies and environment setup requirements

Start by running: pytest --verbose --tb=short
```

## 2. Environment Setup Prompt

```
Based on the test failures, ensure the testing environment is properly configured:

1. Check if all required dependencies are installed:
   - pytest and pytest plugins
   - test-specific libraries (mock, factory_boy, etc.)
   - database drivers and ORMs
   - API testing libraries (requests, httpx, etc.)

2. Verify configuration files:
   - pytest.ini or pyproject.toml
   - .env files for test environment
   - Database configuration for integration tests
   - Mock service configurations

3. Set up test databases and external services:
   - Create test database schemas
   - Configure test data fixtures
   - Set up mock external APIs
   - Ensure proper cleanup between tests

4. Check file permissions and paths:
   - Test data files accessibility
   - Log file write permissions
   - Temporary directory access

Run: pip install -r requirements.txt && pip install -r requirements-dev.txt
```

## 3. Syntax and Import Error Fixing Prompt

```
Focus on fixing syntax errors and import issues first, as they prevent tests from running:

1. Fix Python syntax errors:
   - Missing colons, parentheses, or brackets
   - Incorrect indentation
   - Invalid variable names or keywords
   - String formatting issues

2. Resolve import errors:
   - Missing __init__.py files
   - Incorrect module paths
   - Circular import dependencies
   - Version compatibility issues
   - Missing package installations

3. Fix module structure issues:
   - Incorrect file organization
   - Missing package declarations
   - Relative vs absolute imports
   - Namespace conflicts

4. Address dependency issues:
   - Version mismatches
   - Missing optional dependencies
   - Platform-specific imports
   - Environment-specific modules

For each fix, verify the change doesn't break other parts of the codebase.
```

## 4. Unit Test Fixing Prompt

```
Now focus on unit test failures. Unit tests should be isolated and test individual components:

1. Fix assertion failures:
   - Check expected vs actual values
   - Verify data types and formats
   - Handle floating-point precision issues
   - Account for time-based assertions
   - Fix string encoding/decoding issues

2. Resolve mock and stub issues:
   - Properly configure mock objects
   - Set up correct return values
   - Handle async mocks appropriately
   - Fix mock side effects and call counts
   - Ensure mocks are reset between tests

3. Fix test isolation problems:
   - Ensure tests don't share state
   - Properly clean up after each test
   - Reset global variables and singletons
   - Clear caches and temporary data
   - Use fresh database connections

4. Address timing and async issues:
   - Handle race conditions
   - Fix timeout configurations
   - Properly await async operations
   - Use appropriate event loop management
   - Fix sleep and delay expectations

5. Fix test data issues:
   - Ensure test data is consistent
   - Handle random data generation
   - Fix date/time sensitive tests
   - Resolve UUID and ID conflicts
   - Clean up test files and directories
```

## 5. Integration Test Fixing Prompt

```
Focus on integration tests that test component interactions and external dependencies:

1. Fix database integration issues:
   - Ensure proper database setup/teardown
   - Fix transaction management
   - Handle database connection pooling
   - Resolve schema migration issues
   - Fix data consistency problems

2. Resolve API integration problems:
   - Fix authentication and authorization
   - Handle rate limiting and timeouts
   - Resolve endpoint URL issues
   - Fix request/response format mismatches
   - Handle API version compatibility

3. Fix external service integration:
   - Configure mock external services
   - Handle service availability issues
   - Fix authentication tokens and keys
   - Resolve network connectivity problems
   - Handle service response variations

4. Address file system integration:
   - Fix file path issues across platforms
   - Handle file permissions and access
   - Resolve temporary file cleanup
   - Fix file encoding and format issues
   - Handle file locking and concurrency

5. Fix environment-specific issues:
   - Handle different OS behaviors
   - Resolve locale and timezone issues
   - Fix environment variable dependencies
   - Handle different Python versions
   - Resolve platform-specific libraries
```

## 6. Performance and Flaky Test Fixing Prompt

```
Address performance issues and flaky tests that pass intermittently:

1. Fix flaky tests:
   - Add retry mechanisms for transient failures
   - Increase timeouts for slow operations
   - Fix race conditions and timing issues
   - Handle resource contention
   - Add proper wait conditions

2. Resolve performance bottlenecks:
   - Optimize slow database queries
   - Reduce unnecessary API calls
   - Fix memory leaks and resource cleanup
   - Optimize test data generation
   - Use parallel test execution where appropriate

3. Fix resource management:
   - Ensure proper cleanup of resources
   - Fix memory leaks in tests
   - Handle file descriptor limits
   - Resolve database connection leaks
   - Fix thread and process cleanup

4. Address test isolation:
   - Ensure tests don't interfere with each other
   - Use unique identifiers for test data
   - Fix shared state between tests
   - Handle global configuration conflicts
   - Resolve cache pollution issues
```

## 7. Test Configuration and Setup Prompt

```
Ensure proper test configuration and setup for all test types:

1. Fix pytest configuration:
   - Configure test discovery patterns
   - Set up proper test markers
   - Fix test collection issues
   - Configure test reporting
   - Set up test filtering

2. Resolve fixture issues:
   - Fix fixture scope problems
   - Handle fixture dependencies
   - Resolve fixture cleanup issues
   - Fix parameterized fixtures
   - Handle fixture conftest.py organization

3. Fix test environment setup:
   - Configure test databases properly
   - Set up test logging
   - Handle environment variables
   - Fix test data initialization
   - Resolve service mocking setup

4. Address test execution issues:
   - Fix test runner configuration
   - Handle test parallelization
   - Resolve test ordering dependencies
   - Fix test timeout settings
   - Handle test result reporting
```

## 8. Final Verification Prompt

```
Perform comprehensive verification to ensure all tests are truly passing:

1. Run all tests multiple times:
   - Execute full test suite 3-5 times
   - Check for any intermittent failures
   - Verify no flaky tests remain
   - Ensure consistent results

2. Validate test coverage:
   - Check that all critical code paths are tested
   - Verify edge cases are covered
   - Ensure error conditions are tested
   - Validate integration points

3. Performance validation:
   - Ensure tests complete in reasonable time
   - Check for memory leaks during test runs
   - Verify resource cleanup is working
   - Monitor for performance regressions

4. Documentation and reporting:
   - Update test documentation if needed
   - Document any test configuration changes
   - Report on test execution statistics
   - Provide recommendations for test maintenance

5. Final checklist:
   - All tests pass consistently
   - No flaky or intermittent failures
   - Test execution time is acceptable
   - Resource usage is reasonable
   - Test coverage is adequate
   - Documentation is up to date

Run: pytest --verbose --tb=long --durations=10 --cov=.
```

## 9. Maintenance and Prevention Prompt

```
Establish practices to prevent future test failures:

1. Set up continuous integration:
   - Configure automated test runs
   - Set up test failure notifications
   - Implement test result reporting
   - Configure test environment management

2. Implement test quality checks:
   - Add linting for test code
   - Set up test coverage requirements
   - Implement test performance monitoring
   - Add test documentation requirements

3. Create test maintenance procedures:
   - Regular test review and cleanup
   - Update test dependencies
   - Refactor outdated test patterns
   - Maintain test data freshness

4. Establish best practices:
   - Test naming conventions
   - Test organization standards
   - Mock and fixture guidelines
   - Error handling patterns

5. Monitor and improve:
   - Track test execution metrics
   - Identify slow or problematic tests
   - Optimize test performance
   - Update test strategies as needed
```

## Usage Instructions

1. **Start with Assessment**: Use the initial assessment prompt to understand the current state
2. **Environment Setup**: Ensure proper test environment configuration
3. **Progressive Fixing**: Work through prompts 3-6 in order, focusing on one category at a time
4. **Configuration**: Use prompt 7 to fix any remaining configuration issues
5. **Verification**: Use prompt 8 to ensure all tests are truly passing
6. **Maintenance**: Use prompt 9 to establish ongoing test quality practices

Each prompt should be used sequentially, and the AI should wait for confirmation that the current category is resolved before moving to the next.