# Specialized Test Fixing Prompts

## Database Testing Prompts

### PostgreSQL/MySQL Integration Test Fixing
```
Focus on database-specific integration test issues:

1. Fix connection and authentication:
   - Verify database credentials and permissions
   - Check connection string format
   - Handle SSL/TLS configuration
   - Resolve connection pooling issues

2. Fix schema and migration issues:
   - Ensure test database schema is up to date
   - Handle migration conflicts
   - Fix table creation/dropping in tests
   - Resolve constraint and index issues

3. Fix transaction management:
   - Ensure proper rollback after tests
   - Handle nested transactions
   - Fix isolation level conflicts
   - Resolve deadlock scenarios

4. Fix data consistency:
   - Handle timezone differences
   - Fix character encoding issues
   - Resolve numeric precision problems
   - Handle JSON/JSONB field comparisons

Run: pytest tests/integration/test_database.py -v
```

### SQLite Test Environment Setup
```
Configure SQLite for fast, isolated testing:

1. Set up in-memory database:
   - Configure SQLite for test isolation
   - Set up proper connection handling
   - Enable foreign key constraints
   - Configure WAL mode for better performance

2. Fix test data setup:
   - Create minimal test schemas
   - Set up test data fixtures
   - Handle SQLite-specific data types
   - Fix datetime handling differences

3. Resolve SQLite limitations:
   - Handle missing PostgreSQL/MySQL features
   - Fix case sensitivity issues
   - Resolve transaction isolation differences
   - Handle concurrent access limitations

4. Optimize for speed:
   - Use in-memory databases
   - Disable unnecessary features
   - Optimize query patterns
   - Use appropriate indexes
```

## API Testing Prompts

### REST API Integration Test Fixing
```
Fix REST API integration test issues:

1. Fix authentication and authorization:
   - Handle token generation and validation
   - Fix session management
   - Resolve permission checking
   - Handle API key authentication

2. Fix request/response handling:
   - Validate request format and headers
   - Handle response status codes
   - Fix JSON serialization/deserialization
   - Resolve content-type mismatches

3. Fix endpoint testing:
   - Test all HTTP methods (GET, POST, PUT, DELETE)
   - Handle query parameters and path variables
   - Fix request body validation
   - Test error responses and edge cases

4. Fix API mocking:
   - Mock external API dependencies
   - Handle rate limiting simulation
   - Fix timeout and retry logic
   - Resolve authentication mocking

Run: pytest tests/api/ -v --tb=short
```

### GraphQL API Test Fixing
```
Fix GraphQL-specific test issues:

1. Fix query and mutation testing:
   - Validate GraphQL schema
   - Test query execution
   - Handle mutation side effects
   - Fix subscription testing

2. Fix GraphQL-specific issues:
   - Handle N+1 query problems
   - Fix resolver testing
   - Resolve context and authentication
   - Handle error handling and validation

3. Fix GraphQL mocking:
   - Mock GraphQL resolvers
   - Handle schema stitching
   - Fix directive testing
   - Resolve federation testing

4. Fix performance testing:
   - Test query complexity
   - Handle depth limiting
   - Fix rate limiting
   - Test caching strategies
```

## Async/Concurrent Testing Prompts

### Async Test Fixing
```
Fix async and concurrent test issues:

1. Fix async test execution:
   - Properly await async operations
   - Handle event loop management
   - Fix async context managers
   - Resolve async fixture issues

2. Fix async mocking:
   - Mock async functions and methods
   - Handle async side effects
   - Fix async timeout testing
   - Resolve async error handling

3. Fix concurrent testing:
   - Handle race conditions
   - Fix thread safety testing
   - Resolve process isolation
   - Handle shared resource access

4. Fix async performance:
   - Test async performance characteristics
   - Handle async resource cleanup
   - Fix async memory leaks
   - Resolve async deadlocks

Run: pytest tests/async/ -v --asyncio-mode=auto
```

### Threading and Multiprocessing Test Fixing
```
Fix threading and multiprocessing test issues:

1. Fix thread safety testing:
   - Test shared data access
   - Handle thread synchronization
   - Fix deadlock detection
   - Resolve race condition testing

2. Fix multiprocessing issues:
   - Handle process isolation
   - Fix shared memory access
   - Resolve inter-process communication
   - Handle process cleanup

3. Fix concurrent resource access:
   - Test file system concurrency
   - Handle database connection pooling
   - Fix network connection sharing
   - Resolve cache access patterns

4. Fix performance under load:
   - Test under concurrent load
   - Handle resource exhaustion
   - Fix memory usage patterns
   - Resolve CPU utilization
```

## Frontend Testing Prompts

### Selenium/Playwright Test Fixing
```
Fix browser automation test issues:

1. Fix browser setup and configuration:
   - Configure browser drivers
   - Handle browser version compatibility
   - Fix headless mode configuration
   - Resolve browser-specific issues

2. Fix element interaction:
   - Handle dynamic content loading
   - Fix element selection strategies
   - Resolve timing and wait issues
   - Handle JavaScript execution

3. Fix test data and state:
   - Set up test user accounts
   - Handle database state for UI tests
   - Fix file upload testing
   - Resolve session management

4. Fix cross-browser compatibility:
   - Test across different browsers
   - Handle browser-specific behaviors
   - Fix responsive design testing
   - Resolve accessibility testing

Run: pytest tests/ui/ -v --headed
```

### React/Vue Component Test Fixing
```
Fix frontend component test issues:

1. Fix component rendering:
   - Handle component mounting/unmounting
   - Fix prop validation testing
   - Resolve state management testing
   - Handle lifecycle method testing

2. Fix user interaction testing:
   - Test click and input events
   - Handle form submission testing
   - Fix navigation testing
   - Resolve keyboard and mouse events

3. Fix async component behavior:
   - Handle API call testing
   - Fix loading state testing
   - Resolve error state handling
   - Handle component updates

4. Fix styling and layout testing:
   - Test CSS class application
   - Handle responsive design testing
   - Fix accessibility testing
   - Resolve visual regression testing
```

## Performance Testing Prompts

### Load Testing Fixing
```
Fix performance and load testing issues:

1. Fix test data generation:
   - Generate realistic test data
   - Handle data volume scaling
   - Fix data consistency
   - Resolve data cleanup

2. Fix performance metrics:
   - Measure response times accurately
   - Handle throughput testing
   - Fix resource usage monitoring
   - Resolve performance baselines

3. Fix load simulation:
   - Simulate realistic user behavior
   - Handle concurrent user simulation
   - Fix ramp-up and ramp-down
   - Resolve peak load testing

4. Fix performance analysis:
   - Identify performance bottlenecks
   - Handle performance regression detection
   - Fix performance reporting
   - Resolve performance optimization

Run: pytest tests/performance/ -v --benchmark-only
```

### Memory and Resource Testing
```
Fix memory and resource testing issues:

1. Fix memory leak detection:
   - Monitor memory usage patterns
   - Handle garbage collection testing
   - Fix memory profiling
   - Resolve memory cleanup verification

2. Fix resource management:
   - Test file descriptor limits
   - Handle database connection limits
   - Fix network connection management
   - Resolve thread and process limits

3. Fix resource cleanup:
   - Verify proper resource disposal
   - Handle cleanup in error scenarios
   - Fix resource monitoring
   - Resolve cleanup timing issues

4. Fix performance profiling:
   - Profile CPU usage patterns
   - Handle I/O performance testing
   - Fix network performance testing
   - Resolve disk usage monitoring
```

## Security Testing Prompts

### Security Test Fixing
```
Fix security testing issues:

1. Fix authentication testing:
   - Test password validation
   - Handle session management
   - Fix token validation
   - Resolve permission testing

2. Fix input validation testing:
   - Test SQL injection prevention
   - Handle XSS prevention
   - Fix CSRF protection
   - Resolve input sanitization

3. Fix authorization testing:
   - Test role-based access control
   - Handle privilege escalation
   - Fix resource access control
   - Resolve API security testing

4. Fix security scanning:
   - Run security vulnerability scans
   - Handle dependency security testing
   - Fix code security analysis
   - Resolve security compliance testing

Run: pytest tests/security/ -v --tb=long
```

## Microservices Testing Prompts

### Service Integration Test Fixing
```
Fix microservices integration test issues:

1. Fix service discovery:
   - Handle service registration
   - Fix service endpoint resolution
   - Resolve load balancing testing
   - Handle service health checks

2. Fix inter-service communication:
   - Test API contracts
   - Handle message queuing
   - Fix event-driven testing
   - Resolve synchronous/asynchronous calls

3. Fix service mocking:
   - Mock external service dependencies
   - Handle service response simulation
   - Fix service failure testing
   - Resolve service timeout testing

4. Fix distributed system testing:
   - Handle eventual consistency
   - Fix distributed transaction testing
   - Resolve network partition testing
   - Handle service mesh testing
```

## Data Pipeline Testing Prompts

### ETL/Data Pipeline Test Fixing
```
Fix data pipeline and ETL test issues:

1. Fix data transformation testing:
   - Test data format conversions
   - Handle data validation rules
   - Fix data quality checks
   - Resolve data mapping testing

2. Fix pipeline orchestration:
   - Test workflow dependencies
   - Handle error recovery
   - Fix retry mechanisms
   - Resolve pipeline monitoring

3. Fix data storage testing:
   - Test data warehouse operations
   - Handle data lake testing
   - Fix data archival testing
   - Resolve data backup testing

4. Fix data streaming testing:
   - Test real-time data processing
   - Handle stream processing
   - Fix event sourcing testing
   - Resolve data synchronization
```

## Usage Guidelines

1. **Identify the specific testing domain** from the failing tests
2. **Select the appropriate specialized prompt** based on the test type
3. **Use in conjunction with the main test fixing prompts** for comprehensive coverage
4. **Apply domain-specific best practices** and tools
5. **Validate fixes** using domain-specific validation methods
6. **Document domain-specific configurations** and requirements

These specialized prompts should be used when the general test fixing prompts don't address specific domain requirements or when dealing with complex testing scenarios.