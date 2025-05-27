# MCP Server Memory System Testing Strategy

This directory contains a comprehensive testing strategy for the AI agent memory system components in the MCP server. The tests are organized into three main categories: Unit Tests, Integration Tests, and End-to-End Tests.

## Overview

The testing strategy validates:

1. **Custom Entity Types**: Validation, serialization, and deserialization
2. **Memory Storage Operations**: Storing experiences, prompt patterns, and insights
3. **Memory Retrieval Operations**: Finding relevant memories based on queries
4. **Error Handling**: Robust error handling for invalid inputs and edge cases
5. **MCP Protocol Compliance**: Ensuring the server adheres to MCP standards
6. **Performance**: Response times and memory usage under realistic loads

## Test Categories

### 1. Unit Tests (`test_memory_entity_types.py`, `test_memory_enhanced_server.py`)

**Purpose**: Test individual components in isolation with mocked dependencies.

**Coverage**:
- Custom entity types (ProblemSolution, LessonLearned, CommonMistake, etc.)
- Memory storage functions (`store_problem_solving_experience`)
- Memory retrieval functions (`retrieve_similar_problems`, `get_lessons_for_domain`)
- Input validation and error handling
- Response format validation

**Characteristics**:
- ✅ Fast execution (< 30 seconds)
- ✅ No external dependencies
- ✅ Deterministic results
- ✅ High code coverage

**Run with**:
```bash
python tests/run_tests.py --unit
```

### 2. Integration Tests (`test_integration.py`)

**Purpose**: Test components working together with real Graphiti client and Neo4j.

**Coverage**:
- Full memory workflow (store → retrieve)
- Graphiti integration with custom entity types
- Real database operations
- Memory indexing and search
- Group ID management

**Requirements**:
- Running Neo4j instance
- Valid OpenAI API key (for LLM operations)
- Environment variables configured

**Characteristics**:
- ⏱️ Moderate execution time (1-5 minutes)
- 🔗 Requires external services
- 📊 Tests real data flow
- 🎯 Validates integration points

**Run with**:
```bash
python tests/run_tests.py --integration
```

### 3. End-to-End Tests (`test_e2e_mcp.py`)

**Purpose**: Test the complete system including MCP protocol compliance.

**Coverage**:
- MCP server initialization and tool registration
- HTTP/SSE endpoint availability
- Tool execution through MCP protocol
- Response serialization and format compliance
- Concurrent operations
- Real-world usage scenarios

**Characteristics**:
- 🐌 Slower execution (2-10 minutes)
- 🌐 Tests full system integration
- 📡 Validates MCP protocol compliance
- 🚀 Performance benchmarking

**Run with**:
```bash
python tests/run_tests.py --e2e
```

## Quick Start

### 1. Install Dependencies

```bash
# Install test dependencies
python tests/run_tests.py --install

# Or manually
pip install -e .[test]
```

### 2. Check Environment

```bash
python tests/run_tests.py --check
```

### 3. Run Fast Tests (Development)

```bash
python tests/run_tests.py --fast
```

### 4. Run All Tests

```bash
python tests/run_tests.py --all
```

## Environment Setup

### For Unit Tests Only
No additional setup required.

### For Integration Tests
1. **Start Neo4j**:
   ```bash
   docker run -d \
     --name neo4j-test \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/test \
     neo4j:latest
   ```

2. **Set Environment Variables**:
   ```bash
   export NEO4J_URI=bolt://localhost:7687
   export NEO4J_USER=neo4j
   export NEO4J_PASSWORD=test
   export TEST_OPENAI_API_KEY=your_openai_key
   export TEST_OPENAI_MODEL=gpt-4o-mini
   ```

### For End-to-End Tests
1. Complete integration test setup
2. **Start MCP Server** (optional for some tests):
   ```bash
   python memory_enhanced_server.py --transport sse --use-custom-entities
   ```

## Test Configuration

### Environment Variables

| Variable | Purpose | Required For |
|----------|---------|--------------|
| `NEO4J_URI` | Neo4j connection | Integration, E2E |
| `NEO4J_USER` | Neo4j username | Integration, E2E |
| `NEO4J_PASSWORD` | Neo4j password | Integration, E2E |
| `TEST_OPENAI_API_KEY` | OpenAI API key | Integration, E2E |
| `TEST_OPENAI_MODEL` | OpenAI model name | Integration, E2E |

### Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.unit`: Fast, isolated unit tests
- `@pytest.mark.integration`: Tests requiring external services
- `@pytest.mark.e2e`: End-to-end system tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.asyncio`: Async tests

## Success Criteria

### Unit Tests
- ✅ All entity types validate correctly
- ✅ Functions handle valid inputs properly
- ✅ Error conditions are handled gracefully
- ✅ Response formats match specifications
- ✅ Code coverage > 90%

### Integration Tests
- ✅ Data is stored and retrieved correctly
- ✅ Search returns relevant results
- ✅ Custom entity types are properly indexed
- ✅ Group ID isolation works
- ✅ No data corruption or loss

### End-to-End Tests
- ✅ MCP protocol compliance
- ✅ All tools are properly registered
- ✅ Responses are JSON serializable
- ✅ Concurrent operations work correctly
- ✅ Performance meets benchmarks

## Performance Benchmarks

| Operation | Target Time | Measurement |
|-----------|-------------|-------------|
| Store single experience | < 2 seconds | Integration tests |
| Retrieve similar problems | < 1 second | Integration tests |
| Get domain lessons | < 1 second | Integration tests |
| Store 10 experiences | < 20 seconds | E2E tests |
| Concurrent operations (3) | < 5 seconds | E2E tests |

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   ```
   Solution: Ensure Neo4j is running and credentials are correct
   Check: docker ps, test connection manually
   ```

2. **OpenAI API Key Invalid**
   ```
   Solution: Verify API key is valid and has sufficient credits
   Check: Test key with simple API call
   ```

3. **Import Errors**
   ```
   Solution: Ensure mcp_server directory is in Python path
   Check: Run from correct directory, verify file structure
   ```

4. **Async Test Failures**
   ```
   Solution: Check event loop configuration
   Check: pytest-asyncio version compatibility
   ```

### Debug Mode

Run tests with verbose output and debugging:

```bash
python tests/run_tests.py --unit -v
pytest tests/ -v -s --tb=long
```

### Coverage Reports

Generate detailed coverage reports:

```bash
python tests/run_tests.py --coverage
# View report: open htmlcov/index.html
```

## Continuous Integration

For CI/CD pipelines, use the fast test suite:

```bash
# Quick validation
python tests/run_tests.py --fast

# Full validation (requires services)
python tests/run_tests.py --all
```

## Contributing

When adding new features:

1. **Add unit tests** for new functions/classes
2. **Add integration tests** for new workflows
3. **Update documentation** for new test scenarios
4. **Ensure all tests pass** before submitting PR

### Test File Naming

- `test_*.py`: Test files
- `*_test.py`: Alternative test files
- `conftest.py`: Shared fixtures
- `pytest.ini`: Pytest configuration

### Writing Good Tests

1. **Use descriptive test names**: `test_store_experience_with_valid_data`
2. **Test one thing per test**: Focus on single functionality
3. **Use fixtures**: Share common setup code
4. **Mock external dependencies**: Keep unit tests isolated
5. **Assert meaningful conditions**: Test actual requirements
6. **Handle async properly**: Use `@pytest.mark.asyncio`

## Future Enhancements

- [ ] Property-based testing with Hypothesis
- [ ] Load testing with realistic data volumes
- [ ] Security testing for input validation
- [ ] Compatibility testing across Python versions
- [ ] Memory leak detection for long-running operations
