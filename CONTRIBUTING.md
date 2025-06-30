# Contributing to Trading RL Agent

Thank you for your interest in contributing to the Trading RL Agent project! This document provides guidelines and best practices for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Documentation](#documentation)
- [Release Process](#release-process)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you agree to:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback and solutions
- Respect different viewpoints and experiences
- Report unacceptable behavior to the maintainers

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of reinforcement learning concepts
- Familiarity with PyTorch and financial markets (helpful but not required)

### Initial Setup

1. **Fork the repository**

   ```bash
   git clone https://github.com/yourusername/trading-rl-agent.git
   cd trading-rl-agent
   ```

2. **Set up development environment**

   ```bash
   ./setup-env.sh full
   ```

3. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

4. **Run tests to verify setup**
   ```bash
   ./test-all.sh
   ```

## Development Environment

### Environment Setup

We provide multiple setup options for different development needs:

```bash
# Core dependencies only (fast setup)
./setup-env.sh core

# Add ML dependencies
./setup-env.sh ml

# Full production setup
./setup-env.sh full
```

### Development Tools

- **Python**: 3.9+ (3.12 recommended)
- **Package Manager**: pip with requirements files
- **Testing**: pytest with comprehensive test suite
- **Code Quality**: black, isort, flake8, mypy
- **Documentation**: Sphinx with autodoc
- **Version Control**: Git with pre-commit hooks

### IDE Configuration

#### VS Code (Recommended)

Install the following extensions:

- Python
- Pylance
- Black Formatter
- isort
- GitLens
- Python Docstring Generator

Add to `settings.json`:

```json
{
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "python.sortImports.args": ["--profile", "black"],
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

Configure code style:

1. Go to Settings → Tools → External Tools
2. Add Black and isort configurations
3. Set up code inspections for flake8 and mypy

## Code Standards

### Code Style

We use strict code formatting and linting standards:

- **Formatting**: Black with 88-character line length
- **Import Sorting**: isort with Black profile
- **Linting**: flake8 with max line length 88
- **Type Checking**: mypy for static type analysis

### Code Quality Checks

All code must pass these checks before submission:

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/

# Run all quality checks
python run_comprehensive_tests.py --quality-only
```

### Type Hints

Full type annotation coverage is required:

```python
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

def process_data(
    data: pd.DataFrame,
    config: Dict[str, Union[int, float, str]],
    output_path: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Process trading data with configuration.

    Args:
        data: Input trading data with OHLCV columns
        config: Processing configuration dictionary
        output_path: Optional path to save processed data

    Returns:
        Tuple of processed data and processing statistics

    Raises:
        ValueError: If data is missing required columns
        FileNotFoundError: If output_path parent directory doesn't exist
    """
    # Implementation here
    pass
```

### Docstring Standards

Use Google-style docstrings for all public functions, classes, and modules:

```python
class TradingAgent:
    """Base class for trading agents.

    This class provides the interface and common functionality for all
    trading agents in the system.

    Attributes:
        name: Agent identifier
        config: Agent configuration
        model: Underlying ML model

    Example:
        >>> agent = TradingAgent("td3_agent", config)
        >>> action = agent.select_action(state)
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """Initialize trading agent.

        Args:
            name: Unique identifier for the agent
            config: Configuration dictionary with agent parameters

        Raises:
            ValueError: If config is missing required parameters
        """
        pass

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select trading action based on current state.

        Args:
            state: Current market state observation

        Returns:
            Action vector with trading decisions

        Note:
            This is the main interface method that environments call
            to get trading decisions from the agent.
        """
        pass
```

### File Organization

- **Module Structure**: Follow the existing src/ directory structure
- **Imports**: Group imports in order: standard library, third-party, local
- **Constants**: Define at module level in ALL_CAPS
- **Classes**: One main class per file, with descriptive names
- **Functions**: Keep functions focused and under 50 lines when possible

### Error Handling

Use appropriate exception types and provide helpful error messages:

```python
def load_data(file_path: Path) -> pd.DataFrame:
    """Load trading data from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        data = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Empty data file: {file_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format in {file_path}: {e}")

    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return data
```

## Testing

### Test Organization

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test performance characteristics

### Test Structure

```python
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from pathlib import Path

from src.agents.td3_agent import TD3Agent
from src.agents.configs import TD3Config


class TestTD3Agent:
    """Test suite for TD3Agent class."""

    @pytest.fixture
    def sample_config(self):
        """Provide sample TD3 configuration."""
        return TD3Config(
            learning_rate=3e-4,
            batch_size=64,
            buffer_size=10000
        )

    @pytest.fixture
    def agent(self, sample_config):
        """Create TD3Agent instance for testing."""
        return TD3Agent(
            state_dim=10,
            action_dim=1,
            config=sample_config
        )

    def test_initialization(self, agent, sample_config):
        """Test agent initialization."""
        assert agent.state_dim == 10
        assert agent.action_dim == 1
        assert agent.config == sample_config

    def test_select_action_shape(self, agent):
        """Test action selection returns correct shape."""
        state = np.random.randn(10)
        action = agent.select_action(state)
        assert action.shape == (1,)

    @pytest.mark.parametrize("state_dim,action_dim", [
        (5, 1), (10, 1), (20, 3)
    ])
    def test_different_dimensions(self, state_dim, action_dim, sample_config):
        """Test agent with different state/action dimensions."""
        agent = TD3Agent(state_dim, action_dim, sample_config)
        state = np.random.randn(state_dim)
        action = agent.select_action(state)
        assert action.shape == (action_dim,)

    def test_training_step(self, agent):
        """Test training step execution."""
        # Add training data to buffer
        for _ in range(100):
            state = np.random.randn(10)
            action = np.random.randn(1)
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = False
            agent.replay_buffer.add(state, action, reward, next_state, done)

        # Training should not raise exceptions
        agent.train()

    @patch('torch.save')
    def test_save_model(self, mock_save, agent, tmp_path):
        """Test model saving."""
        save_path = tmp_path / "test_model.pth"
        agent.save(save_path)
        mock_save.assert_called_once()
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m slow          # Slow tests only

# Run with coverage
pytest --cov=src --cov-report=html

# Run performance tests
pytest -m performance --benchmark-only

# Run comprehensive test suite
python run_comprehensive_tests.py
```

### Test Guidelines

- **Test Coverage**: Maintain >90% test coverage
- **Test Naming**: Use descriptive names that explain what is being tested
- **Test Isolation**: Each test should be independent and idempotent
- **Mock External Dependencies**: Use mocks for external services and slow operations
- **Parameterized Tests**: Use pytest.mark.parametrize for multiple input scenarios
- **Fixtures**: Use fixtures for common test data and setup

## Pull Request Process

### Before Submitting

1. **Update your fork**

   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow code standards
   - Add/update tests
   - Update documentation

4. **Run linting**
   ```bash
   flake8 src/ tests/
   ```

5. **Test your changes**
   ```bash
   python run_comprehensive_tests.py
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new trading strategy implementation"
   ```

### Commit Message Format

Use conventional commit format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(agents): add SAC agent implementation
fix(env): correct reward calculation bug
docs(api): update TD3Agent documentation
test(data): add integration tests for data pipeline
```

### Pull Request Template

When submitting a PR, include:

1. **Description**: Clear description of changes
2. **Motivation**: Why this change is needed
3. **Testing**: How you tested the changes
4. **Documentation**: Documentation updates made
5. **Breaking Changes**: Any breaking changes
6. **Checklist**: Completion of required tasks

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: All tests must pass
4. **Documentation**: Documentation must be updated
5. **Performance**: No significant performance regressions

## Issue Guidelines

### Bug Reports

Include the following information:

```markdown
**Bug Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Environment**

- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- Package versions: [paste requirements.txt or pip freeze output]

**Additional Context**
Any other context, logs, or screenshots.
```

### Feature Requests

Use this template:

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Motivation**
Why is this feature needed? What problem does it solve?

**Proposed Solution**
How you envision this feature working.

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

## Documentation

### Documentation Standards

- **API Documentation**: All public APIs must be documented
- **Examples**: Include practical examples for complex features
- **Architecture**: Document design decisions and architecture
- **Tutorials**: Provide learning materials for new users

### Building Documentation

```bash
# Build documentation
cd docs
make html

# Serve documentation locally
python -m http.server 8000 --directory _build/html

# Clean documentation
make clean
```

### Documentation Guidelines

- Use clear, concise language
- Include code examples where helpful
- Keep documentation up-to-date with code changes
- Use proper Sphinx/MyST formatting
- Include type hints in code examples

## Release Process

### Version Management

We use semantic versioning (SemVer):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

1. **Update Version**: Update version in `setup.py` and `__init__.py`
2. **Update Changelog**: Document all changes
3. **Test Release**: Run full test suite
4. **Build Documentation**: Update and build docs
5. **Create Release**: Tag and create GitHub release
6. **Announce**: Notify community of release

### Changelog Format

```markdown
# Changelog

## [1.2.0] - 2025-01-15

### Added

- New SAC agent implementation
- Advanced portfolio management features
- Real-time data streaming support

### Changed

- Improved TD3 agent performance
- Updated data preprocessing pipeline
- Enhanced error handling

### Fixed

- Fixed reward calculation bug in trading environment
- Corrected memory leak in replay buffer
- Fixed type hints in utils module

### Deprecated

- Old configuration format (use new YAML format)

### Removed

- Removed deprecated legacy agent classes

### Security

- Updated dependencies to fix security vulnerabilities
```

## Getting Help

- **Documentation**: Check the documentation first
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Community**: Join our community channels for real-time help

## Recognition

Contributors will be recognized in:

- GitHub contributor list
- Release notes
- Documentation acknowledgments
- Annual contributor highlights

Thank you for contributing to Trading RL Agent!
