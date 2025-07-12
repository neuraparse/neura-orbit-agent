# Contributing to Neura-Orbit-Agent

Thank you for your interest in contributing to Neura-Orbit-Agent! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/neura-orbit-agent.git
   cd neura-orbit-agent
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

5. **Run Tests**
   ```bash
   pytest
   ```

## 📝 Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:
```bash
# Format code
black src tests
isort src tests

# Check linting
flake8 src tests

# Type checking
mypy src
```

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest for testing
- Mock external dependencies in tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=neura_orbit_agent

# Run specific test file
pytest tests/test_screen_capture.py
```

### Documentation

- Use clear, descriptive docstrings
- Follow Google-style docstring format
- Update README.md for significant changes
- Add type hints to all functions

Example docstring:
```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of the function.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is invalid
    """
    pass
```

## 🔧 Project Structure

```
neura-orbit-agent/
├── src/neura_orbit_agent/     # Main package
│   ├── core/                  # Core functionality
│   ├── integrations/          # LLM integrations
│   ├── automation/            # Automation modules
│   ├── security/              # Security components
│   ├── cli/                   # Command line interface
│   ├── api/                   # REST API
│   └── utils/                 # Utilities
├── tests/                     # Test files
├── config/                    # Configuration files
├── docs/                      # Documentation
└── scripts/                   # Utility scripts
```

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - OS and version
   - Python version
   - Package versions (`pip freeze`)

2. **Steps to Reproduce**
   - Clear, numbered steps
   - Expected vs actual behavior
   - Screenshots if applicable

3. **Error Messages**
   - Full error traceback
   - Log files if relevant

## ✨ Feature Requests

For new features:

1. **Check Existing Issues**
   - Search for similar requests
   - Comment on existing issues if relevant

2. **Provide Context**
   - Use case description
   - Why this feature is needed
   - Proposed implementation approach

3. **Consider Scope**
   - Keep features focused
   - Consider breaking large features into smaller parts

## 🔀 Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Test Thoroughly**
   ```bash
   pytest
   black --check src tests
   flake8 src tests
   mypy src
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

## 🔒 Security

- Never commit API keys or secrets
- Use environment variables for sensitive data
- Follow security best practices
- Report security issues privately

## 📋 Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage is maintained
- [ ] Documentation is updated
- [ ] No sensitive data is committed
- [ ] Changes are backwards compatible
- [ ] Performance impact is considered

## 🤝 Community Guidelines

- Be respectful and inclusive
- Help others learn and grow
- Provide constructive feedback
- Follow the code of conduct

## 📞 Getting Help

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time chat (link in README)

## 🎯 Areas for Contribution

We especially welcome contributions in:

- **New LLM Integrations**: Support for additional providers
- **Platform Support**: Enhanced Windows/Linux/macOS compatibility
- **Automation Modules**: New automation capabilities
- **Security Features**: Enhanced security and privacy
- **Documentation**: Tutorials, examples, and guides
- **Testing**: Improved test coverage and integration tests
- **Performance**: Optimization and efficiency improvements

Thank you for contributing to Neura-Orbit-Agent! 🚀
