# Trading RL Agent Documentation

This directory contains the Sphinx documentation for the Trading RL Agent project.

## Quick Start

### 1. Install Dependencies

```bash
# Install development dependencies (includes Sphinx)
pip install -r requirements-dev.txt
```

### 2. Build Documentation

```bash
# Build HTML documentation
cd docs
make html

# Or use the convenience script
python build_docs.py
```

### 3. View Documentation

```bash
# Serve documentation locally
make serve

# Or use the convenience script
python build_docs.py --serve
```

Then open http://localhost:8000 in your browser.

## Available Commands

### Using Make

```bash
# Build HTML documentation
make html

# Fast incremental build
make html-fast

# Clean build directory
make clean

# Serve documentation
make serve

# Auto-build with file watching
make autobuild

# Run quality checks
make check

# Build all formats (HTML, PDF, EPUB)
make all
```

### Using the Python Script

```bash
# Build documentation
python build_docs.py

# Fast build
python build_docs.py --fast

# Clean build
python build_docs.py --clean

# Serve documentation
python build_docs.py --serve

# Run quality checks
python build_docs.py --check

# Build, check, and serve
python build_docs.py --all
```

## Documentation Structure

- **Getting Started**: Quick start guides and tutorials
- **Features**: Detailed feature documentation
- **Development**: Setup and development guides
- **Deployment**: Production deployment guides
- **Architecture**: System architecture documentation
- **API Reference**: Auto-generated API documentation
- **Examples**: Code examples and use cases
- **Integration**: Third-party integrations
- **Quality Assurance**: Testing and quality guides
- **Project**: Project management and roadmap

## Configuration

The documentation is configured in `conf.py` with the following key features:

- **AutoDoc**: Automatically generates API documentation from docstrings
- **MyST Parser**: Supports Markdown files alongside reStructuredText
- **Read the Docs Theme**: Modern, responsive theme
- **Type Hints**: Automatic type hint documentation
- **Cross-references**: Links between different documentation sections
- **Search**: Full-text search functionality

## Adding Documentation

### Adding New Pages

1. Create a new `.md` or `.rst` file in the appropriate directory
2. Add it to the appropriate `toctree` in `index.rst`
3. Build the documentation to verify it appears correctly

### API Documentation

API documentation is automatically generated from docstrings in your Python code. To ensure good documentation:

1. Use Google or NumPy style docstrings
2. Include type hints in your function signatures
3. Add examples in docstrings where appropriate
4. Use the `autodoc` directives in `.rst` files

Example docstring:

```python
def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """Calculate returns from price series.

    Args:
        prices: Price series with datetime index
        method: Return calculation method ('log' or 'simple')

    Returns:
        Returns series with same index as prices

    Example:
        >>> prices = pd.Series([100, 110, 105], index=pd.date_range('2023-01-01', periods=3))
        >>> returns = calculate_returns(prices)
        >>> print(returns)
        2023-01-01         NaN
        2023-01-02    0.095310
        2023-01-03   -0.046520
        dtype: float64
    """
    pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure your Python path includes the project root and `src` directory
2. **Missing Dependencies**: Install all required packages with `pip install -r requirements-dev.txt`
3. **Build Errors**: Check the build output for specific error messages
4. **Missing API Documentation**: Ensure your modules have proper `__init__.py` files and docstrings

### Getting Help

- Check the Sphinx documentation: https://www.sphinx-doc.org/
- Review the `conf.py` file for configuration options
- Use `make help` to see all available make targets

## Continuous Integration

The documentation can be built automatically in CI/CD pipelines. The build process:

1. Installs development dependencies
2. Builds HTML documentation
3. Runs quality checks (link checking, doctests, coverage)
4. Can deploy to hosting services like Read the Docs

## Contributing

When contributing to the documentation:

1. Follow the existing style and structure
2. Use clear, concise language
3. Include examples where helpful
4. Update the table of contents in `index.rst` if adding new pages
5. Test your changes by building the documentation locally
