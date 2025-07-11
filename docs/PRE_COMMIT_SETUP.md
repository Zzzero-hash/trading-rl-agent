# Pre-commit Configuration

This project uses pre-commit hooks to ensure code quality and consistency. The configuration is defined in `.pre-commit-config.yaml` and is optimized to work with the project's current codebase while maintaining high standards.

## Current Status ✅

All pre-commit hooks are now passing successfully!

## What is configured

- **Code Formatting**: `ruff format` and `pyupgrade` ensure consistent code style.
- **Linting**: `ruff`, `mypy`, `bandit`, and `pydocstyle` catch potential errors, and style issues.
- **Security**: `bandit` and `pip-audit` scan for security vulnerabilities.
- **Notebooks**: `nbqa-ruff` and a custom hook to clear outputs ensure notebooks are clean and consistent.
- **File Integrity**: A variety of hooks check for trailing whitespace, end-of-file issues, and valid file syntax (YAML, JSON, TOML).

## Usage

### Install pre-commit hooks

```bash
pre-commit install
```

### Run all hooks manually

```bash
pre-commit run --all-files
```

### Run specific hook

```bash
pre-commit run ruff --all-files
```

### Run manual hooks (tests and security)

```bash
pre-commit run --hook-stage manual
```

### Update hook versions

```bash
pre-commit autoupdate
```

## Hook Summary

The following hooks are configured and passing:

### Automatic hooks (run on every commit):

1.  **File Integrity**: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-json`, etc.
2.  **Security**: `detect-private-key`, `check-added-large-files`, `bandit`.
3.  **Formatting & Linting**: `ruff`, `ruff-format`, `pyupgrade`, `pydocstyle`, `mypy`.
4.  **Documentation**: `prettier` for Markdown and YAML files.
5.  **Notebooks**: `nbqa-ruff` and `clear-notebook-outputs`.

### Manual hooks (run only when explicitly requested):

1.  **Testing**: Full unit test suite with `pytest`.
2.  **Security**: Vulnerability scanning with `pip-audit`.

## Configuration Files

- `.pre-commit-config.yaml`: Main pre-commit configuration.
- `pyproject.toml`: Contains tool configurations for `ruff`, `mypy`, `bandit`, etc.

---

For legal and safety notes see the [project disclaimer](disclaimer.md).
