# Documentation Build Issues - Quick Fix Guide

## Problem

The pre-commit `end-of-file-fixer` hook was failing on Sphinx-generated `.doctree` files in `docs/_build/`.

## Solution Applied

1. **Updated `.pre-commit-config.yaml`** to exclude Sphinx build artifacts:
   - Added `exclude: ^(docs/_build/|\.doctrees/|\.buildinfo$|searchindex\.js$)` to:
     - `end-of-file-fixer`
     - `trailing-whitespace`
     - `mixed-line-ending`

2. **Updated `.gitignore`** to ignore documentation build files:

   ```
   # Documentation build
   docs/_build/
   docs/_autosummary/
   *.doctree
   ```

3. **Removed existing build artifacts** to ensure clean state

## Best Practices

- Always run `make clean` in the docs directory before committing
- The docs Makefile already includes a clean target that removes build artifacts
- Build artifacts should never be committed to version control

## Usage

```bash
# Clean docs before commit
cd docs && make clean

# Or build fresh HTML docs (includes clean)
cd docs && make html
```
