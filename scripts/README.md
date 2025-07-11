# Scripts Directory

This directory contains utility scripts for managing the ML training project.

## Available Scripts

### `fix_locale.sh`

**Fixes locale warnings in Git operations and development environment**

```bash
# Source to fix locale environment variables
source scripts/fix_locale.sh

# Now run git commands without locale warnings
git status
git commit -m "Your commit message"
```

This script permanently fixes the `LC_ALL: cannot change locale` warnings by setting the locale to the stable `C.utf8` that's available in the dev container.

### `pre-commit-hook.sh`

Pre-commit hook for automatic cleanup before git commits.

**Setup:**

```bash
# Copy to git hooks directory
cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Features:**

- Automatically clears notebook outputs before commits
- Warns about large files
- Checks for temporary files

## Maintenance Schedule

### Daily (Automated)

- Use cleanup script with `--status-only` to monitor storage

### Weekly

- Review and clean up experiment outputs manually
- Remove old archives and temporary files

### Before Major Commits

- Run pre-commit hook or manual cleanup
- Clear all notebook outputs
- Archive any important experimental results

## Storage Guidelines

- **optimization_results/**: Keep < 500MB
- **ray_results/**: Keep < 1GB
- **Notebook outputs**: Always clear before commits
- **Python cache**: Clean regularly

## Dependencies

The cleanup script requires:

- Python 3.6+
- `nbformat` (for notebook cleaning): `pip install nbformat`
- `jupyter` (optional, for nbconvert): `pip install jupyter`

## Troubleshooting

### "jupyter not found"

Install Jupyter: `pip install jupyter`

### "nbformat not available"

Install nbformat: `pip install nbformat`

### Permission denied

Make scripts executable: `chmod +x scripts/*.py scripts/*.sh`
