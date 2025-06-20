# Scripts Directory

This directory contains utility scripts for managing the ML training project.

## Available Scripts

### `cleanup_experiments.py`

Automated cleanup tool for managing experiment outputs and disk space.

**Usage:**

```bash
# Check current storage usage
python scripts/cleanup_experiments.py --status-only

# See what would be cleaned (dry run)
python scripts/cleanup_experiments.py --dry-run --all

# Clean up old experiments (keeps last 7 days)
python scripts/cleanup_experiments.py --all

# Archive important results before cleanup
python scripts/cleanup_experiments.py --archive

# Clean up specific components
python scripts/cleanup_experiments.py --clear-notebooks --dry-run
```

**Features:**

- Removes old Ray Tune experiment directories
- Cleans up optimization result directories
- Clears Python cache files
- Archives best experiment results
- Clears Jupyter notebook outputs
- Shows storage usage statistics

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

- Run full cleanup: `python scripts/cleanup_experiments.py --archive --all`
- Review archived results and remove old archives

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
