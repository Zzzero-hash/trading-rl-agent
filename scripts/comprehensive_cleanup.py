#!/usr/bin/env python3
"""
Comprehensive Cleanup Script for Trading RL Agent
Streamlines the codebase by removing redundant files and organizing structure.
"""

from datetime import datetime, timedelta
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List


class TradingRLCleaner:
    """Comprehensive cleanup utility for the trading RL project."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.removed_files = []
        self.cleaned_dirs = []
        self.freed_space = 0

    def cleanup_redundant_files(self) -> None:
        """Remove redundant and obsolete files."""
        print("🗑️  Removing redundant files...")

        # Files to remove - these are redundant or obsolete
        redundant_files = [
            # Root level test files (should be in tests/ directory)
            "test_fix.py",
            "minimal_test.py",
            "quick_integration_test.py",
            # Cleanup scripts (consolidating into this one)
            "cleanup_optimization.py",
            "pipeline_cleanup.py",
            # Redundant optimization files
            "src/optimization/simple_cnn_lstm_optimization.py",
            # Backup files
            "src/agents/__init__.py.bak",
            # Generated files that can be recreated
            "data/sample_training_data_simple_20250607_192034.csv",
            "data/sample_training_data_20250609_195132.csv",
        ]

        for file_path in redundant_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                size_mb = full_path.stat().st_size / (1024 * 1024)
                self.freed_space += size_mb
                self.removed_files.append(str(file_path))
                full_path.unlink()
                print(f"   ✅ Removed: {file_path} ({size_mb:.2f} MB)")

    def cleanup_optimization_results(self, keep_days: int = 7) -> None:
        """Clean up old optimization results."""
        print(f"\n📊 Cleaning optimization results (keeping last {keep_days} days)...")

        directories = ["optimization_results", "ray_results", "results_archive"]
        cutoff_date = datetime.now() - timedelta(days=keep_days)

        for dir_name in directories:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                continue

            print(f"   📁 Processing {dir_name}/")

            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    mod_time = datetime.fromtimestamp(subdir.stat().st_mtime)

                    if mod_time < cutoff_date:
                        size_mb = sum(
                            f.stat().st_size for f in subdir.rglob("*") if f.is_file()
                        ) / (1024 * 1024)
                        self.freed_space += size_mb
                        shutil.rmtree(subdir)
                        print(f"      🗑️  Removed old: {subdir.name} ({size_mb:.1f} MB)")
                    else:
                        size_mb = sum(
                            f.stat().st_size for f in subdir.rglob("*") if f.is_file()
                        ) / (1024 * 1024)
                        print(f"      ✅ Kept recent: {subdir.name} ({size_mb:.1f} MB)")

    def cleanup_cache_files(self) -> None:
        """Remove Python cache and temporary files."""
        print("\n🧹 Cleaning cache and temporary files...")

        cache_patterns = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.coverage",
            "**/htmlcov",
            "**/.DS_Store",
            "**/Thumbs.db",
        ]

        for pattern in cache_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_dir():
                    size_mb = sum(
                        f.stat().st_size for f in path.rglob("*") if f.is_file()
                    ) / (1024 * 1024)
                    self.freed_space += size_mb
                    shutil.rmtree(path)
                    print(
                        f"   🗑️  Removed cache dir: {path.relative_to(self.project_root)}"
                    )
                elif path.is_file():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    self.freed_space += size_mb
                    path.unlink()
                    print(
                        f"   🗑️  Removed cache file: {path.relative_to(self.project_root)}"
                    )

    def organize_test_files(self) -> None:
        """Move any remaining root-level test files to tests/ directory."""
        print("\n📂 Organizing test files...")

        tests_dir = self.project_root / "tests"
        tests_dir.mkdir(exist_ok=True)

        # Find test files in root
        test_files = list(self.project_root.glob("test_*.py"))

        for test_file in test_files:
            target = tests_dir / test_file.name
            if not target.exists():
                shutil.move(str(test_file), str(target))
                print(f"   📁 Moved: {test_file.name} -> tests/")

    def streamline_optimization_structure(self) -> None:
        """Ensure optimization module has clean structure."""
        print("\n⚡ Streamlining optimization structure...")

        opt_dir = self.project_root / "src" / "optimization"
        if not opt_dir.exists():
            return

        # Keep only essential optimization files
        essential_files = [
            "__init__.py",
            "cnn_lstm_optimization.py",
            "rl_optimization.py",
            "model_utils.py",
        ]

        for file in opt_dir.iterdir():
            if file.is_file() and file.name not in essential_files:
                print(f"   🗑️  Would remove non-essential: {file.name}")
                # Commented out for safety - review before removing
                # file.unlink()

    def create_optimization_summary(self) -> None:
        """Create a summary of the streamlined optimization setup."""
        print("\n📋 Creating optimization summary...")

        summary_content = """# Streamlined Optimization Structure

## Core Files
- `src/optimization/cnn_lstm_optimization.py` - Main CNN-LSTM hyperparameter optimization
- `src/optimization/rl_optimization.py` - RL agent hyperparameter optimization
- `src/optimization/model_utils.py` - Model analysis and profiling utilities

## Usage
```python
# CNN-LSTM optimization
from trading_rl_agent.optimization.cnn_lstm_optimization import optimize_cnn_lstm
results = optimize_cnn_lstm(features, targets, num_samples=20)

# RL optimization
from trading_rl_agent.optimization.rl_optimization import optimize_sac_hyperparams
results = optimize_sac_hyperparams(env_config, num_samples=10)

# Model analysis
from trading_rl_agent.optimization.model_utils import get_model_summary
print(get_model_summary(model, input_size=(1, 10)))
```

## Configuration
- Hyperparameter spaces defined in optimization modules
- Ray Tune integration for distributed optimization
- Automatic GPU detection and configuration
"""

        summary_path = self.project_root / "docs" / "OPTIMIZATION_GUIDE.md"
        summary_path.parent.mkdir(exist_ok=True)
        summary_path.write_text(summary_content)
        print(f"   ✅ Created: {summary_path}")

    def run_cleanup(self, keep_optimization_days: int = 7) -> None:
        """Run the complete cleanup process."""
        print("🧹 Starting comprehensive cleanup...")
        print("=" * 60)

        self.cleanup_redundant_files()
        self.cleanup_optimization_results(keep_optimization_days)
        self.cleanup_cache_files()
        self.organize_test_files()
        self.streamline_optimization_structure()
        self.create_optimization_summary()

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 Cleanup Complete!")
        print(f"📁 Removed {len(self.removed_files)} redundant files")
        print(f"💾 Freed {self.freed_space:.1f} MB of disk space")

        if self.removed_files:
            print("\n📋 Removed files:")
            for file in self.removed_files[:10]:  # Show first 10
                print(f"   • {file}")
            if len(self.removed_files) > 10:
                print(f"   ... and {len(self.removed_files) - 10} more")


def main():
    """Main cleanup function."""
    cleaner = TradingRLCleaner()
    cleaner.run_cleanup(keep_optimization_days=7)


if __name__ == "__main__":
    main()
