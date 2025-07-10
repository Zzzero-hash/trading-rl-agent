#!/usr/bin/env python3
"""
Comprehensive Cleanup Script for Trading RL Agent
Streamlines the codebase by removing redundant files and organizing structure.
"""

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path


class TradingRLCleaner:
    """Comprehensive cleanup utility for the trading RL project."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.removed_files = []
        self.cleaned_dirs = []
        self.freed_space = 0

    def cleanup_optimization_results(self, keep_days: int = 7) -> None:
        """Clean up old optimization results."""
        print(f"\n📊 Cleaning optimization results (keeping last {keep_days} days)...")

        directories = ["optimization_results", "ray_results", "results_archive"]
        cutoff_date = datetime.now(UTC) - timedelta(days=keep_days)

        for dir_name in directories:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                continue

            print(f"   📁 Processing {dir_name}/")

            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    mod_time = datetime.fromtimestamp(
                        subdir.stat().st_mtime,
                        tz=UTC,
                    )

                    if mod_time < cutoff_date:
                        size_mb = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file()) / (1024 * 1024)
                        self.freed_space += size_mb
                        shutil.rmtree(subdir)
                        print(f"      🗑️  Removed old: {subdir.name} ({size_mb:.1f} MB)")
                    else:
                        size_mb = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file()) / (1024 * 1024)
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
                    size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
                    self.freed_space += size_mb
                    shutil.rmtree(path)
                    print(
                        f"   🗑️  Removed cache dir: {path.relative_to(self.project_root)}",
                    )
                elif path.is_file():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    self.freed_space += size_mb
                    path.unlink()
                    print(
                        f"   🗑️  Removed cache file: {path.relative_to(self.project_root)}",
                    )

    def run_cleanup(self, keep_optimization_days: int = 7) -> None:
        """Run the complete cleanup process."""
        print("🧹 Starting comprehensive cleanup...")
        print("=" * 60)

        self.cleanup_optimization_results(keep_optimization_days)
        self.cleanup_cache_files()

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 Cleanup Complete!")
        print(f"💾 Freed {self.freed_space:.1f} MB of disk space")


def main():
    """Main cleanup function."""
    cleaner = TradingRLCleaner()
    cleaner.run_cleanup(keep_optimization_days=7)


if __name__ == "__main__":
    main()
