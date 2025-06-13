#!/usr/bin/env python3
"""
Automated cleanup script for Jupyter notebook training outputs.

This script helps manage disk space by cleaning up old training results,
temporary files, and experiment outputs while preserving important data.
"""

import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OutputCleaner:
    """Manages cleanup of ML training outputs and temporary files."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.cleanup_stats = {
            "files_deleted": 0,
            "dirs_deleted": 0,
            "space_freed_mb": 0
        }
    
    def get_directory_size(self, path: Path) -> float:
        """Calculate directory size in MB."""
        total_size = 0
        try:
            for entry in path.rglob("*"):
                if entry.is_file():
                    total_size += entry.stat().st_size
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not calculate size for {path}: {e}")
        return total_size / (1024 * 1024)  # Convert to MB
    
    def is_older_than(self, path: Path, days: int) -> bool:
        """Check if path is older than specified days."""
        try:
            file_time = datetime.fromtimestamp(path.stat().st_mtime)
            cutoff_time = datetime.now() - timedelta(days=days)
            return file_time < cutoff_time
        except (OSError, PermissionError):
            return False
    
    def clean_optimization_results(self, days: int = 7, dry_run: bool = False) -> None:
        """Clean old hyperparameter optimization results."""
        opt_results_dir = self.project_root / "optimization_results"
        if not opt_results_dir.exists():
            return
        
        logger.info(f"Cleaning optimization results older than {days} days...")
        
        # Clean individual files
        patterns = [
            "best_*_config_*.json",
            "*_hparam_results_*.json",
            "*_trials_*.csv",
            "*.png",
            "*.jpg"
        ]
        
        for pattern in patterns:
            for file_path in opt_results_dir.glob(pattern):
                if self.is_older_than(file_path, days):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if dry_run:
                        logger.info(f"Would delete: {file_path} ({size_mb:.1f} MB)")
                    else:
                        logger.info(f"Deleting: {file_path} ({size_mb:.1f} MB)")
                        file_path.unlink()
                        self.cleanup_stats["files_deleted"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
        
        # Clean experiment directories
        for dir_path in opt_results_dir.glob("hparam_opt_*"):
            if dir_path.is_dir() and self.is_older_than(dir_path, days):
                size_mb = self.get_directory_size(dir_path)
                if dry_run:
                    logger.info(
                        f"Would delete directory: {dir_path} ({size_mb:.1f} MB)"
                    )
                else:
                    logger.info(
                        f"Deleting directory: {dir_path} ({size_mb:.1f} MB)"
                    )
                    shutil.rmtree(dir_path)
                    self.cleanup_stats["dirs_deleted"] += 1
                    self.cleanup_stats["space_freed_mb"] += size_mb
    
    def clean_ray_results(self, days: int = 7, dry_run: bool = False) -> None:
        """Clean old Ray Tune results."""
        ray_results_dir = self.project_root / "ray_results"
        if not ray_results_dir.exists():
            return
        
        logger.info(f"Cleaning Ray results older than {days} days...")
        
        for dir_path in ray_results_dir.iterdir():
            if dir_path.is_dir() and self.is_older_than(dir_path, days):
                size_mb = self.get_directory_size(dir_path)
                if dry_run:
                    logger.info(
                        f"Would delete directory: {dir_path} ({size_mb:.1f} MB)"
                    )
                else:
                    logger.info(
                        f"Deleting directory: {dir_path} ({size_mb:.1f} MB)"
                    )
                    shutil.rmtree(dir_path)
                    self.cleanup_stats["dirs_deleted"] += 1
                    self.cleanup_stats["space_freed_mb"] += size_mb
    
    def clean_model_checkpoints(self, days: int = 14, dry_run: bool = False) -> None:
        """Clean old model checkpoints (be more conservative)."""
        models_dir = self.project_root / "models"
        if not models_dir.exists():
            return
        
        logger.info(f"Cleaning model checkpoints older than {days} days...")
        
        # Only clean temporary/intermediate checkpoints
        patterns = [
            "temp_model_*",
            "checkpoint_*.pth",
            "*_epoch_*.pth"
        ]
        
        for pattern in patterns:
            for file_path in models_dir.glob(pattern):
                if self.is_older_than(file_path, days):
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    if dry_run:
                        logger.info(f"Would delete: {file_path} ({size_mb:.1f} MB)")
                    else:
                        logger.info(f"Deleting: {file_path} ({size_mb:.1f} MB)")
                        file_path.unlink()
                        self.cleanup_stats["files_deleted"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
    
    def clean_logs(self, days: int = 30, dry_run: bool = False) -> None:
        """Clean old log files."""
        logs_dir = self.project_root / "logs"
        if not logs_dir.exists():
            return
        
        logger.info(f"Cleaning log files older than {days} days...")
        
        for log_file in logs_dir.glob("*.log"):
            if self.is_older_than(log_file, days):
                size_mb = log_file.stat().st_size / (1024 * 1024)
                if dry_run:
                    logger.info(f"Would delete: {log_file} ({size_mb:.1f} MB)")
                else:
                    logger.info(f"Deleting: {log_file} ({size_mb:.1f} MB)")
                    log_file.unlink()
                    self.cleanup_stats["files_deleted"] += 1
                    self.cleanup_stats["space_freed_mb"] += size_mb
    
    def clean_temp_files(self, dry_run: bool = False) -> None:
        """Clean temporary files and directories."""
        logger.info("Cleaning temporary files...")
        
        temp_patterns = [
            "temp_*",
            "*.tmp",
            "*.temp",
            "__pycache__/*",
            ".pytest_cache/*"
        ]
        
        for pattern in temp_patterns:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    if dry_run:
                        logger.info(f"Would delete: {path} ({size_mb:.1f} MB)")
                    else:
                        logger.info(f"Deleting: {path} ({size_mb:.1f} MB)")
                        path.unlink()
                        self.cleanup_stats["files_deleted"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
                elif path.is_dir():
                    size_mb = self.get_directory_size(path)
                    if dry_run:
                        logger.info(
                            f"Would delete directory: {path} ({size_mb:.1f} MB)"
                        )
                    else:
                        logger.info(
                            f"Deleting directory: {path} ({size_mb:.1f} MB)"
                        )
                        shutil.rmtree(path)
                        self.cleanup_stats["dirs_deleted"] += 1
                        self.cleanup_stats["space_freed_mb"] += size_mb
    
    def archive_best_configs(self, dry_run: bool = False) -> None:
        """Archive best configurations to a separate directory."""
        opt_results_dir = self.project_root / "optimization_results"
        archive_dir = self.project_root / "configs" / "archived"
        
        if not opt_results_dir.exists():
            return
        
        if not dry_run:
            archive_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Archiving best configurations...")
        
        # Find all best config files
        best_configs = list(opt_results_dir.glob("best_*_config_*.json"))
        
        for config_file in best_configs:
            if dry_run:
                logger.info(f"Would archive: {config_file}")
            else:
                archive_path = archive_dir / config_file.name
                shutil.copy2(config_file, archive_path)
                logger.info(f"Archived: {config_file} -> {archive_path}")
    
    def print_summary(self) -> None:
        """Print cleanup summary."""
        logger.info("=" * 50)
        logger.info("CLEANUP SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Files deleted: {self.cleanup_stats['files_deleted']}")
        logger.info(f"Directories deleted: {self.cleanup_stats['dirs_deleted']}")
        logger.info(f"Space freed: {self.cleanup_stats['space_freed_mb']:.1f} MB")
        logger.info("=" * 50)


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(
        description="Clean up Jupyter notebook training outputs and temporary files"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Delete files older than N days (default: 7)"
    )
    parser.add_argument(
        "--type",
        choices=["all", "optimization", "ray", "models", "logs", "temp"],
        default="all",
        help="Type of cleanup to perform (default: all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    parser.add_argument(
        "--archive-configs",
        action="store_true",
        help="Archive best configurations before cleaning"
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    cleaner = OutputCleaner(args.project_root)
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be deleted")
    
    if args.archive_configs:
        cleaner.archive_best_configs(args.dry_run)
    
    # Perform cleanup based on type
    if args.type in ["all", "optimization"]:
        cleaner.clean_optimization_results(args.days, args.dry_run)
    
    if args.type in ["all", "ray"]:
        cleaner.clean_ray_results(args.days, args.dry_run)
    
    if args.type in ["all", "models"]:
        cleaner.clean_model_checkpoints(args.days, args.dry_run)
    
    if args.type in ["all", "logs"]:
        cleaner.clean_logs(args.days, args.dry_run)
    
    if args.type in ["all", "temp"]:
        cleaner.clean_temp_files(args.dry_run)
    
    cleaner.print_summary()


if __name__ == "__main__":
    main()
