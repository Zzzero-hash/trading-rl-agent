#!/usr/bin/env python3
"""
Cleanup script for ML experiment outputs.

This script helps manage disk space by cleaning up old experiment results,
temporary files, and notebook outputs while preserving important results.
"""

import os
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def get_directory_size(path):
    """Calculate the total size of a directory in bytes."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size


def format_size(size_bytes):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def print_storage_status():
    """Print current storage usage of output directories."""
    print("📊 Current Storage Usage:")
    print("-" * 40)
    
    directories = ["optimization_results", "ray_results", "data", "__pycache__"]
    total_size = 0
    
    for directory in directories:
        if Path(directory).exists():
            size = get_directory_size(directory)
            total_size += size
            print(f"{directory:20s}: {format_size(size)}")
        else:
            print(f"{directory:20s}: Not found")
    
    print("-" * 40)
    print(f"{'Total Output Size':20s}: {format_size(total_size)}")
    print()


def cleanup_ray_results(days_to_keep=7, dry_run=True):
    """Clean up old Ray Tune experiment directories."""
    ray_results_dir = Path("ray_results")
    if not ray_results_dir.exists():
        print("❌ ray_results directory not found")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    removed_count = 0
    removed_size = 0
    
    print(f"🧹 Cleaning Ray results older than {days_to_keep} days...")
    
    for experiment_dir in ray_results_dir.iterdir():
        if experiment_dir.is_dir():
            # Check modification time
            dir_time = datetime.fromtimestamp(experiment_dir.stat().st_mtime)
            if dir_time < cutoff_date:
                size = get_directory_size(experiment_dir)
                removed_size += size
                
                if dry_run:
                    print(f"  [DRY RUN] Would remove: {experiment_dir.name} "
                          f"({format_size(size)})")
                else:
                    print(f"  Removing: {experiment_dir.name} "
                          f"({format_size(size)})")
                    shutil.rmtree(experiment_dir)
                
                removed_count += 1
    
    action = "Would remove" if dry_run else "Removed"
    print(f"  {action} {removed_count} directories, {format_size(removed_size)}")
    print()


def cleanup_optimization_results(days_to_keep=7, dry_run=True):
    """Clean up old optimization result directories."""
    opt_results_dir = Path("optimization_results")
    if not opt_results_dir.exists():
        print("❌ optimization_results directory not found")
        return
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    removed_count = 0
    removed_size = 0
    
    print(f"🧹 Cleaning optimization results older than {days_to_keep} days...")
    
    for item in opt_results_dir.iterdir():
        if item.is_dir() and item.name.startswith("hparam_opt_"):
            # Check modification time
            dir_time = datetime.fromtimestamp(item.stat().st_mtime)
            if dir_time < cutoff_date:
                size = get_directory_size(item)
                removed_size += size
                
                if dry_run:
                    print(f"  [DRY RUN] Would remove: {item.name} "
                          f"({format_size(size)})")
                else:
                    print(f"  Removing: {item.name} ({format_size(size)})")
                    shutil.rmtree(item)
                
                removed_count += 1
    
    action = "Would remove" if dry_run else "Removed"
    print(f"  {action} {removed_count} directories, {format_size(removed_size)}")
    print()


def cleanup_python_cache(dry_run=True):
    """Clean up Python cache files."""
    removed_count = 0
    removed_size = 0
    
    print("🧹 Cleaning Python cache files...")
    
    # Find __pycache__ directories
    for cache_dir in Path(".").rglob("__pycache__"):
        size = get_directory_size(cache_dir)
        removed_size += size
        
        if dry_run:
            print(f"  [DRY RUN] Would remove: {cache_dir}")
        else:
            print(f"  Removing: {cache_dir}")
            shutil.rmtree(cache_dir)
        
        removed_count += 1
    
    # Find .pyc files
    for pyc_file in Path(".").rglob("*.pyc"):
        size = pyc_file.stat().st_size
        removed_size += size
        
        if dry_run:
            print(f"  [DRY RUN] Would remove: {pyc_file}")
        else:
            print(f"  Removing: {pyc_file}")
            pyc_file.unlink()
        
        removed_count += 1
    
    action = "Would remove" if dry_run else "Removed"
    print(f"  {action} {removed_count} cache files, {format_size(removed_size)}")
    print()


def clear_notebook_outputs(dry_run=True):
    """Clear outputs from Jupyter notebooks."""
    try:
        from nbformat import read, write, NO_CONVERT
    except ImportError:
        print("❌ jupyter/nbformat not available. "
              "Install with: pip install jupyter nbformat")
        return
    
    print("🧹 Clearing notebook outputs...")
    cleared_count = 0
    
    for notebook_path in Path(".").rglob("*.ipynb"):
        if ".ipynb_checkpoints" in str(notebook_path):
            continue
            
        try:
            with open(notebook_path, 'r') as f:
                notebook = read(f, as_version=NO_CONVERT)
            
            # Check if notebook has outputs
            has_outputs = any(
                cell.get('outputs') or cell.get('execution_count')
                for cell in notebook.cells
                if cell.cell_type == 'code'
            )
            
            if has_outputs:
                if dry_run:
                    print(f"  [DRY RUN] Would clear outputs: {notebook_path}")
                else:
                    # Clear outputs
                    for cell in notebook.cells:
                        if cell.cell_type == 'code':
                            cell.outputs = []
                            cell.execution_count = None
                    
                    with open(notebook_path, 'w') as f:
                        write(notebook, f)
                    
                    print(f"  Cleared outputs: {notebook_path}")
                
                cleared_count += 1
        
        except Exception as e:
            print(f"  ❌ Error processing {notebook_path}: {e}")
    
    action = "Would clear" if dry_run else "Cleared"
    print(f"  {action} outputs from {cleared_count} notebooks")
    print()


def archive_best_results():
    """Archive the best experiment results."""
    archive_dir = Path("results_archive") / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Archiving best results to: {archive_dir}")
    
    opt_results_dir = Path("optimization_results")
    if not opt_results_dir.exists():
        print("❌ optimization_results directory not found")
        return
    
    archived_count = 0
    
    # Archive best config files
    for config_file in opt_results_dir.glob("best_*_config_*.json"):
        dest = archive_dir / config_file.name
        shutil.copy2(config_file, dest)
        print(f"  Archived: {config_file.name}")
        archived_count += 1
    
    # Archive summary results
    for results_file in opt_results_dir.glob("*_hparam_results_*.json"):
        dest = archive_dir / results_file.name
        shutil.copy2(results_file, dest)
        print(f"  Archived: {results_file.name}")
        archived_count += 1
    
    print(f"  Archived {archived_count} files")
    print()


def main():
    parser = argparse.ArgumentParser(description="Clean up ML experiment outputs")
    parser.add_argument("--days", type=int, default=7, 
                       help="Days of results to keep (default: 7)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be deleted without actually deleting")
    parser.add_argument("--status-only", action="store_true",
                       help="Only show storage status")
    parser.add_argument("--archive", action="store_true",
                       help="Archive best results before cleanup")
    parser.add_argument("--clear-notebooks", action="store_true",
                       help="Clear notebook outputs")
    parser.add_argument("--all", action="store_true",
                       help="Run all cleanup operations")
    
    args = parser.parse_args()
    
    print("🧽 ML Experiment Cleanup Tool")
    print("=" * 50)
    
    # Always show status
    print_storage_status()
    
    if args.status_only:
        return
    
    # Archive best results if requested
    if args.archive or args.all:
        archive_best_results()
    
    # Clear notebook outputs if requested
    if args.clear_notebooks or args.all:
        clear_notebook_outputs(dry_run=args.dry_run)
    
    # Clean up experiment results
    if not args.clear_notebooks or args.all:
        cleanup_ray_results(days_to_keep=args.days, dry_run=args.dry_run)
        cleanup_optimization_results(days_to_keep=args.days, dry_run=args.dry_run)
        cleanup_python_cache(dry_run=args.dry_run)
    
    # Show final status
    if not args.dry_run:
        print("📊 Storage Usage After Cleanup:")
        print_storage_status()
    
    if args.dry_run:
        print("💡 Run without --dry-run to actually perform cleanup")


if __name__ == "__main__":
    main()
