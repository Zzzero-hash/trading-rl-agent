#!/usr/bin/env python3
"""
Test script to demonstrate chunked CSV saving functionality.

This script creates a large dataset and compares the performance
of regular pandas to_csv() vs chunked CSV saving.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.trading_rl_agent.data.csv_utils import save_csv_chunked, save_csv_chunked_parallel


def create_large_dataset(rows: int = 100000, cols: int = 50) -> pd.DataFrame:
    """Create a large synthetic dataset for testing."""
    print(f"Creating dataset with {rows:,} rows and {cols} columns...")

    # Create synthetic data
    data = {}
    for i in range(cols):
        if i < 5:  # Price-like columns
            data[f"price_{i}"] = np.random.uniform(10, 1000, rows)
        elif i < 15:  # Technical indicators
            data[f"tech_{i}"] = np.random.normal(0, 1, rows)
        elif i < 25:  # Volume-like columns
            data[f"volume_{i}"] = np.random.uniform(1000, 1000000, rows)
        else:  # Other features
            data[f"feature_{i}"] = np.random.normal(0, 0.1, rows)

    df = pd.DataFrame(data)
    print(f"Dataset created: {df.shape}, Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df


def test_regular_csv_saving(df: pd.DataFrame, filepath: str) -> float:
    """Test regular pandas to_csv() performance."""
    print("\nTesting regular pandas to_csv()...")
    start_time = time.time()

    df.to_csv(filepath, index=False)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Regular CSV saving completed in {duration:.2f} seconds")
    return duration


def test_chunked_csv_saving(df: pd.DataFrame, filepath: str, chunk_size: int) -> float:
    """Test chunked CSV saving performance."""
    print(f"\nTesting chunked CSV saving (chunk_size={chunk_size:,})...")
    start_time = time.time()

    save_csv_chunked(df, filepath, chunk_size=chunk_size, show_progress=True)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Chunked CSV saving completed in {duration:.2f} seconds")
    return duration


def test_parallel_chunked_csv_saving(df: pd.DataFrame, filepath: str, chunk_size: int) -> float:
    """Test parallel chunked CSV saving performance."""
    print(f"\nTesting parallel chunked CSV saving (chunk_size={chunk_size:,})...")
    start_time = time.time()

    save_csv_chunked_parallel(df, filepath, chunk_size=chunk_size, show_progress=True)

    end_time = time.time()
    duration = end_time - start_time
    print(f"Parallel chunked CSV saving completed in {duration:.2f} seconds")
    return duration


def compare_file_sizes(filepaths: list[str]) -> None:
    """Compare file sizes of different CSV files."""
    print("\n📊 File Size Comparison:")
    for filepath in filepaths:
        if Path(filepath).exists():
            size_mb = Path(filepath).stat().st_size / (1024 * 1024)
            print(f"   {Path(filepath).name}: {size_mb:.2f} MB")


def main() -> None:
    """Main test function."""
    print("🚀 Chunked CSV Saving Performance Test")
    print("=" * 60)

    # Create output directory
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    # Test with different dataset sizes
    test_cases = [
        (10000, 20, "small"),
        (50000, 30, "medium"),
        (100000, 50, "large"),
    ]

    results = {}

    for rows, cols, size_name in test_cases:
        print(f"\n{'=' * 20} Testing {size_name.upper()} Dataset {'=' * 20}")

        # Create dataset
        df = create_large_dataset(rows, cols)

        # Test different saving methods
        filepaths = []

        # Regular CSV saving
        regular_path = output_dir / f"{size_name}_regular.csv"
        regular_time = test_regular_csv_saving(df, str(regular_path))
        filepaths.append(str(regular_path))

        # Chunked CSV saving
        chunked_path = output_dir / f"{size_name}_chunked.csv"
        chunked_time = test_chunked_csv_saving(df, str(chunked_path), chunk_size=10000)
        filepaths.append(str(chunked_path))

        # Parallel chunked CSV saving (for larger datasets)
        if rows >= 50000:
            parallel_path = output_dir / f"{size_name}_parallel.csv"
            parallel_time = test_parallel_chunked_csv_saving(df, str(parallel_path), chunk_size=10000)
            filepaths.append(str(parallel_path))
        else:
            parallel_time = None

        # Compare file sizes
        compare_file_sizes(filepaths)

        # Store results
        results[size_name] = {
            "dataset_size": f"{rows:,} x {cols}",
            "regular_time": regular_time,
            "chunked_time": chunked_time,
            "parallel_time": parallel_time,
            "chunked_speedup": regular_time / chunked_time if chunked_time > 0 else 0,
            "parallel_speedup": regular_time / parallel_time if parallel_time and parallel_time > 0 else 0,
        }

    # Print summary
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)

    for size_name, result in results.items():
        print(f"\n{size_name.upper()} Dataset ({result['dataset_size']}):")
        print(f"   Regular CSV:     {result['regular_time']:.2f}s")
        print(f"   Chunked CSV:     {result['chunked_time']:.2f}s")
        if result["parallel_time"]:
            print(f"   Parallel CSV:    {result['parallel_time']:.2f}s")

        print(f"   Chunked speedup: {result['chunked_speedup']:.2f}x")
        if result["parallel_speedup"]:
            print(f"   Parallel speedup: {result['parallel_speedup']:.2f}x")

    print("\n✅ Test completed! Check 'test_outputs/' directory for generated files.")
    print("💡 Use chunked CSV saving for better memory efficiency and performance!")


if __name__ == "__main__":
    main()
