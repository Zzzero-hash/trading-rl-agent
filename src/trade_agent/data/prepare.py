"""
Data preparation module for trading RL agent.

Provides functions for processing and standardizing downloaded data
for training and inference.
"""

from pathlib import Path

import pandas as pd

from .data_standardizer import create_standardized_dataset
from .pipeline import DataPipeline


def prepare_data(
    input_path: Path | None = None,
    output_dir: Path = Path("data/processed"),
    config_path: Path | None = None,
    method: str = "robust",
    save_standardizer: bool = True,
) -> None:
    """
    Process and standardize downloaded data in one command.

    This function combines data processing and standardization into a single step,
    making it easier to prepare data for training and inference.

    Args:
        input_path: Path to input data (file or directory)
        output_dir: Directory to save processed data
        config_path: Path to configuration file
        force_rebuild: Whether to force rebuild of processed data
        parallel: Whether to use parallel processing
        method: Standardization method to use
        save_standardizer: Whether to save the standardizer for later use
    """
    # Set default input path if not provided
    if input_path is None:
        input_path = Path("data/raw")

    # Set default config file if not provided
    if config_path is None:
        config_path = Path("config.yaml")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Process the data
    DataPipeline()

    # Load and process data
    if input_path.is_file():
        # Single file
        df = pd.read_csv(input_path)
    elif input_path.is_dir():
        # Directory with multiple files
        csv_files = list(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_path}")

        dfs = []
        for csv_file in csv_files:
            try:
                # Check if file is empty or has no columns
                if csv_file.stat().st_size == 0:
                    print(f"Warning: Skipping empty file: {csv_file}")
                    continue

                df = pd.read_csv(csv_file)

                # Check if DataFrame is empty or has no columns
                if df.empty or len(df.columns) == 0:
                    print(f"Warning: Skipping file with no data: {csv_file}")
                    continue

                df["source_file"] = csv_file.name
                dfs.append(df)
            except Exception as e:
                print(f"Warning: Error reading {csv_file}: {e}")
                continue

        if not dfs:
            raise ValueError(f"No valid CSV files found in {input_path}")

        df = pd.concat(dfs, ignore_index=True)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # Step 2: Standardize the data
    # Create standardized dataset
    standardized_df, standardizer = create_standardized_dataset(
        df=df,
        save_path=str(output_dir / "data_standardizer.pkl") if save_standardizer else None
    )

    # Step 3: Save processed data
    # Save standardized data
    output_file = output_dir / "standardized_data.csv"
    standardized_df.to_csv(output_file, index=False)

    # Save feature summary
    feature_summary = {
        "total_features": standardizer.get_feature_count(),
        "feature_names": standardizer.get_feature_names(),
        "data_shape": standardized_df.shape,
        "standardization_method": method,
        "missing_value_strategies": standardizer.missing_value_strategies
    }

    import json
    summary_file = output_dir / "feature_summary.json"
    with open(summary_file, "w") as f:
        json.dump(feature_summary, f, indent=2)

    # Step 4: Cleanup raw data for organization
    raw_data_path = Path("data/raw")
    if raw_data_path.exists() and raw_data_path.is_dir():
        try:
            # Count files before deletion for reporting
            files_to_delete = list(raw_data_path.glob("*"))
            len([f for f in files_to_delete if f.is_file()])
            len([f for f in files_to_delete if f.is_dir()])

            # Remove all contents of data/raw
            import shutil
            shutil.rmtree(raw_data_path)
            raw_data_path.mkdir(parents=True, exist_ok=True)  # Recreate empty directory
        except Exception:
            # Silently continue if cleanup fails
            pass
