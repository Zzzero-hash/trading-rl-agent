"""
CSV utilities for efficient chunked saving and loading operations.

This module provides optimized CSV operations for large datasets,
including chunked saving to reduce memory usage and improve performance.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from trade_agent.core.logging import get_logger

logger = get_logger(__name__)


def save_csv_chunked(
    df: pd.DataFrame,
    filepath: str | Path,
    chunk_size: int = 10000,
    index: bool = False,
    show_progress: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save a DataFrame to CSV using chunked writing for memory efficiency.

    This function is particularly useful for large DataFrames that might
    not fit in memory when writing to CSV all at once.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Union[str, Path]
        Output file path
    chunk_size : int, default=10000
        Number of rows to write per chunk
    index : bool, default=False
        Whether to write DataFrame index
    show_progress : bool, default=True
        Whether to show progress bar
    **kwargs : Any
        Additional arguments passed to pandas to_csv()

    Returns
    -------
    None
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)

    if n_rows <= chunk_size:
        # Small enough, save all at once
        logger.debug(f"Saving {n_rows} rows to {filepath} (single chunk)")
        df.to_csv(filepath, index=index, **kwargs)
        return

    # Save in chunks
    logger.info(f"Saving {n_rows} rows to {filepath} in chunks of {chunk_size}")

    # Write header first
    df.head(0).to_csv(filepath, index=index, **kwargs)

    # Process chunks
    iterator = range(0, n_rows, chunk_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=(n_rows + chunk_size - 1) // chunk_size,
            desc="Saving CSV",
            unit="chunk",
        )

    for start in iterator:
        end = min(start + chunk_size, n_rows)
        chunk = df.iloc[start:end]

        # Append chunk without header
        chunk.to_csv(filepath, mode="a", header=False, index=index, **kwargs)

    logger.info(f"Successfully saved {n_rows} rows to {filepath}")


def save_csv_chunked_parallel(
    df: pd.DataFrame,
    filepath: str | Path,
    chunk_size: int = 10000,
    max_workers: int | None = None,
    index: bool = False,
    show_progress: bool = True,
    **kwargs: Any,
) -> None:
    """
    Save a DataFrame to CSV using parallel chunked writing for maximum efficiency.

    This function processes chunks in parallel and then combines them,
    which can be faster for very large datasets on multi-core systems.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : Union[str, Path]
        Output file path
    chunk_size : int, default=10000
        Number of rows per chunk
    max_workers : Optional[int], default=None
        Maximum number of worker processes
    index : bool, default=False
        Whether to write DataFrame index
    show_progress : bool, default=True
        Whether to show progress bar
    **kwargs : Any
        Additional arguments passed to pandas to_csv()

    Returns
    -------
    None
    """
    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
    except ImportError:
        logger.warning("multiprocessing not available, falling back to sequential chunked saving")
        return save_csv_chunked(df, filepath, chunk_size, index, show_progress, **kwargs)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(df)

    if n_rows <= chunk_size:
        # Small enough, save all at once
        logger.debug(f"Saving {n_rows} rows to {filepath} (single chunk)")
        df.to_csv(filepath, index=index, **kwargs)
        return None

    # Determine number of workers
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead

    logger.info(f"Saving {n_rows} rows to {filepath} using {max_workers} workers")

    # Create temporary directory for chunks
    temp_dir = filepath.parent / f"{filepath.stem}_chunks"
    temp_dir.mkdir(exist_ok=True)

    def save_chunk(chunk_data: tuple[int, pd.DataFrame]) -> str:
        """Save a single chunk to temporary file."""
        chunk_idx, chunk_df = chunk_data
        temp_file = temp_dir / f"chunk_{chunk_idx:06d}.csv"
        chunk_df.to_csv(temp_file, index=index, **kwargs)
        return str(temp_file)

    # Prepare chunks
    chunks = []
    for i, start in enumerate(range(0, n_rows, chunk_size)):
        end = min(start + chunk_size, n_rows)
        chunk = df.iloc[start:end]
        chunks.append((i, chunk))

    # Process chunks in parallel
    temp_files = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        if show_progress:
            futures = {executor.submit(save_chunk, chunk): chunk[0] for chunk in chunks}

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing chunks",
                unit="chunk",
            ):
                temp_file = future.result()
                temp_files.append(temp_file)
        else:
            temp_files = list(executor.map(save_chunk, chunks))

    # Combine chunks
    logger.info("Combining chunks...")
    with open(filepath, "w") as outfile:
        # Write header from first chunk
        if temp_files:
            with open(temp_files[0]) as first_chunk:
                header = first_chunk.readline()
                outfile.write(header)

            # Write data from all chunks
            for temp_file in temp_files:
                with open(temp_file) as chunk_file:
                    next(chunk_file)  # Skip header
                    outfile.write(chunk_file.read())

    # Clean up temporary files
    for temp_file in temp_files:
        Path(temp_file).unlink()
    temp_dir.rmdir()

    logger.info(f"Successfully saved {n_rows} rows to {filepath}")
    return None


def load_csv_chunked(
    filepath: str | Path,
    chunk_size: int = 10000,
    show_progress: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a CSV file using chunked reading for memory efficiency.

    This function is useful for very large CSV files that might not
    fit in memory when loaded all at once.

    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path
    chunk_size : int, default=10000
        Number of rows to read per chunk
    show_progress : bool, default=True
        Whether to show progress bar
    **kwargs : Any
        Additional arguments passed to pandas read_csv()

    Returns
    -------
    pd.DataFrame
        Combined DataFrame from all chunks
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Get file size for progress estimation (much faster than counting rows)
    file_size_mb = filepath.stat().st_size / (1024 * 1024)

    # Estimate total rows based on file size (rough approximation)
    # Assume average row size of ~1KB for progress estimation
    estimated_chunks = max(1, int(file_size_mb * 1024 / (chunk_size * 0.001)))

    logger.info(f"Loading CSV from {filepath} ({file_size_mb:.1f}MB) in chunks of {chunk_size}")

    # Read chunks
    chunks = []
    chunk_iterator = pd.read_csv(filepath, chunksize=chunk_size, **kwargs)

    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, total=estimated_chunks, desc="Loading CSV", unit="chunk")

    for chunk in chunk_iterator:
        chunks.append(chunk.copy())

    # Combine chunks
    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        logger.info(f"Successfully loaded {len(result)} rows from {filepath}")
        return result
    logger.warning(f"No data found in {filepath}")
    return pd.DataFrame()


def get_csv_info(filepath: str | Path) -> dict[str, Any]:
    """
    Get information about a CSV file without loading it entirely.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to CSV file

    Returns
    -------
    dict[str, Any]
        Dictionary containing file information
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Get file size
    file_size = filepath.stat().st_size

    # Read first few rows to get column info
    sample_df = pd.read_csv(filepath, nrows=1000)

    # Estimate total rows
    sample_size = len(sample_df)
    estimated_rows = int(file_size / filepath.stat().st_size * sample_size) if sample_size > 0 else 0

    return {
        "filepath": str(filepath),
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
        "columns": list(sample_df.columns),
        "dtypes": sample_df.dtypes.to_dict(),
        "estimated_rows": estimated_rows,
        "sample_rows": sample_size,
    }


def process_csv_in_stream(
    filepath: str | Path,
    processor_func: Callable[[pd.DataFrame], pd.DataFrame],
    chunk_size: int = 5000,
    show_progress: bool = True,
    output_file: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    Process a large CSV file in streaming fashion without loading everything into memory.

    This function is ideal for very large CSV files that need to be processed
    but don't fit in memory when loaded all at once.

    Parameters
    ----------
    filepath : Union[str, Path]
        Input file path
    processor_func : Callable[[pd.DataFrame], pd.DataFrame]
        Function to process each chunk of data
    chunk_size : int, default=5000
        Number of rows to process per chunk
    show_progress : bool, default=True
        Whether to show progress bar
    output_file : Optional[Union[str, Path]], default=None
        If provided, save processed chunks to this file instead of returning DataFrame
    **kwargs : Any
        Additional arguments passed to pandas read_csv()

    Returns
    -------
    Optional[pd.DataFrame]
        Processed DataFrame (only if output_file is None, otherwise None)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Get file size for progress estimation
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    estimated_chunks = max(1, int(file_size_mb * 1024 / (chunk_size * 0.001)))

    logger.info(f"Processing CSV from {filepath} ({file_size_mb:.1f}MB) in streaming mode")

    # Process chunks in streaming fashion
    chunk_iterator = pd.read_csv(filepath, chunksize=chunk_size, **kwargs)

    if show_progress:
        chunk_iterator = tqdm(chunk_iterator, total=estimated_chunks, desc="Processing CSV", unit="chunk")

    if output_file:
        # Stream to file - much more memory efficient
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        first_chunk = True
        for chunk in chunk_iterator:
            # Process the chunk
            processed_chunk = processor_func(chunk)

            # Write to file
            if first_chunk:
                processed_chunk.to_csv(output_file, index=False, mode="w")
                first_chunk = False
            else:
                processed_chunk.to_csv(output_file, index=False, mode="a", header=False)

        logger.info(f"Successfully processed and saved to {output_file}")
        return None
    # Keep in memory (only for small files)
    processed_chunks = []
    for chunk in chunk_iterator:
        # Process the chunk
        processed_chunk = processor_func(chunk)
        processed_chunks.append(processed_chunk)

        # Keep only the last few chunks in memory to avoid memory buildup
        if len(processed_chunks) > 3:
            # Combine and keep only the result
            combined = pd.concat(processed_chunks[:-1], ignore_index=True)
            processed_chunks = [combined, processed_chunks[-1]]

    # Combine all processed chunks
    if processed_chunks:
        result = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"Successfully processed {len(result)} rows from {filepath}")
        return result
    logger.warning(f"No data found in {filepath}")
    return pd.DataFrame()


def create_standardized_dataset_streaming(
    filepath: str | Path,
    standardizer_path: str = "data/processed/data_standardizer.pkl",
    output_file: str = "data/processed/standardized_dataset.csv",
    chunk_size: int = 5000,
    show_progress: bool = True,
) -> tuple[pd.DataFrame | None, Any]:
    """
    Create a standardized dataset from a large CSV file using streaming processing.

    This function processes the data in chunks to avoid memory issues with large files.

    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the CSV file
    standardizer_path : str
        Path to save the standardizer
    output_file : str
        Path to save the standardized dataset
    chunk_size : int, default=5000
        Number of rows to process per chunk
    show_progress : bool, default=True
        Whether to show progress bar

    Returns
    -------
    tuple[Optional[pd.DataFrame], Any]
        Standardized DataFrame (None if saved to file) and fitted standardizer
    """
    from .data_standardizer import (
        DataStandardizer,
        FeatureConfig,
    )

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Creating standardized dataset from {filepath} using streaming processing")

    # Initialize standardizer
    standardizer = DataStandardizer(feature_config=FeatureConfig())

    # First pass: fit the standardizer on a sample
    logger.info("Fitting standardizer on sample data...")
    sample_df = pd.read_csv(filepath, nrows=chunk_size)
    standardizer.fit(sample_df)

    # Second pass: transform all data in streaming fashion and save to file
    def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        """Transform a chunk of data using the fitted standardizer."""
        return standardizer.transform(chunk)

    logger.info("Transforming data in streaming fashion and saving to file...")
    process_csv_in_stream(
        filepath,
        transform_chunk,
        chunk_size=chunk_size,
        show_progress=show_progress,
        output_file=output_file,
    )

    # Save the standardizer
    if standardizer_path:
        standardizer.save(standardizer_path)
        logger.info(f"Standardizer saved to {standardizer_path}")

    logger.info(f"Streaming standardization complete. Output saved to: {output_file}")
    return None, standardizer
