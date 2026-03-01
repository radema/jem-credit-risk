import logging
import os

import polars as pl
import logging
from pathlib import Path
import math

logger = logging.getLogger(__name__)


def evaluate_eda_stats(df: pl.DataFrame, output_path: str):
    """
    Computes .describe() on the finalized dataframe and writes fundamental
    statistics to a log file to highlight need for scaling/imputation.
    """
    logger.info("Computing EDA summary statistics...")
    describe_df = df.describe()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("=== Data Pipeline Phase 1: Feature Statistics ===\n")
        f.write(
            "The following statistics highlight distributions to inform Phase 3 scaling.\n\n"
        )

        # We can write the Polars output natively to strings
        f.write(str(describe_df))

    logger.info(f"EDA statistics exported to {output_path}")


def export_to_parquet(df: pl.DataFrame, output_path: str):
    """
    Writes the finalized, unscaled feature matrix to parquet.
    """
    logger.info(f"Exporting final dataset to {output_path} (Shape: {df.shape})")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write_parquet(output_path)
    logger.info("Export complete.")


def export_to_chunked_parquet(
    df: pl.DataFrame,
    output_dir: str,
    chunk_size: int = 200_000,
    prefix: str = "train_chunk",
) -> list[Path]:
    """
    Writes a DataFrame to disk as numbered Parquet partitions.

    Algorithm:
    1. Calculate total_chunks = ceil(len(df) / chunk_size)
    2. For i in range(total_chunks):
       - Slice df[i*chunk_size : (i+1)*chunk_size]
       - Write to {output_dir}/{prefix}_{i+1:03d}.parquet
    3. Return list of written paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_rows = df.height
    total_chunks = math.ceil(n_rows / chunk_size)

    logger.info(
        f"Exporting {n_rows} rows in {total_chunks} chunks to {output_dir} (prefix: {prefix})"
    )

    written_paths = []
    for i in range(total_chunks):
        start = i * chunk_size
        chunk_df = df.slice(start, chunk_size)
        file_path = Path(output_dir) / f"{prefix}_{i + 1:03d}.parquet"
        chunk_df.write_parquet(file_path)
        written_paths.append(file_path)

    logger.info(f"Successfully exported {len(written_paths)} chunks.")
    return written_paths
