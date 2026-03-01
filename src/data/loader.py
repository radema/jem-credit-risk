import logging
import os

import polars as pl

logger = logging.getLogger(__name__)


def scan_table(table_base_name: str, cache_dir: str) -> pl.LazyFrame:
    """
    Lazily scans all parquet files matching a given table base name using Polars native globbing.
    Because parquet schemas align identically across chunks, we rely on glob resolution.

    Args:
        table_base_name: e.g., 'train_credit_bureau_a_1'
        cache_dir: Directory where the parquet files exist.

    Returns:
        A pl.LazyFrame encompassing all matching parts of the table.
    """
    glob_match = os.path.join(cache_dir, f"{table_base_name}*.parquet")
    logger.info(f"Scanning table with wildcard globbing: {glob_match}")
    return pl.scan_parquet(glob_match)
