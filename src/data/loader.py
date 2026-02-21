import os
import polars as pl
import logging

logger = logging.getLogger(__name__)


def scan_table(table_base_name: str, cache_dir: str) -> pl.LazyFrame:
    """
    Lazily scans all parquet files matching a given table base name.
    Polars natively supports reading multiple files matching a glob pattern.

    Args:
        table_base_name: e.g., 'train_bureau_a_1'
        cache_dir: Directory where the parquet files exist.

    Returns:
        A pl.LazyFrame encompassing all parts of the table.
    """
    # Create the glob pattern, e.g., .cache/train_bureau_a_1*.parquet
    # We use table_base_name to capture partitioned chunks like _0, _1, etc. if they exist
    # Sometimes kaggle datasets partition them without underscores, so an asterisk is safest.

    # If the file exists directly without partitions:
    exact_match = os.path.join(cache_dir, f"{table_base_name}.parquet")
    glob_match = os.path.join(cache_dir, f"*{table_base_name}*.parquet")

    if os.path.exists(exact_match):
        logger.info(f"Scanning exact match: {exact_match}")
        return pl.scan_parquet(exact_match)

    logger.info(f"Scanning glob match: {glob_match}")
    # Using scan_parquet with a wild string
    return pl.scan_parquet(glob_match)
