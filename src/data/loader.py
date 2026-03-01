import os
import glob
import polars as pl
import logging

logger = logging.getLogger(__name__)


def scan_table(table_base_name: str, cache_dir: str) -> pl.LazyFrame:
    """
    Lazily scans all parquet files matching a given table base name.
    Handles potential data type mismatches between shards (e.g., Null vs String/Boolean).

    Args:
        table_base_name: e.g., 'train_credit_bureau_a_1'
        cache_dir: Directory where the parquet files exist.

    Returns:
        A pl.LazyFrame encompassing all matching parts of the table.
    """
    glob_match = os.path.join(cache_dir, "**", f"{table_base_name}*.parquet")
    logger.info(f"Scanning table with wildcard globbing: {glob_match}")

    files = glob.glob(glob_match, recursive=True)
    if not files:
        # Fallback to normal behavior if no files found (let scan_parquet fail clearly)
        return pl.scan_parquet(glob_match)

    if len(files) == 1:
        return pl.scan_parquet(files[0])

    # Multiple files: scan individually and normalize types to avoid shard mismatch
    # Home Credit dataset has specific naming conventions for types:
    # D: Date, A: Amount, M: masking/categorical, T: text/string, P: DPD/count, L: bool/other
    shards = []
    for f in sorted(files):
        lf = pl.scan_parquet(f)

        schema = lf.collect_schema()
        cast_cols = []
        for col, dtype in schema.items():
            if dtype == pl.Null:
                # If we have an all-null shard, cast it to the most likely type
                # based on competition feature suffixes to prevent mismatch with active shards.
                if col.endswith("D"):  # Date
                    cast_cols.append(pl.col(col).cast(pl.String))
                elif col.endswith(("A", "L", "P")):  # Numeric/Boolean archetypes
                    cast_cols.append(pl.col(col).cast(pl.Float64))
                elif col.endswith(("M", "T")):  # Categorical archetypes
                    cast_cols.append(pl.col(col).cast(pl.String))

        if cast_cols:
            lf = lf.with_columns(cast_cols)
        shards.append(lf)

    # Diagonal/Relaxed concatenation to handle schema drift across shards
    return pl.concat(shards, how="vertical_relaxed")
