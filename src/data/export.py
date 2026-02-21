import os
import polars as pl
import logging

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
