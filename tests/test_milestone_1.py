import os
import polars as pl
from src.data.config import DataPipelineConfig
from src.data.unpack import extract_relevant_parquets
from src.data.loader import scan_table


def test_milestone_1():
    import logging

    logging.basicConfig(level=logging.INFO)

    import os
    zip_path = "data/raw/home-credit-credit-risk-model-stability.zip"
    if not os.path.exists(zip_path):
        import pytest
        pytest.skip(f"Skipping test, file {zip_path} not found.")

    cache_dir = ".cache_test"

    config = DataPipelineConfig(
        data_dir=zip_path,
        sample_ratio=0.001,
        cache_dir=cache_dir,
    )

    # 1. Unpack
    print("Testing extraction...")
    new_dir = extract_relevant_parquets(config.data_dir, config.cache_dir)
    assert os.path.isdir(new_dir), "Extraction did not return a valid directory"

    # 2. Scanning Base Table
    print("Testing scanning...")
    df_lazy = scan_table("train_base", new_dir)

    assert isinstance(df_lazy, pl.LazyFrame), "Did not return a LazyFrame"
    print("Successfully scanned train_base as LazyFrame.")
    print("Schema Snippet:")
    print(df_lazy.collect_schema())


if __name__ == "__main__":
    test_milestone_1()
