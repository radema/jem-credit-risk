import os

import polars as pl

from src.model.jem.train import get_dataloaders

parquet_path = "data/processed/train_features_unscaled.parquet"


def test_dataloaders():
    if not os.path.exists(parquet_path):
        import pytest

        pytest.skip(f"Skipping dataloader test, file {parquet_path} not found.")

    df = pl.read_parquet(parquet_path)
    df_train = df.head(100)
    df_val = df.tail(100)
    get_dataloaders(df_train, df_val)
