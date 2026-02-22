import polars as pl
import torch
from src.model.jem.train import get_dataloaders
df = pl.read_parquet("data/processed/train_features_unscaled.parquet")
df_train = df.head(100)
df_val = df.tail(100)
get_dataloaders(df_train, df_val)
print("Dataloaders generated successfully!")

