import torch
import polars as pl
import numpy as np
import pytest
from src.model.jem.data_utils import ChunkedParquetDataset
from src.model.jem.scaler import TorchStandardScaler


def test_chunked_parquet_dataset_yields_all_rows(tmp_path):
    # Setup: Create 3 chunks of 100 rows each
    feature_cols = ["f0", "f1"]
    total_rows = 0
    for i in range(3):
        # f0: 0..99, 100..199, 200..299
        # f1: all 1.0
        # target: 90% class 0, 10% class 1
        df = pl.DataFrame(
            {
                "f0": np.arange(i * 100, (i + 1) * 100, dtype=np.float32),
                "f1": np.ones(100, dtype=np.float32),
                "target": np.array([0] * 90 + [1] * 10, dtype=np.int64),
                "WEEK_NUM": np.full(100, i, dtype=np.int64),
            }
        )
        df.write_parquet(tmp_path / f"train_chunk_{i:03d}.parquet")
        total_rows += len(df)

    chunk_paths = sorted(tmp_path.glob("train_chunk_*.parquet"))

    # Initialize dataset
    dataset = ChunkedParquetDataset(
        chunk_paths=chunk_paths,
        feature_cols=feature_cols,
        shuffle_buffer_size=50,
        shuffle_chunks=False,  # Disable for deterministic sequential check first
        seed=42,
    )

    # Collect all rows
    rows = list(dataset)
    assert len(rows) == total_rows

    # Check structure of first row
    x, y, week, weight = rows[0]
    assert torch.is_tensor(x)
    assert torch.is_tensor(y)
    assert torch.is_tensor(week)
    assert torch.is_tensor(weight)
    assert x.shape == (2,)
    assert y.dtype == torch.long
    assert week.dtype == torch.long
    assert weight.dtype == torch.float32


def test_chunked_parquet_dataset_scaling(tmp_path):
    feature_cols = ["f0"]
    df = pl.DataFrame(
        {
            "f0": [10.0, 20.0, 30.0],
            "target": [0, 1, 0],
            "WEEK_NUM": [1, 1, 1],
        }
    )
    path = tmp_path / "train_chunk_000.parquet"
    df.write_parquet(path)

    scaler = TorchStandardScaler(num_features=1)
    # Manual fit: mean=20, std=10 (unbiased var=100)
    scaler.mean.copy_(torch.tensor([20.0]))
    scaler.var.copy_(torch.tensor([100.0]))
    scaler.is_fitted.copy_(torch.tensor(True))

    dataset = ChunkedParquetDataset(
        chunk_paths=[path],
        feature_cols=feature_cols,
        scaler=scaler,
        shuffle_buffer_size=1,
    )

    rows = list(dataset)
    # Expected scaled values: (10-20)/10 = -1, (20-20)/10 = 0, (30-20)/10 = 1
    scaled_values = [r[0].item() for r in rows]
    assert pytest.approx(scaled_values) == [-1.0, 0.0, 1.0]


def test_chunked_parquet_dataset_shuffling(tmp_path):
    feature_cols = ["f0"]
    # 50 rows, 0..49
    df = pl.DataFrame(
        {
            "f0": np.arange(50, dtype=np.float32),
            "target": [0] * 50,
            "WEEK_NUM": [0] * 50,
        }
    )
    path = tmp_path / "train_chunk_000.parquet"
    df.write_parquet(path)

    # Fixed seed for reproducibility
    dataset = ChunkedParquetDataset(
        chunk_paths=[path], feature_cols=feature_cols, shuffle_buffer_size=50, seed=42
    )

    rows1 = [r[0].item() for r in dataset]
    rows2 = [r[0].item() for r in dataset]

    # Should be shuffled (not 0, 1, 2...)
    assert rows1 != list(range(50))
    # Should be different across calls if epoch changes or seed is handled?
    # Actually, IterableDataset __iter__ usually produces same sequence unless set_epoch is called
    assert rows1 == rows2

    dataset.set_epoch(1)
    rows3 = [r[0].item() for r in dataset]
    assert rows1 != rows3


def test_chunked_parquet_dataset_weights(tmp_path):
    feature_cols = ["f0"]
    # Chunk with 10 rows: 8 class 0, 2 class 1
    df = pl.DataFrame(
        {
            "f0": np.arange(10, dtype=np.float32),
            "target": [0] * 8 + [1] * 2,
            "WEEK_NUM": [0] * 10,
        }
    )
    path = tmp_path / "train_chunk_000.parquet"
    df.write_parquet(path)

    dataset = ChunkedParquetDataset(
        chunk_paths=[path],
        feature_cols=feature_cols,
        shuffle_buffer_size=1,
    )

    rows = list(dataset)
    # Weights should be 1/8 for class 0, 1/2 for class 1
    # Normalized? The spec says 1.0 / class_count
    for x, y, week, weight in rows:
        if y == 0:
            assert pytest.approx(weight.item()) == 1.0 / 8.0
        else:
            assert pytest.approx(weight.item()) == 1.0 / 2.0
