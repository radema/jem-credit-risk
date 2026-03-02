"""Tests for the chunked AE training integration (TASK-4).

Verifies that the chunked training path in train_autoencoder.py correctly:
- Discovers chunk files
- Fits a scaler via streaming_fit
- Trains an autoencoder using ChunkedParquetDataset
- Saves encoder and scaler artifacts
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from src.model.autoencoder.model import TabularAutoencoder
from src.model.autoencoder.train_autoencoder import (
    _discover_chunks,
    weighted_mse_loss,
)
from src.model.jem.data_utils import ChunkedParquetDataset
from src.model.jem.scaler import TorchStandardScaler


def _create_synthetic_chunks(
    tmp_path: Path, n_chunks=3, rows_per_chunk=100, n_features=4
):
    """Helper: create synthetic Parquet chunks for testing."""
    feature_cols = [f"f{i}" for i in range(n_features)]
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for c in range(n_chunks):
        data = {
            col: np.random.randn(rows_per_chunk).astype(np.float32)
            for col in feature_cols
        }
        data["target"] = np.array(
            [0] * (rows_per_chunk - 10) + [1] * 10, dtype=np.int64
        )
        data["WEEK_NUM"] = np.full(rows_per_chunk, c, dtype=np.int64)
        df = pl.DataFrame(data)
        df.write_parquet(chunk_dir / f"train_chunk_{c + 1:03d}.parquet")

    return chunk_dir, feature_cols


def test_discover_chunks(tmp_path):
    chunk_dir, _ = _create_synthetic_chunks(tmp_path, n_chunks=3)
    paths = _discover_chunks(str(chunk_dir))
    assert len(paths) == 3
    # Verify sorted order
    assert "001" in str(paths[0])
    assert "003" in str(paths[2])


def test_discover_chunks_empty(tmp_path):
    paths = _discover_chunks(str(tmp_path / "nonexistent"))
    assert len(paths) == 0


def test_weighted_mse_loss():
    x = torch.ones(4, 3)
    x_recon = torch.zeros(4, 3)
    weights = torch.tensor([1.0, 2.0, 1.0, 0.5])

    loss = weighted_mse_loss(x, x_recon, weights)
    # MSE per sample = 1.0 (each feature diff is 1, squared is 1, mean of 3 features = 1)
    # Weighted: (1*1 + 2*1 + 1*1 + 0.5*1) / 4 = 4.5 / 4 = 1.125
    assert pytest.approx(loss.item(), rel=1e-5) == 1.125


def test_chunked_ae_training_loop(tmp_path):
    """Integration test: 2 epochs of AE training on synthetic chunked data."""
    chunk_dir, feature_cols = _create_synthetic_chunks(
        tmp_path, n_chunks=3, rows_per_chunk=50, n_features=4
    )
    chunk_paths = sorted(chunk_dir.glob("train_chunk_*.parquet"))

    # Fit scaler
    scaler = TorchStandardScaler(num_features=len(feature_cols))
    scaler.streaming_fit(chunk_paths, feature_cols)
    assert scaler.is_fitted

    # Create dataset + loader
    dataset = ChunkedParquetDataset(
        chunk_paths=chunk_paths,
        feature_cols=feature_cols,
        scaler=scaler,
        shuffle_buffer_size=20,
        seed=42,
    )
    loader = DataLoader(dataset, batch_size=16, num_workers=0)

    # Model
    ae = TabularAutoencoder(input_dim=len(feature_cols), latent_dim=4)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)

    # Train 2 epochs
    losses = []
    for epoch in range(2):
        dataset.set_epoch(epoch)
        ae.train()
        epoch_loss = 0.0
        n_batches = 0
        for x, _, _, sample_weight in loader:
            optimizer.zero_grad()
            x_recon, _ = ae(x)
            loss = weighted_mse_loss(x, x_recon, sample_weight)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    # Verify training ran and loss decreased (or at least didn't explode)
    assert len(losses) == 2
    assert all(loss_val > 0 for loss_val in losses)
    # Loss should decrease or remain reasonable
    assert losses[-1] < losses[0] * 2  # not exploding

    # Save and verify encoder
    encoder_path = tmp_path / "encoder.pt"
    torch.save(ae.encoder.state_dict(), encoder_path)
    assert encoder_path.exists()

    # Save and verify scaler
    scaler_path = tmp_path / "scaler.pt"
    scaler.save(str(scaler_path))
    assert scaler_path.exists()

    # Reload and verify
    loaded_scaler = TorchStandardScaler.load(str(scaler_path))
    assert loaded_scaler.is_fitted
    assert torch.allclose(scaler.mean, loaded_scaler.mean)
