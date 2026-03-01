"""Tests for ChunkedLatentDataset + JEM Integration (TASK-6).

Verifies:
- ChunkedLatentDataset yields all rows with (z, y, week, weight) structure.
- Shuffle buffer shuffles across epochs.
- Per-chunk class weighting is correct.
- A 2-epoch JEM training loop using SGLD buffer runs end-to-end.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.model.jem.config import JEMConfig
from src.model.jem.data_utils import ChunkedLatentDataset
from src.model.jem.loss import JEMLoss
from src.model.jem.model import TabularJEM
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.train import train_jem_epoch


def _create_latent_chunks(tmp_path: Path, n_chunks=3, rows_per_chunk=50, latent_dim=8):
    """Create synthetic latent .pt chunks."""
    chunk_dir = tmp_path / "latent_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for c in range(n_chunks):
        z = torch.randn(rows_per_chunk, latent_dim)
        # 80% class 0, 20% class 1
        y = torch.tensor([0] * (rows_per_chunk - 10) + [1] * 10, dtype=torch.long)
        weeks = torch.full((rows_per_chunk,), c, dtype=torch.long)
        path = chunk_dir / f"latent_chunk_{c + 1:03d}.pt"
        torch.save({"z": z, "y": y, "weeks": weeks}, path)
        paths.append(path)
    return paths


class TestChunkedLatentDataset:
    """Tests for ChunkedLatentDataset."""

    def test_yields_all_rows(self, tmp_path):
        n_chunks, rows_per_chunk = 3, 50
        paths = _create_latent_chunks(
            tmp_path, n_chunks=n_chunks, rows_per_chunk=rows_per_chunk
        )

        dataset = ChunkedLatentDataset(
            chunk_paths=paths,
            shuffle_buffer_size=20,
            shuffle_chunks=False,
            seed=42,
        )
        rows = list(dataset)
        assert len(rows) == n_chunks * rows_per_chunk

        # Check tuple structure
        z, y, week, weight = rows[0]
        assert z.shape == (8,)
        assert y.dtype == torch.long
        assert week.dtype == torch.long
        assert weight.dtype == torch.float32

    def test_class_weights(self, tmp_path):
        paths = _create_latent_chunks(tmp_path, n_chunks=1, rows_per_chunk=50)

        dataset = ChunkedLatentDataset(chunk_paths=paths, shuffle_buffer_size=1, seed=0)
        rows = list(dataset)
        # Normalized weights: n / (n_classes * class_count)
        # n=50, n_classes=2, class_0=40, class_1=10
        for _z, y, _week, weight in rows:
            if y == 0:
                assert pytest.approx(weight.item()) == 50.0 / (2 * 40)  # = 0.625
            else:
                assert pytest.approx(weight.item()) == 50.0 / (2 * 10)  # = 2.5

    def test_shuffle_across_epochs(self, tmp_path):
        paths = _create_latent_chunks(
            tmp_path, n_chunks=1, rows_per_chunk=30, latent_dim=4
        )

        dataset = ChunkedLatentDataset(
            chunk_paths=paths, shuffle_buffer_size=30, seed=42
        )
        dataset.set_epoch(0)
        order_0 = [r[0][0].item() for r in dataset]

        dataset.set_epoch(1)
        order_1 = [r[0][0].item() for r in dataset]

        # Different epoch → different order
        assert order_0 != order_1

    def test_set_epoch_returns_none(self, tmp_path):
        paths = _create_latent_chunks(tmp_path)
        dataset = ChunkedLatentDataset(chunk_paths=paths)
        assert dataset.set_epoch(5) is None


class TestJEMTrainingLoopWithChunkedLatent:
    """Integration test: 2-epoch JEM training on synthetic chunked latent data."""

    def test_jem_training_runs(self, tmp_path):
        latent_dim = 8
        paths = _create_latent_chunks(
            tmp_path, n_chunks=3, rows_per_chunk=50, latent_dim=latent_dim
        )

        config = JEMConfig(
            hidden_dims=[32, 32],
            sgld_steps=3,
            sgld_step_size=0.01,
            sgld_sigma=0.01,
            l2_energy_weight=1e-4,
            buffer_size=100,
        )

        dataset = ChunkedLatentDataset(
            chunk_paths=paths, shuffle_buffer_size=20, seed=42
        )
        loader = DataLoader(dataset, batch_size=16, num_workers=0)

        model = TabularJEM(input_dim=latent_dim, num_classes=2, config=config)
        buffer = SGLDReplayBuffer(buffer_size=100, feature_dim=latent_dim)
        sampler = SGLDSampler(config=config)
        criterion = JEMLoss(config=config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        losses = []
        for epoch in range(2):
            dataset.set_epoch(epoch)
            logs = train_jem_epoch(
                model, loader, optimizer, criterion, buffer, sampler, device
            )
            losses.append(logs["loss"])

        # Training must run without error and produce finite loss
        assert len(losses) == 2
        assert all(np.isfinite(loss_val) for loss_val in losses)

        # SGLD buffer should have been populated
        assert buffer.buffer.shape[0] > 0
