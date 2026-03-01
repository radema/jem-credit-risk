"""
Tests for TorchStandardScaler.streaming_fit() — Welford's online algorithm.

TASK-2: Streaming Scaler Fit
AC2: streaming_fit() computes global mean/variance using Welford's online
     algorithm across all chunks, requiring O(num_features) memory.

The test creates 3 synthetic Parquet chunks, fits a scaler via full tensor
fit() and another via streaming_fit(), then asserts allclose(rtol=1e-5).
"""

import torch
import polars as pl
import pytest
from pathlib import Path

from src.model.jem.scaler import TorchStandardScaler


class TestStreamingFit:
    """Test suite for the streaming_fit method on TorchStandardScaler."""

    NUM_FEATURES = 10
    CHUNK_SIZE = 200
    NUM_CHUNKS = 3
    FEATURE_COLS = [f"f{i}" for i in range(NUM_FEATURES)]

    @pytest.fixture
    def synthetic_chunks(self, tmp_path: Path) -> tuple[torch.Tensor, list[Path]]:
        """
        Creates 3 synthetic Parquet chunks of 200 rows each (600 total).
        Returns (full_tensor, list_of_chunk_paths).
        """
        torch.manual_seed(42)
        full_data = torch.randn(self.CHUNK_SIZE * self.NUM_CHUNKS, self.NUM_FEATURES)

        chunk_paths = []
        for i in range(self.NUM_CHUNKS):
            chunk = full_data[i * self.CHUNK_SIZE : (i + 1) * self.CHUNK_SIZE]
            df = pl.DataFrame(
                {col: chunk[:, j].numpy() for j, col in enumerate(self.FEATURE_COLS)}
            )
            path = tmp_path / f"chunk_{i:03d}.parquet"
            df.write_parquet(path)
            chunk_paths.append(path)

        return full_data, sorted(chunk_paths)

    def test_streaming_fit_matches_full_fit(
        self, synthetic_chunks: tuple[torch.Tensor, list[Path]]
    ):
        """Core test: streaming_fit must match single-pass fit within rtol=1e-5."""
        full_data, chunk_paths = synthetic_chunks

        # Full fit (reference)
        scaler_full = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_full.fit(full_data)

        # Streaming fit
        scaler_stream = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_stream.streaming_fit(chunk_paths, self.FEATURE_COLS)

        assert scaler_stream.is_fitted, "Scaler should be marked as fitted"
        assert torch.allclose(scaler_full.mean, scaler_stream.mean, rtol=1e-5), (
            f"Mean mismatch:\n  full={scaler_full.mean}\n  stream={scaler_stream.mean}"
        )
        assert torch.allclose(scaler_full.var, scaler_stream.var, rtol=1e-5), (
            f"Var mismatch:\n  full={scaler_full.var}\n  stream={scaler_stream.var}"
        )

    def test_streaming_fit_single_chunk_fallback(self, tmp_path: Path):
        """When only 1 chunk is provided, streaming_fit should still work correctly."""
        torch.manual_seed(123)
        data = torch.randn(300, self.NUM_FEATURES)
        df = pl.DataFrame(
            {col: data[:, j].numpy() for j, col in enumerate(self.FEATURE_COLS)}
        )
        path = tmp_path / "single_chunk.parquet"
        df.write_parquet(path)

        scaler_full = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_full.fit(data)

        scaler_stream = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_stream.streaming_fit([path], self.FEATURE_COLS)

        assert scaler_stream.is_fitted
        assert torch.allclose(scaler_full.mean, scaler_stream.mean, rtol=1e-5)
        assert torch.allclose(scaler_full.var, scaler_stream.var, rtol=1e-5)

    def test_streaming_fit_transform_matches_full(
        self, synthetic_chunks: tuple[torch.Tensor, list[Path]]
    ):
        """The transform output should be identical regardless of how the scaler was fitted."""
        full_data, chunk_paths = synthetic_chunks

        scaler_full = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_full.fit(full_data)

        scaler_stream = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_stream.streaming_fit(chunk_paths, self.FEATURE_COLS)

        # Transform a small test batch
        test_batch = full_data[:32]
        out_full = scaler_full.transform(test_batch)
        out_stream = scaler_stream.transform(test_batch)

        assert torch.allclose(out_full, out_stream, rtol=1e-5), (
            "Transformed outputs diverge between fit() and streaming_fit()"
        )

    def test_streaming_fit_returns_self(
        self, synthetic_chunks: tuple[torch.Tensor, list[Path]]
    ):
        """streaming_fit should return self for chaining."""
        _, chunk_paths = synthetic_chunks
        scaler = TorchStandardScaler(num_features=self.NUM_FEATURES)
        result = scaler.streaming_fit(chunk_paths, self.FEATURE_COLS)
        assert result is scaler

    def test_streaming_fit_empty_chunks_raises(self):
        """Passing an empty list of chunk paths should raise ValueError."""
        scaler = TorchStandardScaler(num_features=self.NUM_FEATURES)
        with pytest.raises(ValueError, match="chunk_paths"):
            scaler.streaming_fit([], self.FEATURE_COLS)

    def test_streaming_fit_uneven_chunks(self, tmp_path: Path):
        """Handles chunks of different sizes correctly."""
        torch.manual_seed(99)
        sizes = [100, 300, 50]  # uneven chunk sizes
        full_rows = []

        chunk_paths = []
        for i, size in enumerate(sizes):
            data = torch.randn(size, self.NUM_FEATURES)
            full_rows.append(data)
            df = pl.DataFrame(
                {col: data[:, j].numpy() for j, col in enumerate(self.FEATURE_COLS)}
            )
            path = tmp_path / f"chunk_{i:03d}.parquet"
            df.write_parquet(path)
            chunk_paths.append(path)

        full_data = torch.cat(full_rows, dim=0)

        scaler_full = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_full.fit(full_data)

        scaler_stream = TorchStandardScaler(num_features=self.NUM_FEATURES)
        scaler_stream.streaming_fit(chunk_paths, self.FEATURE_COLS)

        assert torch.allclose(scaler_full.mean, scaler_stream.mean, rtol=1e-5)
        assert torch.allclose(scaler_full.var, scaler_stream.var, rtol=1e-5)
