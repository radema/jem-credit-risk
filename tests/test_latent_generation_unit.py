from pathlib import Path

import numpy as np
import polars as pl
import torch

from src.model.autoencoder.model import TabularAutoencoder
from src.model.autoencoder.train_autoencoder import generate_latent_chunks
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
        data["target"] = np.random.randint(0, 2, rows_per_chunk).astype(np.int64)
        data["WEEK_NUM"] = np.full(rows_per_chunk, c, dtype=np.int64)
        df = pl.DataFrame(data)
        df.write_parquet(chunk_dir / f"train_chunk_{c + 1:03d}.parquet")

    return chunk_dir, feature_cols


def test_generate_latent_chunks(tmp_path: Path):
    n_chunks = 3
    rows_per_chunk = 100
    n_features = 4
    latent_dim = 2

    chunk_dir, feature_cols = _create_synthetic_chunks(
        tmp_path,
        n_chunks=n_chunks,
        rows_per_chunk=rows_per_chunk,
        n_features=n_features,
    )
    chunk_paths = sorted(chunk_dir.glob("train_chunk_*.parquet"))

    # Setup model and scaler
    ae = TabularAutoencoder(input_dim=n_features, latent_dim=latent_dim)
    scaler = TorchStandardScaler(num_features=n_features)
    # Mock fit
    scaler.mean.copy_(torch.zeros(n_features))
    scaler.var.copy_(torch.ones(n_features))
    scaler.is_fitted.copy_(torch.tensor(True))

    output_dir = tmp_path / "latent_chunks"
    device = torch.device("cpu")

    # Run generation
    latent_paths = generate_latent_chunks(
        autoencoder=ae,
        scaler=scaler,
        chunk_paths=chunk_paths,
        feature_cols=feature_cols,
        output_dir=str(output_dir),
        device=device,
        batch_size=32,
    )

    # Assertions
    assert len(latent_paths) == n_chunks
    for i, path in enumerate(latent_paths):
        assert path.exists()
        data = torch.load(path, weights_only=False)
        assert "z" in data
        assert "y" in data
        assert "weeks" in data

        assert data["z"].shape == (rows_per_chunk, latent_dim)
        assert data["y"].shape == (rows_per_chunk,)
        assert data["weeks"].shape == (rows_per_chunk,)

        # Verify week number matches chunk index (as per _create_synthetic_chunks)
        assert torch.all(data["weeks"] == i)
    print(f"Successfully generated {len(latent_paths)} latent chunks.")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_generate_latent_chunks(Path(tmp_dir))
    print("Test passed!")
