import resource

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.export import export_to_chunked_parquet
from src.model.autoencoder.model import TabularAutoencoder
from src.model.autoencoder.train_autoencoder import (
    generate_latent_chunks,
    weighted_mse_loss,
)
from src.model.jem.config import JEMConfig
from src.model.jem.data_utils import (
    ChunkedLatentDataset,
    ChunkedParquetDataset,
)
from src.model.jem.loss import JEMLoss
from src.model.jem.model import TabularJEM
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.scaler import TorchStandardScaler
from src.model.jem.train import train_jem_epoch


@pytest.fixture
def e2e_setup(tmp_path):
    """Sets up a temporary directory structure for E2E testing."""
    data_dir = tmp_path / "data"
    chunks_dir = data_dir / "chunks"
    latent_dir = data_dir / "latent_chunks"
    artifact_dir = tmp_path / "artifacts"

    chunks_dir.mkdir(parents=True)
    latent_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    # Create synthetic data
    n_rows = 1000
    n_features = 20
    feature_cols = [f"feature_{i}" for i in range(n_features)]

    np.random.seed(42)
    data = {col: np.random.randn(n_rows).astype(np.float32) for col in feature_cols}
    data["target"] = np.random.randint(0, 2, n_rows).astype(np.int64)
    data["WEEK_NUM"] = np.repeat(np.arange(50), 20)  # 50 weeks, 20 rows each
    data["case_id"] = np.arange(n_rows)

    df = pl.DataFrame(data)

    # Export to chunks
    chunk_paths = export_to_chunked_parquet(
        df, str(chunks_dir), chunk_size=200, prefix="train_chunk"
    )

    return {
        "tmp_path": tmp_path,
        "chunks_dir": chunks_dir,
        "latent_dir": latent_dir,
        "artifact_dir": artifact_dir,
        "feature_cols": feature_cols,
        "chunk_paths": chunk_paths,
        "df_full": df,
    }


class TestChunkedPipelineE2E:
    def test_full_pipeline_flow(self, e2e_setup):
        """
        Tests the full flow:
        Export -> Scaler Fit -> AE Train -> Latent Gen -> JEM Train
        """
        setup = e2e_setup
        device = torch.device("cpu")

        # 1. Scaler Fit (Phase 1)
        scaler = TorchStandardScaler(num_features=len(setup["feature_cols"]))
        scaler.streaming_fit(setup["chunk_paths"], setup["feature_cols"])
        assert scaler.is_fitted.item() is True

        # 2. ChunkedParquetDataset (Phase 2)
        train_dataset = ChunkedParquetDataset(
            chunk_paths=setup["chunk_paths"],
            feature_cols=setup["feature_cols"],
            scaler=scaler,
            shuffle_buffer_size=100,
            seed=42,
        )
        train_loader = DataLoader(train_dataset, batch_size=32)

        # Verify dataset yields correctly
        all_rows = []
        for x, _y, _week, _weight in train_loader:
            all_rows.append(x)
        assert torch.cat(all_rows).shape[0] == 1000

        # 3. AE Training (Phase 3 - Mini Loop)
        latent_dim = 8
        autoencoder = TabularAutoencoder(
            input_dim=len(setup["feature_cols"]), latent_dim=latent_dim
        ).to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

        # 1 Epoch AE
        train_dataset.set_epoch(0)
        autoencoder.train()
        for x, _, _, weight in train_loader:
            optimizer.zero_grad()
            x_recon, _ = autoencoder(x.to(device))
            loss = weighted_mse_loss(x.to(device), x_recon, weight.to(device))
            loss.backward()
            optimizer.step()

        # Save AE artifacts
        encoder_path = setup["artifact_dir"] / "encoder.pt"
        torch.save(autoencoder.encoder.state_dict(), encoder_path)
        scaler_path = setup["artifact_dir"] / "scaler.pt"
        scaler.save(str(scaler_path))

        # 4. Latent Generation (Phase 4)
        latent_paths = generate_latent_chunks(
            autoencoder=autoencoder,
            scaler=scaler,
            chunk_paths=setup["chunk_paths"],
            feature_cols=setup["feature_cols"],
            output_dir=str(setup["latent_dir"]),
            device=device,
            batch_size=64,
        )
        assert len(latent_paths) == 5
        for p in latent_paths:
            data = torch.load(p, weights_only=True)
            assert "z" in data and "y" in data and "weeks" in data
            assert data["z"].shape[1] == latent_dim

        # 5. JEM Training (Phase 5)
        config = JEMConfig(hidden_dims=[16, 16], sgld_steps=2, buffer_size=100)
        latent_dataset = ChunkedLatentDataset(
            chunk_paths=latent_paths, shuffle_buffer_size=100, seed=42
        )
        latent_loader = DataLoader(latent_dataset, batch_size=32)

        jem_model = TabularJEM(input_dim=latent_dim, num_classes=2, config=config).to(
            device
        )
        jem_optimizer = torch.optim.Adam(jem_model.parameters(), lr=1e-3)
        jem_buffer = SGLDReplayBuffer(buffer_size=100, feature_dim=latent_dim)
        jem_sampler = SGLDSampler(config=config)
        jem_criterion = JEMLoss(config=config)

        # 1 Epoch JEM
        latent_dataset.set_epoch(0)
        logs = train_jem_epoch(
            jem_model,
            latent_loader,
            jem_optimizer,
            jem_criterion,
            jem_buffer,
            jem_sampler,
            device,
        )
        assert "loss" in logs
        assert np.isfinite(logs["loss"])

    def test_reproducibility(self, e2e_setup):
        """Verify two runs with same seed produce identical results."""
        setup = e2e_setup

        def run_ae_epoch(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            scaler = TorchStandardScaler(num_features=len(setup["feature_cols"]))
            scaler.streaming_fit(setup["chunk_paths"], setup["feature_cols"])

            dataset = ChunkedParquetDataset(
                chunk_paths=setup["chunk_paths"],
                feature_cols=setup["feature_cols"],
                scaler=scaler,
                shuffle_buffer_size=100,
                seed=seed,
            )
            loader = DataLoader(dataset, batch_size=32)

            TabularAutoencoder(input_dim=len(setup["feature_cols"]), latent_dim=4)
            dataset.set_epoch(0)

            # Sum of first batch features to check order
            first_batch = next(iter(loader))
            return first_batch[0].sum().item()

        val1 = run_ae_epoch(42)
        val2 = run_ae_epoch(42)
        val3 = run_ae_epoch(43)

        assert val1 == val2
        assert val1 != val3

    def test_memory_usage(self, e2e_setup):
        """Log peak memory usage during pipeline execution."""
        setup = e2e_setup

        before_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Run a minimal version of the pipeline
        scaler = TorchStandardScaler(num_features=len(setup["feature_cols"]))
        scaler.streaming_fit(setup["chunk_paths"], setup["feature_cols"])

        dataset = ChunkedParquetDataset(
            chunk_paths=setup["chunk_paths"],
            feature_cols=setup["feature_cols"],
            scaler=scaler,
            shuffle_buffer_size=100,
        )
        loader = DataLoader(dataset, batch_size=32)
        for _ in loader:
            pass

        after_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Linux ru_maxrss is in KB
        delta_mb = (after_rss - before_rss) / 1024
        print(f"\nPeak Memory Delta: {delta_mb:.2f} MB")
        # informational assert
        assert delta_mb < 500  # Should be very small for this synthetic data
