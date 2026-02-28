import os
import json
import logging
import torch
import torch.optim as optim
import polars as pl
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.pipeline import run_pipeline
from src.data.config import DataPipelineConfig
from src.model.jem.data_utils import (
    split_data_chronologically,
    get_dataloaders,
    get_feature_cols,
    prepare_raw_dataframe,
    CreditRiskDataset,
    ChunkedParquetDataset,
)
from src.model.jem.scaler import TorchStandardScaler
from src.model.autoencoder.model import TabularAutoencoder, AutoencoderLoss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def weighted_mse_loss(
    x: torch.Tensor, x_recon: torch.Tensor, sample_weight: torch.Tensor
) -> torch.Tensor:
    """MSE weighted by per-sample class weights."""
    mse_per_sample = ((x - x_recon) ** 2).mean(dim=1)  # (batch,)
    return (mse_per_sample * sample_weight.to(mse_per_sample.device)).mean()


def _discover_chunks(chunk_dir: str, prefix: str = "train_chunk") -> list[Path]:
    """Discover and return sorted chunk Parquet files."""
    chunk_path = Path(chunk_dir)
    if not chunk_path.exists():
        return []
    return sorted(chunk_path.glob(f"{prefix}_*.parquet"))


def generate_latent_chunks(
    autoencoder: TabularAutoencoder,
    scaler: TorchStandardScaler,
    chunk_paths: list[Path],
    feature_cols: list[str],
    output_dir: str,
    device: torch.device,
    batch_size: int = 512,
):
    """
    Processes each Parquet chunk independently:
    1. Read chunk -> extract features + target + weeks.
    2. Scale features via scaler.transform().
    3. Encode via autoencoder.encode() in batches.
    4. Save {z, y, weeks} as latent_chunk_XXX.pt.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    autoencoder.eval()

    for i, chunk_path in enumerate(chunk_paths):
        df = pl.read_parquet(chunk_path)
        df = prepare_raw_dataframe(df, feature_cols)

        x = torch.tensor(df.select(feature_cols).to_numpy(), dtype=torch.float32)
        y = torch.tensor(df["target"].to_numpy(), dtype=torch.long)
        weeks = torch.tensor(df["WEEK_NUM"].to_numpy(), dtype=torch.float32)

        # Scale
        x_scaled = scaler.transform(x)

        # Encode in batches to avoid GPU OOM
        z_list = []
        with torch.no_grad():
            for j in range(0, len(x_scaled), batch_size):
                batch = x_scaled[j : j + batch_size].to(device)
                z = autoencoder.encode(batch)
                z_list.append(z.cpu())

        z_all = torch.cat(z_list, dim=0)

        # Save
        out_path = output_dir / f"latent_chunk_{i + 1:03d}.pt"
        torch.save({"z": z_all, "y": y, "weeks": weeks}, out_path)

        logger.info(f"Saved {out_path} (shape: {z_all.shape})")

    return sorted(output_dir.glob("latent_chunk_*.pt"))


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # --- Configuration ---
    artifact_dir = "models/artifacts"
    chunk_dir = "data/processed/chunks"
    latent_dim = 128
    batch_size = 256
    epochs = 30
    lr = 1e-3
    weight_decay = 1e-5

    # 1. Run the data pipeline
    cfg = DataPipelineConfig(
        data_dir="data/raw/home-credit-credit-risk-model-stability.zip",
        sample_ratio=0.1,
        cache_dir=".cache",
        chunked_export=True,
        artifact_dir=artifact_dir,
    )
    logger.info("Building feature manifold...")
    df_full = run_pipeline(
        config=cfg,
        depth_0_tables=["train_static_0", "train_static_cb_0"],
        depth_1_tables=[
            "train_person_1",
            "train_deposit",
            "train_debitcard",
            "train_other",
        ],
        depth_2_tables=[],
    )
    logger.info(f"Data shape: {df_full.shape}")

    # 2. Detect whether chunked data is available
    chunk_paths = _discover_chunks(chunk_dir)
    use_chunked = len(chunk_paths) > 0

    if use_chunked:
        logger.info(
            f"Chunked mode: found {len(chunk_paths)} chunk files in {chunk_dir}"
        )
        _train_chunked(
            df_full=df_full,
            chunk_paths=chunk_paths,
            artifact_dir=artifact_dir,
            device=device,
            latent_dim=latent_dim,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        logger.info("In-memory mode: no chunk files found, using legacy path")
        _train_in_memory(
            df_full=df_full,
            artifact_dir=artifact_dir,
            device=device,
            latent_dim=latent_dim,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )


def _train_chunked(
    df_full: pl.DataFrame,
    chunk_paths: list[Path],
    artifact_dir: str,
    device: torch.device,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    """Chunked training path: streaming scaler + ChunkedParquetDataset."""

    # 1. Determine feature columns
    feature_cols = get_feature_cols(df_full)
    df_full = prepare_raw_dataframe(df_full, feature_cols)

    # Low variance filtering
    variances = df_full.select(pl.col(feature_cols).var()).to_dicts()[0]
    low_var_cols = [c for c in feature_cols if variances[c] < 1e-4]
    if low_var_cols:
        logger.info(f"Dropping {len(low_var_cols)} features with variance < 1e-4")
        feature_cols = [c for c in feature_cols if c not in low_var_cols]
    logger.info(f"Features after low-variance filtering: {len(feature_cols)}")

    # Persist feature_cols
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    with open(artifact_path / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    logger.info(f"Saved feature_cols to {artifact_path / 'feature_cols.json'}")

    input_dim = len(feature_cols)

    # 2. Streaming scaler fit (Welford's algorithm)
    logger.info("Fitting scaler via streaming_fit (Welford's online algorithm)...")
    scaler = TorchStandardScaler(num_features=input_dim)
    scaler.streaming_fit(chunk_paths, feature_cols)
    logger.info("Scaler fitted successfully via streaming pass.")

    # 3. Training DataLoader (chunked, streamed from disk)
    train_dataset = ChunkedParquetDataset(
        chunk_paths=chunk_paths,
        feature_cols=feature_cols,
        scaler=scaler,
        shuffle_buffer_size=50_000,
        shuffle_chunks=True,
        seed=42,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)

    # 4. Validation DataLoader (in-memory, small)
    _, df_val = split_data_chronologically(df_full, val_weeks=12)
    df_val = prepare_raw_dataframe(df_val, feature_cols)
    val_dataset = CreditRiskDataset(df_val, feature_cols, scaler=scaler, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Initialize model
    autoencoder = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(
        device
    )
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)

    # 6. Training loop
    logger.info("Starting Autoencoder Pre-training (chunked mode)...")
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        autoencoder.train()
        total_loss = 0.0
        num_batches = 0

        for x, _, _, sample_weight in train_loader:
            x = x.to(device)
            sample_weight = sample_weight.to(device)
            optimizer.zero_grad()

            x_recon, _ = autoencoder(x)
            loss = weighted_mse_loss(x, x_recon, sample_weight)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation
        autoencoder.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for x_real, _, _ in val_loader:
                x_real = x_real.to(device)
                x_recon, _ = autoencoder(x_real)
                loss = ((x_real - x_recon) ** 2).mean()
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}"
        )

    # 7. Save artifacts
    _save_artifacts(autoencoder, scaler)

    # 8. Generate latent chunks
    latent_chunk_dir = "data/processed/latent_chunks"
    logger.info(f"Generating latent chunks in {latent_chunk_dir}...")
    generate_latent_chunks(
        autoencoder=autoencoder,
        scaler=scaler,
        chunk_paths=chunk_paths,
        feature_cols=feature_cols,
        output_dir=latent_chunk_dir,
        device=device,
    )

    # 9. Also generate validation latent (small, in-memory)
    # Reuse legacy logic for single file latent_val.pt
    val_dataset = CreditRiskDataset(df_val, feature_cols, scaler=scaler, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    Z_list, y_list, weeks_list = [], [], []
    autoencoder.eval()
    logger.info("Generating validation latent (in-memory)...")
    with torch.no_grad():
        for x, y, weeks in tqdm(val_loader, desc="Mapping Val"):
            z = autoencoder.encode(x.to(device))
            Z_list.append(z.cpu())
            y_list.append(y)
            weeks_list.append(weeks)

    val_latent = {
        "z": torch.cat(Z_list, dim=0),
        "y": torch.cat(y_list, dim=0),
        "weeks": torch.cat(weeks_list, dim=0),
    }
    torch.save(val_latent, "data/processed/latent_val.pt")
    logger.info(f"Saved latent_val.pt with shape: {val_latent['z'].shape}")


def _train_in_memory(
    df_full: pl.DataFrame,
    artifact_dir: str,
    device: torch.device,
    latent_dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    """Legacy in-memory training path (unchanged behavior)."""

    # Time-based split
    df_train, df_val = split_data_chronologically(df_full, val_weeks=12)
    logger.info(f"Train size: {df_train.shape[0]}, Val size: {df_val.shape[0]}")

    # Create DataLoaders (handles feature selection, scaling, sampling)
    train_loader, val_loader, scaler, feature_cols = get_dataloaders(
        df_train, df_val, batch_size=batch_size, artifact_dir=artifact_dir
    )
    input_dim = len(feature_cols)
    logger.info(f"Input feature dimension: {input_dim}")

    # Initialize model
    autoencoder = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(
        device
    )
    criterion = AutoencoderLoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    logger.info("Starting Autoencoder Pre-training (in-memory mode)...")
    for epoch in range(1, epochs + 1):
        autoencoder.train()
        total_loss = 0.0

        for batch_idx, (x_real, _, _) in enumerate(train_loader):
            x_real = x_real.to(device)
            optimizer.zero_grad()

            x_recon, _ = autoencoder(x_real)
            loss = criterion(x_real, x_recon)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        autoencoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_real, _, _ in val_loader:
                x_real = x_real.to(device)
                x_recon, _ = autoencoder(x_real)
                loss = criterion(x_real, x_recon)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}"
        )

    # Save artifacts
    _save_artifacts(autoencoder, scaler)

    # Generate latent tensors (legacy path only — chunked path defers to TASK-5)
    _generate_latent_datasets(
        autoencoder, scaler, df_train, df_val, feature_cols, device
    )


def _save_artifacts(autoencoder: TabularAutoencoder, scaler: TorchStandardScaler):
    """Save encoder and scaler artifacts."""
    os.makedirs("data/processed", exist_ok=True)

    encoder_path = "data/processed/encoder.pt"
    torch.save(autoencoder.encoder.state_dict(), encoder_path)
    logger.info(f"Encoder saved to {encoder_path}")

    scaler_path = "data/processed/scaler.pt"
    scaler.save(scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")


def _generate_latent_datasets(
    autoencoder: TabularAutoencoder,
    scaler: TorchStandardScaler,
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    feature_cols: list[str],
    device: torch.device,
):
    """Generate latent datasets for the in-memory legacy path."""
    df_train = prepare_raw_dataframe(df_train, feature_cols)
    df_val = prepare_raw_dataframe(df_val, feature_cols)

    seq_train_dataset = CreditRiskDataset(
        df_train, feature_cols, scaler=scaler, is_train=False
    )
    seq_val_dataset = CreditRiskDataset(
        df_val, feature_cols, scaler=scaler, is_train=False
    )

    seq_train_loader = DataLoader(seq_train_dataset, batch_size=512, shuffle=False)
    seq_val_loader = DataLoader(seq_val_dataset, batch_size=512, shuffle=False)

    autoencoder.eval()

    def generate_latent_dataset(loader, desc="Mapping"):
        Z_list, y_list, weeks_list = [], [], []
        with torch.no_grad():
            for x, y, weeks in tqdm(loader, desc=desc):
                z = autoencoder.encode(x.to(device))
                Z_list.append(z.cpu())
                y_list.append(y)
                weeks_list.append(weeks)
        return {
            "z": torch.cat(Z_list, dim=0),
            "y": torch.cat(y_list, dim=0),
            "weeks": torch.cat(weeks_list, dim=0),
        }

    train_latent = generate_latent_dataset(seq_train_loader, desc="Mapping Train")
    val_latent = generate_latent_dataset(seq_val_loader, desc="Mapping Val")

    torch.save(train_latent, "data/processed/latent_train.pt")
    torch.save(val_latent, "data/processed/latent_val.pt")

    logger.info(f"Saved latent_train.pt with shape: {train_latent['z'].shape}")
    logger.info(f"Saved latent_val.pt with shape: {val_latent['z'].shape}")


if __name__ == "__main__":
    main()
