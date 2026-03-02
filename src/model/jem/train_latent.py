import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.jem.config import JEMConfig
from src.model.jem.data_utils import ChunkedLatentDataset, get_latent_dataloaders
from src.model.jem.loss import JEMLoss
from src.model.jem.model import LatentJEMWrapper, TabularJEM
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.scaler import TorchStandardScaler
from src.model.jem.train import evaluate_jem, train_jem_epoch
from src.utils.callbacks import EarlyStopping

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 1. Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Track B specific: latent dim is 64
    latent_dim = 64
    config = JEMConfig(
        hidden_dims=[256, 128],
        spectral_norm=True,
        sgld_steps=20,  # A bit more steps per epoch
        sgld_step_size=0.001,  # 10x smaller
        sgld_sigma=0.001,  # 10x smaller
        l2_energy_weight=1e-4,
    )

    # 2. Data Loading (Latent Space)
    train_path = Path("data/processed/latent_train.pt")
    val_path = Path("data/processed/latent_val.pt")
    latent_chunks_dir = Path("data/processed/latent_chunks")

    if latent_chunks_dir.exists() and any(latent_chunks_dir.glob("latent_chunk_*.pt")):
        logger.info(f"Using chunked latent data from {latent_chunks_dir}")
        chunk_paths = sorted(latent_chunks_dir.glob("latent_chunk_*.pt"))
        train_dataset = ChunkedLatentDataset(
            chunk_paths=chunk_paths,
            shuffle_buffer_size=config.shuffle_buffer_size,
        )
        train_loader = DataLoader(train_dataset, batch_size=256)

        # Validation remains monolithic for stability if available
        if val_path.exists():
            _, val_loader = get_latent_dataloaders(
                str(train_path), str(val_path), batch_size=256
            )
        else:
            # Fallback or take last chunk
            logger.warning(
                "No monolithic validation path found, using a chunk for validation."
            )
            val_path = chunk_paths[-1]
            val_data = torch.load(val_path, weights_only=False)
            val_dataset = TensorDataset(val_data["z"], val_data["y"], val_data["weeks"])
            val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    else:
        if not train_path.exists():
            logger.error(
                f"Latent data not found at {train_path} or chunks at {latent_chunks_dir}. Run train_autoencoder.py first."
            )
            return

        train_loader, val_loader = get_latent_dataloaders(
            str(train_path), str(val_path), batch_size=256
        )

    # 3. Model, Buffer, Sampler
    model = TabularJEM(input_dim=latent_dim, num_classes=2, config=config).to(device)
    buffer = SGLDReplayBuffer(buffer_size=10000, feature_dim=latent_dim)
    sampler = SGLDSampler(config=config)
    criterion = JEMLoss(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=5, mode="max")

    # 4. Training Loop
    epochs = 10  # Quick run for verification
    logger.info("Starting JEM training on Latent Space...")

    for epoch in range(1, epochs + 1):
        train_logs = train_jem_epoch(
            model, train_loader, optimizer, criterion, buffer, sampler, device
        )
        val_logs = evaluate_jem(model, val_loader, device)

        stability = val_logs["stability_result"]["stability_metric"]
        logger.info(
            f"Epoch [{epoch:02d}/{epochs}] | Loss: {train_logs['loss']:.4f} | Validation Stability: {stability:.4f}"
        )

        if early_stopping(stability, model):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Restore best weights
    early_stopping.restore_best_weights(model)

    # 5. Packaging (Integrate with Encoder and Scaler)
    logger.info("Packaging model into LatentJEMWrapper...")

    # Load Scaler
    # We need to know the original input dim. We can infer it from the saved scaler or assume it's known.
    # From previous runs, input_dim was ~347.
    # Let's check how the scaler was saved.
    scaler_path = "data/processed/scaler.pt"
    scaler_state = torch.load(scaler_path, map_location="cpu")
    # Infere input_dim from mean shape
    input_dim = scaler_state["mean"].shape[0]
    scaler = TorchStandardScaler(num_features=input_dim)
    scaler.load_state_dict(scaler_state)
    scaler.to(device)

    # Load Encoder
    from src.model.autoencoder.model import TabularAutoencoder

    ae = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae.encoder.load_state_dict(
        torch.load("data/processed/encoder.pt", map_location="cpu")
    )
    ae.encoder.to(device)

    # Create Wrapper
    wrapper = LatentJEMWrapper(scaler, ae.encoder, model)

    # Save the full package
    os.makedirs("models", exist_ok=True)
    torch.save(wrapper.state_dict(), "models/latent_jem_full.pt")
    logger.info("Final model saved to models/latent_jem_full.pt")


if __name__ == "__main__":
    main()
