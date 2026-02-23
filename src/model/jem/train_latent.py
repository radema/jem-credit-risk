import os
import torch
import logging
import numpy as np
from src.model.jem.config import JEMConfig
from src.model.jem.model import TabularJEM, LatentJEMWrapper
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.loss import JEMLoss
from src.model.jem.train import train_jem_epoch, evaluate_jem
from src.model.jem.data_utils import get_latent_dataloaders
from src.model.jem.scaler import TorchStandardScaler

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
        sgld_steps=10,
        sgld_step_size=0.01,
        sgld_sigma=0.01,
        l2_energy_weight=1e-4,
    )

    # 2. Data Loading (Latent Space)
    train_path = "data/processed/latent_train.pt"
    val_path = "data/processed/latent_val.pt"

    if not os.path.exists(train_path):
        logger.error(
            f"Latent data not found at {train_path}. Run train_autoencoder.py first."
        )
        return

    train_loader, val_loader = get_latent_dataloaders(
        train_path, val_path, batch_size=256
    )

    # 3. Model, Buffer, Sampler
    model = TabularJEM(input_dim=latent_dim, num_classes=2, config=config).to(device)
    buffer = SGLDReplayBuffer(buffer_size=10000, feature_dim=latent_dim)
    sampler = SGLDSampler(config=config)
    criterion = JEMLoss(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

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
