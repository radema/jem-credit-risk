import os
import logging
import torch
import torch.optim as optim
import polars as pl
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.data.pipeline import run_pipeline
from src.data.config import DataPipelineConfig
from src.model.jem.data_utils import (
    split_data_chronologically,
    get_dataloaders,
    prepare_raw_dataframe,
    CreditRiskDataset,
)
from src.model.autoencoder.model import TabularAutoencoder, AutoencoderLoss

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # 1. Load the restricted manifold dataset
    cfg = DataPipelineConfig(
        data_dir="data/raw/home-credit-credit-risk-model-stability.zip",
        sample_ratio=0.1,
        cache_dir=".cache",
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

    # 2. Time-based split
    df_train, df_val = split_data_chronologically(df_full, val_weeks=12)
    logger.info(f"Train size: {df_train.shape[0]}, Val size: {df_val.shape[0]}")

    # 3. Create DataLoaders
    train_loader, val_loader_for_eval, scaler, feature_cols = get_dataloaders(
        df_train, df_val, batch_size=256
    )
    input_dim = len(feature_cols)
    latent_dim = 128
    logger.info(f"Input feature dimension: {input_dim}")

    # 4. Initialize Autoencoder
    autoencoder = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(
        device
    )
    criterion = AutoencoderLoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-5)

    epochs = 30
    logger.info("Starting Autoencoder Pre-training...")
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
            for x_real, _, _ in val_loader_for_eval:
                x_real = x_real.to(device)
                x_recon, _ = autoencoder(x_real)
                loss = criterion(x_real, x_recon)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader_for_eval)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs}] | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}"
        )

    # 5. Save the trained encoder weights
    os.makedirs("data/processed", exist_ok=True)
    encoder_path = "data/processed/encoder.pt"
    torch.save(autoencoder.encoder.state_dict(), encoder_path)
    logger.info(f"Encoder saved to {encoder_path}")

    # Also save the scaler
    scaler_path = "data/processed/scaler.pt"
    torch.save(scaler.state_dict(), scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # 6. Map entire dataset into fixed latent feature tensors Z
    # Ensure dataframe is prepared (null fillers etc)
    df_train = prepare_raw_dataframe(df_train, feature_cols)
    df_val = prepare_raw_dataframe(df_val, feature_cols)

    # We need to create standard dataloaders that don't shuffle or drop last.
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
