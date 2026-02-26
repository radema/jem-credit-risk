import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from typing import Tuple, List
from src.model.jem.model import TabularJEM
from src.model.autoencoder.model import TabularAutoencoder
from src.model.jem.scaler import TorchStandardScaler

logger = logging.getLogger(__name__)


def perform_inference(
    scaler: torch.nn.Module,
    autoencoder: torch.nn.Module,
    jem: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], np.ndarray]:
    """
    Runs an end-to-end inference loop over a DataLoader.
    Ensures memory efficiency by processing in batches and performing
    computations under torch.no_grad().

    Args:
        scaler: Pre-trained TorchStandardScaler instance.
        autoencoder: Pre-trained TabularAutoencoder instance.
        jem: Pre-trained TabularJEM instance.
        loader: DataLoader yielding (case_id, x_batch).
        device: Torch device (cpu/cuda/mps).

    Returns:
        tuple: (list of case_ids, numpy array of score probabilities).
    """
    # Initialize all modules into eval mode
    scaler.to(device).eval()
    autoencoder.to(device).eval()
    jem.to(device).eval()

    all_case_ids = []
    all_probs = []

    logger.info("Executing Batch Inference...")

    with torch.no_grad():
        for case_ids, x_batch in loader:
            x_batch = x_batch.to(device)

            # 1. Scale
            x_scaled = scaler.transform(x_batch)

            # 2. Project to Latent Space
            z = autoencoder.encode(x_scaled)

            # 3. Logistic Forward Pass
            logits = jem(z)

            # 4. Predict P(y=1|x)
            # Home Credit competition targets binary classification (0 or 1).
            # JEM outputs unnormalized logits for each class.
            # We apply Softmax to get probabilities and select class 1 (Credit Default).
            probs = torch.softmax(logits, dim=1)[:, 1]

            # Storage
            if torch.is_tensor(case_ids):
                all_case_ids.extend(case_ids.cpu().tolist())
            else:
                all_case_ids.extend(case_ids)

            all_probs.append(probs.cpu().numpy())

    # Consolidate results
    final_probs = np.concatenate(all_probs)
    logger.info(f"Inference Loop complete. Processed {len(all_case_ids)} samples.")

    return all_case_ids, final_probs


def load_inference_pipeline(
    artifact_dir: str,
    input_dim: int,
    latent_dim: int,
    num_classes: int,
    jem_config,
    device: torch.device = torch.device("cpu"),
) -> Tuple[TorchStandardScaler, TabularAutoencoder, TabularJEM]:
    """
    Helper to instantiate and load weights for the full inference stack.
    """
    import os

    # 1. Scaler
    scaler_path = os.path.join(artifact_dir, "scaler.pth")
    if os.path.exists(scaler_path):
        scaler = TorchStandardScaler.load(scaler_path)
        logger.info(f"Loaded Scaler from {scaler_path}")
    else:
        scaler = TorchStandardScaler(num_features=input_dim)
        logger.warning(f"Scaler NOT found at {scaler_path}. Using empty scaler.")

    # 2. Autoencoder
    ae = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ae_path = os.path.join(artifact_dir, "autoencoder.pth")
    if os.path.exists(ae_path):
        ae.load_state_dict(torch.load(ae_path, map_location=device, weights_only=True))
        logger.info(f"Loaded Autoencoder from {ae_path}")

    # 3. JEM
    jem = TabularJEM(input_dim=latent_dim, num_classes=num_classes, config=jem_config)
    jem_path = os.path.join(artifact_dir, "jem_model.pth")
    if os.path.exists(jem_path):
        jem.load_state_dict(
            torch.load(jem_path, map_location=device, weights_only=True)
        )
        logger.info(f"Loaded JEM Model from {jem_path}")

    return scaler, ae, jem
