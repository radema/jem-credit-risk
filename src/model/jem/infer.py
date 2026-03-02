import logging

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.autoencoder.model import TabularAutoencoder
from src.model.jem.model import TabularJEM
from src.model.jem.scaler import TorchStandardScaler

logger = logging.getLogger(__name__)


def perform_inference(
    scaler: torch.nn.Module,
    autoencoder: torch.nn.Module,
    jem: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    return_energies: bool = False,
) -> tuple[list[int], np.ndarray, np.ndarray | None]:
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
        return_energies: Whether to return the energy of each sample.

    Returns:
        tuple: (list of case_ids, numpy array of score probabilities, optional energy array).
    """
    # Initialize all modules into eval mode
    scaler.to(device).eval()
    autoencoder.to(device).eval()
    jem.to(device).eval()

    all_case_ids = []
    all_probs = []
    all_energies = []

    logger.info("Executing Batch Inference...")

    with torch.no_grad():
        for case_ids, x_batch in loader:
            x_batch = x_batch.to(device)

            # 1. Scale
            x_scaled = scaler.transform(x_batch)

            # 2. Project to Latent Space
            # TabularAutoencoder has .encode(x)
            z = autoencoder.encode(x_scaled)

            # 3. Logistic Forward Pass
            logits = jem(z)

            # 4. Predict P(y=1|x)
            # Home Credit competition targets binary classification (0 or 1).
            # JEM outputs unnormalized logits for each class.
            # We apply Softmax to get probabilities and select class 1 (Credit Default).
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs)

            # 5. Energy (Optional)
            if return_energies:
                # Optimization: compute energy from logits to avoid redundant forward pass
                # TabularJEM compute_energy(z) = -LogSumExp(logits)
                energies = -torch.logsumexp(logits, dim=1)
                all_energies.append(energies)

            # Storage
            if torch.is_tensor(case_ids):
                all_case_ids.extend(case_ids.tolist())
            else:
                all_case_ids.extend(case_ids)

    # Consolidate results
    final_probs = torch.cat(all_probs).cpu().numpy()
    final_energies = torch.cat(all_energies).cpu().numpy() if return_energies else None

    logger.info(f"Inference Loop complete. Processed {len(all_case_ids)} samples.")

    return all_case_ids, final_probs, final_energies


def load_inference_pipeline(
    artifact_dir: str,
    input_dim: int,
    latent_dim: int,
    num_classes: int,
    jem_config,
    device: torch.device | None = None,
) -> tuple[TorchStandardScaler, TabularAutoencoder, TabularJEM]:
    if device is None:
        device = torch.device("cpu")
    """
    Helper to instantiate and load weights for the full inference stack.
    """
    import os

    if device is None:
        device = torch.device("cpu")

    # 1. Scaler
    scaler_path = os.path.join(artifact_dir, "scaler.pth")
    if os.path.exists(scaler_path):
        scaler = TorchStandardScaler.load(scaler_path)
        logger.info(f"Loaded Scaler from {scaler_path}")

        # Task 3.1 - Validate dimensions
        if input_dim != scaler.num_features:
            raise ValueError(
                f"Feature dimension mismatch: got {input_dim} features but scaler expects "
                f"{scaler.num_features}. Check that you are using the correct feature_cols.json."
            )
    else:
        scaler = TorchStandardScaler(num_features=input_dim)
        logger.warning(f"Scaler NOT found at {scaler_path}. Using empty scaler.")

    ae = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    ae_path = os.path.join(artifact_dir, "autoencoder.pth")
    if os.path.exists(ae_path):
        state_dict = torch.load(ae_path, map_location=device, weights_only=True)
        # Adapt keys if the saved dictionary contains only the encoder state
        # (meaning the keys look like "0.weight" instead of "encoder.0.weight")
        if not any(k.startswith("encoder.") for k in state_dict.keys()):
            state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}

        # strict=False allows loading just the encoder while ignoring the decoder
        ae.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded Autoencoder (Encoder weights) from {ae_path}")

    # 3. JEM
    jem = TabularJEM(input_dim=latent_dim, num_classes=num_classes, config=jem_config)
    jem_path = os.path.join(artifact_dir, "jem_model.pth")
    if os.path.exists(jem_path):
        jem.load_state_dict(
            torch.load(jem_path, map_location=device, weights_only=True)
        )
        logger.info(f"Loaded JEM Model from {jem_path}")

    return scaler, ae, jem
