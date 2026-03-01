import torch
import polars as pl
import numpy as np
from src.model.jem.data_utils import get_inference_dataloader
from src.model.jem.infer import perform_inference
from src.model.jem.model import TabularJEM
from src.model.autoencoder.model import TabularAutoencoder
from src.model.jem.scaler import TorchStandardScaler
from src.model.jem.config import JEMConfig


def test_inference_loop_consistency():
    # 1. Mock Data
    input_dim = 10
    latent_dim = 4
    num_classes = 2
    batch_size = 5
    num_samples = 12

    df = pl.DataFrame(
        {
            "case_id": list(range(num_samples)),
            **{f"feat_{i}": np.random.randn(num_samples) for i in range(input_dim)},
        }
    )
    feature_cols = [f"feat_{i}" for i in range(input_dim)]

    # 2. Mock Models
    scaler = TorchStandardScaler(num_features=input_dim)
    # Fit scaler so it doesn't complain
    scaler.fit(torch.randn(10, input_dim))

    ae = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    config = JEMConfig(hidden_dims=[16], spectral_norm=True)
    jem = TabularJEM(input_dim=latent_dim, num_classes=num_classes, config=config)

    # 3. Dataloader
    loader = get_inference_dataloader(df, feature_cols, batch_size=batch_size)

    # 4. Run Inference
    device = torch.device("cpu")
    case_ids, probs = perform_inference(
        scaler=scaler, autoencoder=ae, jem=jem, loader=loader, device=device
    )

    # 5. Verifications
    assert len(case_ids) == num_samples
    assert len(probs) == num_samples
    assert isinstance(probs, np.ndarray)
    assert probs.dtype == np.float32 or probs.dtype == np.float64

    # Ensure probabilities are in [0, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)

    # Ensure case_id order is preserved
    assert case_ids == list(range(num_samples))

    print("Inference loop test successful!")
