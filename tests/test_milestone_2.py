import torch
import pytest
import os
from src.model.jem.model import TabularJEM
from src.model.jem.config import JEMConfig
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.data_utils import get_latent_dataloaders


def test_jem_latent_space_compatibility():
    # Test if JEM can handle 64-dimensional latent input properly
    latent_dim = 64
    num_classes = 2
    config = JEMConfig(hidden_dims=[128, 64])

    model = TabularJEM(input_dim=latent_dim, num_classes=num_classes, config=config)

    # Check weights/architecture
    assert model.input_dim == 64
    x = torch.randn(16, latent_dim)
    logits = model(x)
    assert logits.shape == (16, num_classes)

    energy = model.compute_energy(x)
    assert energy.shape == (16,)


def test_sampler_latent_space_compatibility():
    # Test if Sampler/Buffer can handle 64-dimensional latent space
    latent_dim = 64
    buffer_size = 100
    config = JEMConfig(sgld_steps=5, sgld_step_size=0.1, sgld_sigma=0.01)

    buffer = SGLDReplayBuffer(buffer_size=buffer_size, feature_dim=latent_dim)
    sampler = SGLDSampler(config=config)

    model = TabularJEM(input_dim=latent_dim, num_classes=2, config=config)

    # Generate samples in latent space
    samples = sampler.generate(model, buffer, batch_size=8)

    assert samples.shape == (8, latent_dim)
    assert buffer.pointer == 8


def test_latent_data_loading():
    train_path = "data/processed/latent_train.pt"
    val_path = "data/processed/latent_val.pt"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        pytest.skip("Latent files not found. Run train_autoencoder.py first.")

    train_loader, val_loader = get_latent_dataloaders(
        train_path, val_path, batch_size=32
    )

    assert len(train_loader) > 0
    z, y, weeks = next(iter(train_loader))
    assert z.shape == (32, 64)
    assert y.shape == (32,)
    assert weeks.shape == (32,)


def test_latent_jem_wrapper():
    from src.model.jem.model import LatentJEMWrapper, TabularJEM
    from src.model.jem.scaler import TorchStandardScaler
    import torch.nn as nn

    input_dim = 100
    latent_dim = 64
    config = JEMConfig(hidden_dims=[128])

    scaler = TorchStandardScaler(input_dim)
    # Fit the scaler to avoid RuntimeError
    scaler.fit(torch.randn(10, input_dim))
    # Mock encoder
    encoder = nn.Linear(input_dim, latent_dim)
    jem = TabularJEM(latent_dim, num_classes=2, config=config)

    wrapper = LatentJEMWrapper(scaler, encoder, jem)

    x_raw = torch.randn(16, input_dim)
    logits = wrapper(x_raw)
    energy = wrapper.compute_energy(x_raw)

    assert logits.shape == (16, 2)
    assert energy.shape == (16,)
