import torch

from src.model.autoencoder.model import AutoencoderLoss, TabularAutoencoder


def test_tabular_autoencoder_initialization():
    input_dim = 347
    latent_dim = 64
    autoencoder = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim)

    # Assert network structure is created
    assert hasattr(autoencoder, "encoder")
    assert hasattr(autoencoder, "decoder")


def test_tabular_autoencoder_forward_shapes():
    input_dim = 300
    latent_dim = 32
    batch_size = 16

    autoencoder = TabularAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    x = torch.randn(batch_size, input_dim)

    # Test encode
    z = autoencoder.encode(x)
    assert z.shape == (batch_size, latent_dim), (
        f"Expected encode shape {(batch_size, latent_dim)}, got {z.shape}"
    )

    # Test decode
    x_recon = autoencoder.decode(z)
    assert x_recon.shape == (batch_size, input_dim), (
        f"Expected decode shape {(batch_size, input_dim)}, got {x_recon.shape}"
    )

    # Test full forward pass
    autoencoder.eval()
    with torch.no_grad():
        z = autoencoder.encode(x)
        x_recon = autoencoder.decode(z)
        x_recon_fwd, z_fwd = autoencoder(x)

    assert torch.allclose(z, z_fwd)
    assert torch.allclose(x_recon, x_recon_fwd)


def test_autoencoder_loss():
    loss_fn = AutoencoderLoss()

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    x_recon = torch.tensor([[1.0, 2.0], [3.5, 4.5]])

    # Expected standard MSE: ((1-1)^2 + (2-2)^2 + (3-3.5)^2 + (4-4.5)^2) / 4 = (0 + 0 + 0.25 + 0.25) / 4 = 0.5 / 4 = 0.125
    expected_loss = 0.125
    actual_loss = loss_fn(x, x_recon).item()

    assert torch.isclose(torch.tensor(actual_loss), torch.tensor(expected_loss))


def test_autoencoder_numerical_stability():
    """Test with extreme values to ensure no naive overflow (since we use standard normalization)."""
    autoencoder = TabularAutoencoder(input_dim=10, latent_dim=5)
    autoencoder.eval()  # prevent batchnorm issues with artificial tiny batch size
    loss_fn = AutoencoderLoss()

    # Extremely large input
    x_extreme = torch.randn(4, 10) * 1e5

    x_recon_extreme, z_extreme = autoencoder(x_extreme)
    loss = loss_fn(x_extreme, x_recon_extreme)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert not torch.isnan(z_extreme).any()
    assert not torch.isnan(x_recon_extreme).any()
