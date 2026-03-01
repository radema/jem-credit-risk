import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class TabularAutoencoder(nn.Module):
    """
    A symmetric Autoencoder designed to map high-dimensional, sparse tabular datasets
    into a continuous, dense latent representation.
    """

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super(TabularAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 256)),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            spectral_norm(nn.Linear(256, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, latent_dim)),
            # Use LayerNorm or explicit Batchnorm for the latent representation.
            nn.LayerNorm(latent_dim),
        )

        self.decoder = nn.Sequential(
            spectral_norm(nn.Linear(latent_dim, 128)),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            spectral_norm(nn.Linear(128, 256)),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # No bounding activation on the output to allow natural regression mapping.
            nn.Linear(256, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projects input features to continuous D-dimensional space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs features from the continuous D-dimensional space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoencoderLoss(nn.Module):
    """
    Standard Mean Squared Error Loss for Autoencoder Reconstruction.
    """

    def __init__(self):
        super(AutoencoderLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, x_true: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        return self.mse(x_recon, x_true)
