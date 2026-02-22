import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.model.jem.config import JEMConfig


class TabularJEM(nn.Module):
    """
    Joint Energy-Based Model for tabular data.

    This architecture is a simple feedforward neural network that outputs unnormalized
    class logits. For numerical stability during SGLD, it exclusively uses
    Spectral Normalization on all Linear layers and avoids BatchNorm entirely
    to ensure perfectly deterministic energy calculations for single samples
    (i.e., the energy of a sample shouldn't depend on other samples in a batch).
    """

    def __init__(self, input_dim: int, num_classes: int, config: JEMConfig):
        """
        Initializes the Tabular JEM architecture.

        Args:
            input_dim (int): Number of input features.
            num_classes (int): Number of target classes.
            config (JEMConfig): Configuration object holding architecture specifics.
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.config = config

        layers = []
        in_features = input_dim

        # Build hidden layers
        for hidden_dim in config.hidden_dims:
            linear = nn.Linear(in_features, hidden_dim)
            # Apply spectral normalization to constrain the Lipschitz constant
            if config.spectral_norm:
                linear = spectral_norm(linear)

            layers.append(linear)
            # LeakyReLU is commonly used in energy-based models to avoid dead gradients
            layers.append(nn.LeakyReLU(negative_slope=0.2))

            in_features = hidden_dim

        # Output layer mapping to class logits
        out_linear = nn.Linear(in_features, num_classes)
        if config.spectral_norm:
            out_linear = spectral_norm(out_linear)

        layers.append(out_linear)

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding unnormalized class logits.

        Args:
            x (torch.Tensor): Input samples of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Class logits of shape (batch_size, num_classes).
        """
        return self.net(x)

    def compute_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the real energy of the input samples.
        Following the standard formalism for classifier-based JEMs:
        E(x) = -LogSumExp_y( f_theta(x)[y] )

        Args:
            x (torch.Tensor): Input samples of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Energy values of shape (batch_size,).
        """
        logits = self.forward(x)
        # LogSumExp across the class dimension (dim=1)
        return -torch.logsumexp(logits, dim=1)
