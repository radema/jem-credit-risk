import torch
import torch.nn as nn


class TorchStandardScaler(nn.Module):
    """
    A PyTorch-native standard scaler for tabular data.

    This scaler computes and stores the mean and variance of numerical
    features and provides a fast, hardware-agnostic transformation.
    It inherits from nn.Module to leverage PyTorch's buffer management,
    making it easy to save/load state dicts and move across devices (CPU/GPU/MPS).
    """

    def __init__(self, num_features: int, eps: float = 1e-8):
        """
        Initializes the standard scaler.

        Args:
            num_features (int): Number of features to scale.
            eps (float): A small constant added to the variance to avoid
                         division by zero during normalization.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Register buffers so they are moved to the correct device automatically
        # and saved in the state dict.
        self.register_buffer("mean", torch.zeros(num_features))
        self.register_buffer("var", torch.ones(num_features))
        self.register_buffer("is_fitted", torch.tensor(False))

    def fit(self, x: torch.Tensor) -> "TorchStandardScaler":
        """
        Computes and stores the mean and variance of the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            self
        """
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor of shape (batch_size, num_features), got {x.dim()}D tensor."
            )

        if x.shape[1] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.shape[1]}")

        # Compute mean and variance along the batch dimension (dim=0)
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)  # Unbiased sample variance

        # Update buffers
        self.mean.copy_(mean)
        self.var.copy_(var)
        self.is_fitted.copy_(torch.tensor(True))

        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Scales the input tensor using the fitted mean and variance.

        Args:
            x (torch.Tensor): Input tensor to scale.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Scaler has not been fitted yet. Call fit() before transform()."
            )

        # If variance is very small (constant feature), set it to 1.0 to avoid exploding features
        var_safe = torch.where(self.var < self.eps, 1.0, self.var)
        std = torch.sqrt(var_safe)

        return (x - self.mean) / std

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to fit the scaler and then transform the data.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        return self.fit(x).transform(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Alias for transform(), conforming to the nn.Module interface.
        """
        return self.transform(x)
