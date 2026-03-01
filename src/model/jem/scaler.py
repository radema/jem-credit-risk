from pathlib import Path

import polars as pl
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

    def streaming_fit(
        self,
        chunk_paths: list[Path],
        feature_cols: list[str],
    ) -> "TorchStandardScaler":
        """
        Computes global mean and variance using Welford's parallel/batch
        online algorithm by iterating through Parquet chunk files.

        Uses FP64 accumulators internally, casts to FP32 at the end.
        Memory: O(num_features), independent of total dataset size.

        Args:
            chunk_paths: Ordered list of Parquet file paths to process.
            feature_cols: Column names to select from each Parquet file.

        Returns:
            self
        """
        if not chunk_paths:
            raise ValueError("chunk_paths must be a non-empty list of Parquet paths.")

        # --- FP64 accumulators for numerical stability ---
        running_mean = torch.zeros(self.num_features, dtype=torch.float64)
        running_m2 = torch.zeros(self.num_features, dtype=torch.float64)
        total_count: int = 0

        for path in chunk_paths:
            # Read chunk: Parquet → Polars → NumPy → FP64 Tensor
            chunk_np = pl.read_parquet(path).select(feature_cols).to_numpy()
            chunk_t = torch.from_numpy(chunk_np).to(torch.float64)
            n = chunk_t.shape[0]

            if n == 0:
                continue

            # Chunk-level statistics (FP64)
            chunk_mean = chunk_t.mean(dim=0)
            chunk_m2 = ((chunk_t - chunk_mean) ** 2).sum(dim=0)

            # Welford's parallel combine
            combined_count = total_count + n
            delta = chunk_mean - running_mean

            running_mean = running_mean + delta * (n / combined_count)
            running_m2 = (
                running_m2 + chunk_m2 + (delta**2) * (total_count * n / combined_count)
            )
            total_count = combined_count

        # Unbiased sample variance, then cast to FP32
        variance = running_m2 / (total_count - 1) if total_count > 1 else running_m2

        self.mean.copy_(running_mean.float())
        self.var.copy_(variance.float())
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

        return torch.clamp((x - self.mean) / std, -10.0, 10.0)

    def save(self, path: str):
        """Saves the scaler state to a file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "num_features": self.num_features,
                "eps": self.eps,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TorchStandardScaler":
        """Loads the scaler state from a file."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        if "num_features" in state:
            scaler = cls(num_features=state["num_features"], eps=state.get("eps", 1e-8))
            scaler.load_state_dict(state["state_dict"])
        else:
            # It's just a raw state_dict
            num_features = state["mean"].shape[0]
            scaler = cls(num_features=num_features)
            scaler.load_state_dict(state)
        return scaler

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
