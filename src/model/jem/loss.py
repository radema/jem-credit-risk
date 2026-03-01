import torch
import torch.nn as nn
from src.model.jem.config import JEMConfig


class JEMLoss(nn.Module):
    """
    Joint Loss formulation for Energy-Based Models.
    Combines:
    1. Discriminative loss (CrossEntropy)
    2. Generative loss (Contrastive Divergence: E(x_real) - E(x_fake))
    3. Stabilization loss (L2 penalty on energies)
    """

    def __init__(self, config: JEMConfig, clf_weight: torch.Tensor = None):
        """
        Initializes the JEM loss module.

        Args:
            config (JEMConfig): Configuration object containing regularization weights.
            clf_weight (torch.Tensor, optional): Class weights for CrossEntropyLoss to handle imbalanced datasets.
        """
        super().__init__()
        self.config = config
        self.ce_loss = nn.CrossEntropyLoss(weight=clf_weight)

    def forward(
        self,
        model: nn.Module,
        x_real: torch.Tensor,
        y_real: torch.Tensor,
        x_fake: torch.Tensor,
    ) -> dict:
        """
        Computes the joint loss components.

        Args:
            model (nn.Module): The JEM model.
            x_real (torch.Tensor): Real data samples.
            y_real (torch.Tensor): Labels for real data.
            x_fake (torch.Tensor): Generated fake samples from SGLD.

        Returns:
            dict: Dictionary containing the total loss and its components.
        """
        # 1. Discriminative Loss: Predict the correct class
        logits_real = model(x_real)
        clf_loss = self.ce_loss(logits_real, y_real)

        # 2. Generative Loss (Contrastive Divergence)
        # We want p(x_real) to be high (low energy) and p(x_fake) to be low (high energy)
        # L_gen = E(x_real) - E(x_fake)

        # Optimization: compute e_real from logits_real to avoid redundant forward pass
        e_real = -torch.logsumexp(logits_real, dim=1)
        e_fake = model.compute_energy(x_fake)

        gen_loss = e_real.mean() - e_fake.mean()

        # 3. Energy Regularization (L2 penalty)
        # Prevents absolute magnitudes of energy from drifting to infinity,
        # which is a common failure mode in EBM training.
        l2_loss = self.config.l2_energy_weight * (e_real**2 + e_fake**2).mean()

        total_loss = clf_loss + gen_loss + l2_loss

        return {
            "total_loss": total_loss,
            "clf_loss": clf_loss,
            "gen_loss": gen_loss,
            "l2_loss": l2_loss,
            "e_real": e_real.mean(),
            "e_fake": e_fake.mean(),
            "logits_real": logits_real,
        }
