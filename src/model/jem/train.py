import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.model.jem.model import TabularJEM
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.loss import JEMLoss
from src.model.jem.data_utils import calculate_gini_stability

logger = logging.getLogger(__name__)


def train_jem_epoch(
    model: TabularJEM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: JEMLoss,
    buffer: SGLDReplayBuffer,
    sampler: SGLDSampler,
    device: torch.device,
) -> dict:
    """
    Trains the model for one epoch.
    """
    model.train()

    total_loss, clf_loss, gen_loss, l2_loss = 0.0, 0.0, 0.0, 0.0
    all_targets, all_preds, all_weeks = [], [], []

    for batch_idx, batch in enumerate(loader):
        if len(batch) == 4:
            x_real, y_real, weeks, sample_weight = batch
        else:
            x_real, y_real, weeks = batch
            sample_weight = None

        x_real = x_real.to(device)
        y_real = y_real.to(device)
        if sample_weight is not None:
            sample_weight = sample_weight.to(device)

        # 1. Generate Fake Samples (requires eval mode conceptually, but we detach anyway)
        model.eval()
        x_fake = sampler.generate(model, buffer, batch_size=x_real.size(0))
        model.train()

        # 2. Forward pass and Loss Computation
        optimizer.zero_grad()
        loss_dict = criterion(
            model, x_real, y_real, x_fake, sample_weight=sample_weight
        )
        loss = loss_dict["total_loss"]

        # 3. Backward and step
        loss.backward()
        # Optional: clip gradients if we get instability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        clf_loss += loss_dict["clf_loss"].item()
        gen_loss += loss_dict["gen_loss"].item()
        l2_loss += loss_dict["l2_loss"].item()

        # For AUC/Gini tracking:
        # Optimization: use logits_real from loss_dict to avoid redundant forward pass
        logits = loss_dict["logits_real"].detach()
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of Class 1
        all_targets.append(y_real.cpu().numpy())
        all_preds.append(probs.cpu().numpy())
        all_weeks.append(weeks.numpy())

    num_batches = batch_idx + 1  # Works with both map-style and iterable loaders
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_weeks = np.concatenate(all_weeks)

    stability_metrics = calculate_gini_stability(all_targets, all_preds, all_weeks)

    return {
        "loss": total_loss / num_batches,
        "clf_loss": clf_loss / num_batches,
        "gen_loss": gen_loss / num_batches,
        "l2_loss": l2_loss / num_batches,
        "stability_result": stability_metrics,
    }


def evaluate_jem(model: TabularJEM, loader: DataLoader, device: torch.device) -> dict:
    """
    Evaluates the model on validation data.
    """
    model.eval()

    all_targets, all_preds, all_weeks = [], [], []
    all_energies = []

    with torch.no_grad():
        for x_real, y_real, weeks in loader:
            x_real = x_real.to(device)

            logits = model(x_real)
            probs = torch.softmax(logits, dim=1)[:, 1]
            # Optimization: compute energy from logits to avoid redundant forward pass
            energies = -torch.logsumexp(logits, dim=1)

            all_targets.append(y_real.numpy())
            all_preds.append(probs.cpu().numpy())
            all_weeks.append(weeks.numpy())
            all_energies.append(energies.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_weeks = np.concatenate(all_weeks)
    all_energies = np.concatenate(all_energies)

    stability_metrics = calculate_gini_stability(all_targets, all_preds, all_weeks)

    return {
        "stability_result": stability_metrics,
        "avg_energy": float(np.mean(all_energies)),
        "energies": all_energies,
        "probs": all_preds,
        "labels": all_targets,
    }
