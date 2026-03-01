import logging
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

from src.model.jem.model import TabularJEM
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.loss import JEMLoss
from src.model.jem.scaler import TorchStandardScaler

logger = logging.getLogger(__name__)


def calculate_gini_stability(
    y_true: np.ndarray, y_pred: np.ndarray, week_nums: np.ndarray
) -> dict:
    """
    Calculates the Gini stability metric over WEEK_NUM.
    gini = 2 * AUC - 1
    stability metric = mean(gini) + 88.0 * min(0, a) - 0.5 * std(residuals)
    where a is the slope of the linear regression fit through weekly gini scores.
    """
    weeks = np.unique(week_nums)
    ginis = []
    w_list = []

    for w in weeks:
        mask = week_nums == w
        if len(np.unique(y_true[mask])) < 2:
            # Need both classes to calculate AUC
            continue
        auc = roc_auc_score(y_true[mask], y_pred[mask])
        gini = 2 * auc - 1
        ginis.append(gini)
        w_list.append(w)

    if len(ginis) < 2:
        return {
            "stability_metric": 0.0,
            "mean_gini": 0.0,
            "falling_rate": 0.0,
            "std_residuals": 0.0,
        }

    ginis = np.array(ginis)
    w_list = np.array(w_list)

    # Linear regression
    a, b = np.polyfit(w_list, ginis, deg=1)

    falling_rate = min(0.0, a)
    residuals = ginis - (a * w_list + b)
    std_residuals = np.std(residuals)

    mean_gini = np.mean(ginis)
    stability_metric = mean_gini + 88.0 * falling_rate - 0.5 * std_residuals

    return {
        "stability_metric": stability_metric,
        "mean_gini": mean_gini,
        "falling_rate": falling_rate,
        "std_residuals": std_residuals,
        "weekly_ginis": ginis.tolist(),
    }


class CreditRiskDataset(Dataset):
    """
    PyTorch Dataset for Credit Risk.
    Expects unscaled features and will scale them using TorchStandardScaler if provided.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        scaler: TorchStandardScaler = None,
        is_train: bool = True,
    ):
        self.features = torch.tensor(
            df.select(feature_cols).to_numpy(), dtype=torch.float32
        )

        self.scaler = scaler
        if self.scaler is not None:
            if is_train:
                self.scaler.fit(self.features)
            self.features = self.scaler.transform(self.features)

        self.targets = torch.tensor(df["target"].to_numpy(), dtype=torch.long)
        self.week_nums = df["WEEK_NUM"].to_numpy()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.week_nums[idx]


def get_dataloaders(
    df_train: pl.DataFrame, df_val: pl.DataFrame, batch_size: int = 256
):
    """
    Prepares DataLoaders for train and validation using weighted random sampling to combat class imbalance.
    """
    exclude_cols = ["case_id", "MONTH", "WEEK_NUM", "target"]
    feature_cols = [
        col
        for col in df_train.columns
        if col not in exclude_cols and df_train[col].dtype.is_numeric()
    ]

    # Fill nulls in features directly inside the dataframe to prevent dtype object fallback
    df_train = df_train.with_columns(
        pl.col(feature_cols).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
    )
    df_val = df_val.with_columns(
        pl.col(feature_cols).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
    )

    scaler = TorchStandardScaler(num_features=len(feature_cols))

    train_dataset = CreditRiskDataset(
        df_train, feature_cols, scaler=scaler, is_train=True
    )
    val_dataset = CreditRiskDataset(df_val, feature_cols, scaler=scaler, is_train=False)

    # Compute sample weights for imbalanced classification
    class_counts = torch.bincount(train_dataset.targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_dataset.targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler, feature_cols


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

    for batch_idx, (x_real, y_real, weeks) in enumerate(loader):
        x_real = x_real.to(device)
        y_real = y_real.to(device)

        # 1. Generate Fake Samples (requires eval mode conceptually, but we detach anyway)
        model.eval()
        x_fake = sampler.generate(model, buffer, batch_size=x_real.size(0))
        model.train()

        # 2. Forward pass and Loss Computation
        optimizer.zero_grad()
        loss_dict = criterion(model, x_real, y_real, x_fake)
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

    num_batches = len(loader)
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
    }
