import torch
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, TensorDataset
from src.model.jem.scaler import TorchStandardScaler


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


def prepare_raw_dataframe(df: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Fills nulls and ensures correct types for the feature columns."""
    return df.with_columns(
        pl.col(feature_cols).cast(pl.Float32).fill_null(0.0).fill_nan(0.0)
    )


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Identifies numeric feature columns excluding metadata."""
    exclude_cols = ["case_id", "MONTH", "WEEK_NUM", "target"]
    return [
        col
        for col in df.columns
        if col not in exclude_cols and df[col].dtype.is_numeric()
    ]


def split_data_chronologically(df: pl.DataFrame, val_weeks: int = 12):
    """Splits dataframe into train and validation sets based on WEEK_NUM."""
    max_week = df["WEEK_NUM"].max()
    df_train = df.filter(pl.col("WEEK_NUM") <= max_week - val_weeks)
    df_val = df.filter(pl.col("WEEK_NUM") > max_week - val_weeks)
    return df_train, df_val


def get_dataloaders(
    df_train: pl.DataFrame, df_val: pl.DataFrame, batch_size: int = 256
):
    """Prepares DataLoaders for train and validation using weighted random sampling."""
    feature_cols = get_feature_cols(df_train)
    df_train = prepare_raw_dataframe(df_train, feature_cols)
    df_val = prepare_raw_dataframe(df_val, feature_cols)

    # Low variance filtering: remove features with var < 1e-4
    variances = df_train.select(pl.col(feature_cols).var()).to_dicts()[0]
    low_var_cols = [c for c in feature_cols if variances[c] < 1e-4]

    if low_var_cols:
        print(f"Dropping {len(low_var_cols)} features with variance < 1e-4")
        feature_cols = [c for c in feature_cols if c not in low_var_cols]

    print(f"Features after low-variance filtering: {len(feature_cols)}")

    scaler = TorchStandardScaler(num_features=len(feature_cols))

    train_dataset = CreditRiskDataset(
        df_train, feature_cols, scaler=scaler, is_train=True
    )
    val_dataset = CreditRiskDataset(df_val, feature_cols, scaler=scaler, is_train=False)

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


def get_latent_dataloaders(train_path: str, val_path: str, batch_size: int = 256):
    """Loads precomputed latent tensors and returns DataLoaders."""
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)

    # train_dataset mapping: (Z, y, weeks)
    # y is torch.long, weeks is numpy array in generation, let's ensure consistency
    train_dataset = TensorDataset(train_data["z"], train_data["y"], train_data["weeks"])
    val_dataset = TensorDataset(val_data["z"], val_data["y"], val_data["weeks"])

    # Compute sample weights for the latent train set as well
    targets = train_data["y"]
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
