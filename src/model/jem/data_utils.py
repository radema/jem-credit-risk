import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    TensorDataset,
    WeightedRandomSampler,
)

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


def load_feature_cols(artifact_dir: str) -> list[str]:
    """Loads the persisted feature column list from the artifact directory."""
    feature_cols_path = Path(artifact_dir) / "feature_cols.json"
    if not feature_cols_path.exists():
        raise FileNotFoundError(
            f"Feature columns file not found at {feature_cols_path}. "
            "Ensure training has been run with the updated pipeline."
        )
    with open(feature_cols_path) as f:
        return json.load(f)


def split_data_chronologically(df: pl.DataFrame, val_weeks: int = 12):
    """Splits dataframe into train and validation sets based on WEEK_NUM."""
    max_week = df["WEEK_NUM"].max()
    df_train = df.filter(pl.col("WEEK_NUM") <= max_week - val_weeks)
    df_val = df.filter(pl.col("WEEK_NUM") > max_week - val_weeks)
    return df_train, df_val


def get_dataloaders(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    batch_size: int = 256,
    artifact_dir: str | None = None,
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

    # Persist the final feature columns if artifact_dir is provided
    if artifact_dir is not None:
        feature_cols_path = Path(artifact_dir) / "feature_cols.json"
        feature_cols_path.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_cols_path, "w") as f:
            json.dump(feature_cols, f)
        print(f"Saved feature_cols to {feature_cols_path}")

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


class InferenceDataset(Dataset):
    """
    Dataset for inference that yields (case_id, features).
    """

    def __init__(self, df: pl.DataFrame, feature_cols: list[str]):
        self.case_ids = df["case_id"].to_numpy()
        self.features = df.select(feature_cols).to_numpy()

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        # We wrap in torch.tensor here row-by-row to save overall memory
        # compared to pre-converting the entire dataframe to a single giant tensor.
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        case_id = self.case_ids[idx]
        return case_id, x


def get_inference_dataloader(
    df: pl.DataFrame, feature_cols: list[str], batch_size: int = 4096
):
    """Returns a simple DataLoader yielding (case_id, x_batch) for inference."""
    dataset = InferenceDataset(df, feature_cols)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class _ShuffleBuffer:
    """Fixed-size buffer that yields random samples when full."""

    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = capacity
        self.buffer = []
        self.rng = random.Random(seed)

    def add(self, item: Any):
        self.buffer.append(item)

    def is_full(self) -> bool:
        return len(self.buffer) >= self.capacity

    def pop_random(self) -> Any:
        if not self.buffer:
            raise IndexError("pop from empty buffer")
        idx = self.rng.randint(0, len(self.buffer) - 1)
        # O(1) removal by swapping with last element
        self.buffer[idx], self.buffer[-1] = self.buffer[-1], self.buffer[idx]
        return self.buffer.pop()

    def drain(self) -> Iterator[Any]:
        self.rng.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class ChunkedParquetDataset(IterableDataset):
    """
    Memory-bounded IterableDataset that reads Parquet chunk files
    from disk, applies optional scaling, and uses a shuffle buffer
    for pseudo-random sample ordering.
    """

    def __init__(
        self,
        chunk_paths: list[Path],
        feature_cols: list[str],
        scaler: TorchStandardScaler | None = None,
        shuffle_buffer_size: int = 50_000,
        shuffle_chunks: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.chunk_paths = [Path(p) for p in chunk_paths]
        self.feature_cols = feature_cols
        self.scaler = scaler
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number for deterministic chunk shuffling."""
        self._epoch = epoch

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Yields (x, y, week_num, sample_weight) tuples.
        """
        worker_info = torch.utils.data.get_worker_info()

        # Determine chunks for this worker
        chunks = self.chunk_paths.copy()
        if self.shuffle_chunks:
            # Deterministic shuffle per epoch
            rng = random.Random(
                self.seed + self._epoch if self.seed is not None else None
            )
            rng.shuffle(chunks)

        if worker_info is not None:
            # Partition chunks among workers
            per_worker = int(np.ceil(len(chunks) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(chunks))
            chunks = chunks[iter_start:iter_end]

        buffer = _ShuffleBuffer(
            self.shuffle_buffer_size,
            seed=self.seed + self._epoch if self.seed is not None else None,
        )

        for path in chunks:
            # Read chunk
            df = pl.read_parquet(path)

            # Extract components
            x_np = df.select(self.feature_cols).to_numpy()
            y_np = df["target"].to_numpy()
            w_np = df["WEEK_NUM"].to_numpy()

            x_t = torch.from_numpy(x_np.copy()).to(torch.float32)
            y_t = torch.from_numpy(y_np.copy()).to(torch.long)
            w_t = torch.from_numpy(w_np.copy()).to(torch.long)

            # Apply scaling
            if self.scaler is not None:
                x_t = self.scaler.transform(x_t)

            # Compute normalized per-sample class weights for this chunk
            # Uses n / (n_classes * count) so mean weight ≈ 1.0
            n = len(y_t)
            n_classes = 2
            class_counts = (
                torch.bincount(y_t, minlength=n_classes).float().clamp(min=1.0)
            )
            class_weights = n / (n_classes * class_counts)
            sample_weights = class_weights[y_t]

            # Zip and add to shuffle buffer
            for i in range(len(y_t)):
                item = (x_t[i], y_t[i], w_t[i], sample_weights[i])
                buffer.add(item)

                if buffer.is_full():
                    yield buffer.pop_random()

        # Drain remaining items
        yield from buffer.drain()


class ChunkedLatentDataset(IterableDataset):
    """
    Memory-bounded IterableDataset that reads .pt latent chunk files sequentially
    with shuffle buffer and chunk-level class weighting.
    """

    def __init__(
        self,
        chunk_paths: list[Path],
        shuffle_buffer_size: int = 50_000,
        shuffle_chunks: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.chunk_paths = [Path(p) for p in chunk_paths]
        self.shuffle_buffer_size = shuffle_buffer_size
        self.shuffle_chunks = shuffle_chunks
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch number for deterministic chunk shuffling."""
        self._epoch = epoch

    def __iter__(
        self,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Yields (z, y, week_num, sample_weight) tuples from .pt files.
        """
        worker_info = torch.utils.data.get_worker_info()

        # Determine chunks for this worker
        chunks = self.chunk_paths.copy()
        if self.shuffle_chunks:
            # Deterministic shuffle per epoch
            rng = random.Random(
                self.seed + self._epoch if self.seed is not None else None
            )
            rng.shuffle(chunks)

        if worker_info is not None:
            # Partition chunks among workers
            per_worker = int(np.ceil(len(chunks) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(chunks))
            chunks = chunks[iter_start:iter_end]

        buffer = _ShuffleBuffer(
            self.shuffle_buffer_size,
            seed=self.seed + self._epoch if self.seed is not None else None,
        )

        for path in chunks:
            # Load latent chunk
            data = torch.load(path, weights_only=False, map_location="cpu")
            z_t = data["z"]
            y_t = data["y"]
            w_t = data["weeks"]

            # Convert weeks to tensor if it's numpy from old generation scripts
            if isinstance(w_t, np.ndarray):
                w_t = torch.from_numpy(w_t).to(torch.long)

            # Compute normalized per-sample class weights for this chunk
            # Uses n / (n_classes * count) so mean weight ≈ 1.0
            n = len(y_t)
            n_classes = 2
            class_counts = (
                torch.bincount(y_t, minlength=n_classes).float().clamp(min=1.0)
            )
            class_weights = n / (n_classes * class_counts)
            sample_weights = class_weights[y_t]

            # Zip and add to shuffle buffer
            for i in range(len(y_t)):
                item = (z_t[i], y_t[i], w_t[i], sample_weights[i])
                buffer.add(item)

                if buffer.is_full():
                    yield buffer.pop_random()

        # Drain remaining items
        yield from buffer.drain()
