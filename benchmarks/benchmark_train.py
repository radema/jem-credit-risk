import time

import numpy as np
import polars as pl
import torch

from src.model.jem.config import JEMConfig
from src.model.jem.data_utils import get_dataloaders
from src.model.jem.loss import JEMLoss
from src.model.jem.model import TabularJEM
from src.model.jem.sampler import SGLDReplayBuffer, SGLDSampler
from src.model.jem.train import train_jem_epoch

"""
Benchmark script for the Joint Energy-Based Model (JEM) training loop.

Scope:
This script establishes a performance baseline for the `train_jem_epoch` function.
It measures the time taken to process a single epoch of training using synthetic data.
This is used to quantify the impact of performance optimizations in the training loop.

Usage:
    PYTHONPATH=. uv run benchmarks/benchmark_train.py

The script will:
1. Initialize a TabularJEM model and its associated components (loss, optimizer, sampler, buffer).
2. Create a synthetic dataset of 100 batches.
3. Perform a warmup epoch.
4. Measure and print the total time and average time per batch for a second epoch.
"""


def main():
    device = torch.device("cpu")
    input_dim = 100
    num_classes = 2
    batch_size = 256
    num_batches = 100

    config = JEMConfig(
        hidden_dims=[256, 256, 256],
        spectral_norm=True,
        sgld_steps=5,
        sgld_step_size=0.1,
        sgld_sigma=0.01,
        l2_energy_weight=0.1,
    )

    model = TabularJEM(input_dim=input_dim, num_classes=num_classes, config=config).to(
        device
    )
    criterion = JEMLoss(config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    buffer = SGLDReplayBuffer(buffer_size=10000, feature_dim=input_dim)
    sampler = SGLDSampler(config=config)

    # Create synthetic data
    df_train = pl.DataFrame(
        {
            "case_id": np.arange(batch_size * num_batches),
            "MONTH": np.zeros(batch_size * num_batches),
            "WEEK_NUM": np.random.randint(0, 10, batch_size * num_batches),
            "target": np.random.randint(0, 2, batch_size * num_batches),
            **{
                f"feat_{i}": np.random.randn(batch_size * num_batches)
                for i in range(input_dim)
            },
        }
    )
    df_val = df_train.clone()

    train_loader, val_loader, scaler, feature_cols = get_dataloaders(
        df_train, df_val, batch_size=batch_size
    )

    # Warmup
    print("Warming up...")
    train_jem_epoch(model, train_loader, optimizer, criterion, buffer, sampler, device)

    # Benchmark
    print("Benchmarking...")
    start_time = time.time()
    train_jem_epoch(model, train_loader, optimizer, criterion, buffer, sampler, device)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Total time for 1 epoch ({num_batches} batches): {duration:.4f}s")
    print(f"Average time per batch: {duration / num_batches:.4f}s")


if __name__ == "__main__":
    main()
