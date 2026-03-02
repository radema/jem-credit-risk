import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.jem.config import JEMConfig
from src.model.jem.model import TabularJEM
from src.model.jem.train import evaluate_jem


def main():
    device = torch.device("cpu")
    input_dim = 100
    num_classes = 2
    batch_size = 256
    num_batches = 100

    config = JEMConfig(
        hidden_dims=[256, 256, 256],
        spectral_norm=True,
    )

    model = TabularJEM(input_dim=input_dim, num_classes=num_classes, config=config).to(
        device
    )
    model.eval()

    # Create synthetic data
    X = np.random.randn(batch_size * num_batches, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, batch_size * num_batches).astype(np.int64)
    weeks = np.random.randint(0, 10, batch_size * num_batches).astype(np.int32)

    # Simple DataLoader
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(weeks)
    )
    loader = DataLoader(dataset, batch_size=batch_size)

    # Warmup
    print("Warming up...")
    evaluate_jem(model, loader, device)

    # Benchmark
    print("Benchmarking...")
    start_time = time.time()
    for _ in range(5):  # Run 5 times to get better average
        evaluate_jem(model, loader, device)
    end_time = time.time()

    duration = (end_time - start_time) / 5
    print(f"Average time for evaluate_jem ({num_batches} batches): {duration:.4f}s")
    print(f"Average time per batch: {duration / num_batches:.4f}s")


if __name__ == "__main__":
    main()
