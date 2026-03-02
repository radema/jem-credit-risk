import torch
import time
import numpy as np
from src.model.jem.model import TabularJEM
from src.model.jem.config import JEMConfig

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Large model and batch to make it measurable
    config = JEMConfig(hidden_dims=[2048, 2048, 2048], spectral_norm=True)
    model = TabularJEM(input_dim=1000, num_classes=2, config=config).to(device)
    x = torch.randn(4096, 1000).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Case 1: Redundant forward pass
    start = time.time()
    for _ in range(50):
        logits = model(x)
        energy = model.compute_energy(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    redundant_time = end - start
    print(f"Redundant forward pass: {redundant_time:.4f}s")

    # Case 2: Optimized
    start = time.time()
    for _ in range(50):
        logits = model(x)
        energy = -torch.logsumexp(logits, dim=1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    optimized_time = end - start
    print(f"Optimized forward pass: {optimized_time:.4f}s")

    improvement = (redundant_time - optimized_time) / redundant_time * 100
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    benchmark()
