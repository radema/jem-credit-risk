# Inference Engine Specification

## 1. Overview
The JEM Inference Engine is a memory-efficient PyTorch-based component designed to generate probabilistic scores from preprocessed feature matrices. It utilizes batch-oriented streaming to process millions of samples without exceeding Kaggle's memory limits.

## 2. Chunked Memory Management
To avoid Out-Of-Memory (OOM) errors, the engine uses:
* **Lazy Tensor Conversion**: The `InferenceDataset` in `src/model/jem/data_utils.py` converts only one batch of features to tensors at a time.
* **Deterministic Batching**: A standard `DataLoader` with `shuffle=False` ensures that the results can be mapped back to their original `case_id` with 100% integrity.

## 3. The Forward Subroutine
The inference loop (`perform_inference` in `src/model/jem/infer.py`) processes each batch through a fixed sequence of frozen modules:

1.  **Scaling**: `TorchStandardScaler.transform(x_batch)`.
2.  **Latent Projection**: `TabularAutoencoder.encode(x_scaled)` reduces the high-dimensional feature matrix into a dense $\mathbb{R}^{64}$ embedding.
3.  **JEM Logits**: `TabularJEM.forward(z)` outputs unnormalized class logits.
4.  **Probabilistic Normalization**: `torch.softmax(logits, dim=1)[:, 1]` extracts the probability of default ($P(y=1|x)$).

All computations are performed within a `with torch.no_grad():` block to disable gradient tracking and reduce memory overhead.

## 4. Internal Diagnostics: Out-of-Distribution (OOD) Monitoring
A key advantage of using the JEM architecture for inference is the availability of the **Energy** function $E(x)$. 

### Logic
The pipeline calculates the energy for every sample in the test set:
$$E(x) = -\text{LogSumExp}(\text{f}_\theta(x))$$

### Diagnostic Utility
While the main competition output only requires the `score` ($P(y=1|x)$), the pipeline reports $E(x)$ statistics at the end of the run. Monitoring for extreme energy spikes on the test set serves as a vital indicator of **temporal shift** or **OOD data**, alerting the Data Scientist that the model's confidence in the test manifold has diverged from training.

## 5. Deployment Orchestration
The root `scripts/generate_submission.py` combines the data pipeline with the inference engine to produce the final `submission.csv`. This script is optimized for both local development and Kaggle production environments.
