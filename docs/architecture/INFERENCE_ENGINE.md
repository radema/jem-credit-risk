# Inference Engine Specification

## 1. Overview
The JEM Inference Engine (`src/model/jem/infer.py`) is a PyTorch-based inference pipeline enforcing stable memory bounds via chunked execution.

* **Target Constraint**: Must execute under Kaggle's memory limits ($\sim 16\text{GB}$).
* **Solution**: Constant memory profile using PyTorch `DataLoader` streams.

> [!NOTE]
> The inference engine is **independent of the training data format**. Whether training used chunked Parquet partitions or a single in-memory DataFrame, inference loads the same model artifacts (`scaler.pth`, `autoencoder.pth`, `jem_model.pth`, `feature_cols.json`) and processes test data in constant-memory batches.

## 2. Component Pipeline
The forward pass runs predictably through pre-trained frozen modules (`torch.no_grad()` enabled):

1. **Scaler**: `TorchStandardScaler.transform(batch)` normalizes inputs to unit variance.
2. **Latent Map**: `TabularAutoencoder.encode(scaled_batch)` drastically reduces dimension footprint to $\mathbb{R}^{64}$.
3. **Logits**: `TabularJEM.forward(z)` acquires boundary predictions.
4. **Probability Maps**: `torch.softmax(logits, dim=1)[:, 1]` normalizes predictions into actionable probabilities for the $P(y=1|x)$ default target.

## 3. Out-of-Distribution (OOD) Diagnostics
A distinctive feature of the JEM inference pipeline is Energy reporting.

### Mathematical Definition
$$E(x) = -\text{LogSumExp}_y(f_\theta(x)[y])$$

* **Usage**: Extracted natively during `perform_inference(return_energies=True)`.
* **Value**: Real-time diagnostic evaluation of Test-Set temporal drift.
* **Alerting**: Massive deviations of Test $E(x)$ against Train $\mathbb{E}[E(x)]$ indicate the model is traversing unknown, unstable manifold space, severely impacting prediction reliability.

## 4. Operational Artifacts Required
To successfully bootstrap the pipeline, the `artifact_dir` must contain:

* `imputer_state.pkl` - Categorical frequencies and median values.
* `scaler.pth` - Global Mean and Variance maps.
* `autoencoder.pth` - Projection weights.
* `jem_model.pth` - Classifier and Generative weights.
* `feature_cols.json` - Immutable list of trained features (preserves order).
