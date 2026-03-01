# Data Pipeline Specification

## 1. Overview
The JEM Credit Risk Data Pipeline is built on Polars' LazyFrame API to enable memory-efficient processing of massive tabular datasets ($>10$M rows). It orchestrates multiple depth levels ($0, 1, 2$) and archetypes into a consolidated feature matrix.

## 2. Dynamic Archetype Loading
The pipeline identifies tables based on their competition archetypes:
* **Depth 0**: Static tables (e.g., `train_static_0`). No aggregation required before join.
* **Depth 1**: Historical tables (e.g., `train_person_1`). Requires aggregation (mean, max, var) to the `case_id` grain.
* **Depth 2**: Nested history (e.g., `train_person_2`). Requires double-aggregation: first to Depth 1 grain (`case_id` + `num_group1`), then joined to Depth 1 before the final aggregation to Depth 0 grain.

## 3. Inference Mode (`is_inference=True`)
When the `is_inference` flag is set in the `DataPipelineConfig`, the following adaptations occur:

### A. Prefix Mapping
The pipeline automatically swaps the `train_` prefix for `test_` during table scanning. This allows the same orchestration code to process competition test chunks seamlessly.

### B. Target Bypassing
In inference mode, the pipeline skips:
* Loading of the `target` column from the base table.
* Stratified Sampling (which requires target availability).
* Any logic that attempts to calculate statistics based on target labels.

### C. State-Stored Imputation
Instead of fitting new medians or frequency maps, the pipeline loads `imputer_state.pkl` from the specified `artifact_dir`. This ensures that the test data is projected onto the *exact same distribution* as the training data.

### D. Robust Schema Alignment
Test data chunks may differ in schema from the training set (e.g., missing features that were entirely null in a specific chunk). The pipeline enforces consistency via:
1. **Feature Injection**: Any feature present in training but missing in test is injected and filled with the training-time median.
2. **Feature Pruning**: Any "leakage" or extra features present in test but not in training are dropped.
3. **Deterministic Ordering**: Columns are strictly reordered to match the training manifold before being converted to tensors for the JEM model.

## 4. Execution Examples

### Training Mode (Chunked — Default)
```python
cfg = DataPipelineConfig(sample_ratio=0.05, is_inference=False, chunked_export=True, chunk_size=200_000)
run_pipeline(cfg, ...) # Saves imputer_state.pkl and data/processed/chunks/train_chunk_*.parquet
```

### Training Mode (Legacy In-Memory)
```python
cfg = DataPipelineConfig(sample_ratio=0.05, is_inference=False, chunked_export=False)
run_pipeline(cfg, ...) # Saves imputer_state.pkl and train_features_unscaled.parquet
```

### Inference Mode
```python
cfg = DataPipelineConfig(sample_ratio=1.0, is_inference=True, artifact_dir="models/artifacts")
run_pipeline(cfg, ...) # Loads imputer_state.pkl and saves test_features_unscaled.parquet
```

---

## 5. Chunked Export Mode

When `DataPipelineConfig.chunked_export=True` (the default), the pipeline writes processed training data as numbered Parquet partitions instead of a single file.

### Configuration
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunked_export` | `bool` | `True` | Enable partitioned export |
| `chunk_size` | `int` | `200,000` | Maximum rows per partition file |

### Output Layout
```
data/processed/chunks/
├── train_chunk_001.parquet    # ≤200k rows
├── train_chunk_002.parquet
└── train_chunk_NNN.parquet
```

### Downstream Consumers
1. **`TorchStandardScaler.streaming_fit()`** — Reads chunks sequentially to compute global mean/variance via Welford's online algorithm. Memory: O(num_features).
2. **`ChunkedParquetDataset`** — PyTorch `IterableDataset` that reads one chunk at a time, applies the fitted scaler, and uses a shuffle buffer (default 50k rows) for pseudo-random ordering.
3. **`generate_latent_chunks()`** — After AE training, encodes each chunk independently and saves `latent_chunk_XXX.pt` files to `data/processed/latent_chunks/`.
4. **`ChunkedLatentDataset`** — Streams `.pt` latent files for JEM training.

> [!IMPORTANT]
> The chunked export path is for **training only**. Inference continues to use the existing single-file pipeline.
