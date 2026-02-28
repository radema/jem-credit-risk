# System Architecture & Data Flow

## 1. Core Components

### 1.1 Data Engineering Pipeline (Polars)
* **Goal**: Process and aggregate multi-depth relational tables into a single tabular feature matrix.
* **Mechanism**: Lazy evaluation engine leveraging predicate pushdown.
* **Steps**: 
  1. Load Base table and apply stratified sampling (if training).
  2. Perform temporal aggregations (mean, max, var) on Depth 1 & 2 relational tables.
  3. Join aggregated tables onto the Base table grain (`case_id`).
  4. Handle missing values and apply Frequency Encoding tracking state via `imputer_state.pkl`.
  5. Output unscaled feature parquet.

### 1.2 Dimensionality Reduction (Autoencoder)
* **Goal**: Overcome sparsity and strict discrete constraints of tabular datasets.
* **Mechanism**: Maps input dimensional space $D$ to a continuous latent vector $Z \in \mathbb{R}^{64}$.
* **Usage**: Provides a strict, dense, and continuous manifold for the SGLD sampler to operate stably.

### 1.3 Joint Energy-Based Model (TabularJEM)
* **Goal**: Simultaneous discriminative classification $P(y|x)$ and generative modeling $P(x)$.
* **Mechanism**: Multi-layer perceptron (Default `[256, 256]`) employing Spectral Normalization to enforce Lipschitz constraints.
* **Outputs**: Unnormalized logits mapping to Normalizing Flows (Softmax) and Energy landscapes (LogSumExp).

### 1.4 Generation Engine (SGLD & Replay Buffer)
* **Goal**: Provide contrastive negative samples for EBM training.
* **Mechanism**: 
  1. Seeding from a Replay Buffer or random noise (bound by `reinit_freq=0.05`).
  2. Following energy gradients via Langevin Dynamics to find $Z_{fake}$.

---

## 2. High-Level Data Flow Diagram

```mermaid
graph TD
    subgraph Data Pipeline
        A[Raw Parquet] --> B[Polars LazyFrame]
        B --> C[Depth 1 & 2 Aggregation]
        C --> D[Left Join Base case_id]
        D --> E[Imputation & Encoding]
        E --> F[(Processed DataFrame)]
    end

    subgraph Autoencoder Projection
        F -->|TorchStandardScaler| G[Scaled Tensors]
        G --> H[TabularAutoencoder 64D]
        H --> I[(Latent Z)]
    end

    subgraph Joint Energy-Based Model
        I -->|Batch| J[TabularJEM f_theta]
        J --> K[Unnormalized Logits]
        
        K -->|Softmax| L[Classification P y=1|x]
        K -->|-LogSumExp| M[Energy E_real]
    end

    subgraph SGLD Sampler
        N[(Replay Buffer)] -->|Noise| O[z_init]
        O -->|Gradient Ascent| P[z_fake]
        P --> J
        P -->|-LogSumExp| Q[Fake Energy E_fake]
        P -->|Update| N
    end

    subgraph Loss Formulation
        M --> R((Total Loss))
        Q --> R
        L -->|Cross Entropy| R
        R -->|Adam| J
    end
```

---

## 3. Inference Orchestration
The pipeline runs inference via `scripts/generate_submission.py` in 4 explicit phases:

1. **Phase 1: Data Pipeline (INFERENCE mode)** 
   * `is_inference=True`, `sample_ratio=1.0`. 
   * Uses prefix-mapping replacing `train_` with `test_`.
   * Restores `imputer_state.pkl` without updating it.
2. **Phase 2: Load Pre-Trained Models**
   * Imports `TorchStandardScaler`, `TabularAutoencoder`, and `TabularJEM`.
   * Validates target feature dimensionality against the scaled columns.
3. **Phase 3: Batched Inference Engine**
   * Employs PyTorch `DataLoader` to map chunks into $O(batch\_size)$ constant memory.
   * Derives `P(y=1|x)` through Softmax and gathers Energy $E(x)$.
4. **Phase 4: Output & Monitoring**
   * Exports `submission.csv` containing `case_id` and normalized `score`.
   * Evaluates $E_{test}$ mean and variance to detect out-of-distribution (OOD) sets.
