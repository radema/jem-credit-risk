## Architectural Data Flow

```mermaid
graph TD
    subgraph Polars Data Engineering Pipeline
        A[Raw Parquet: Base, Bureau, Person] --> B[Polars LazyFrames API]
        B --> C[Temporal Aggregations max, mean, var]
        C --> D[Left Join onto Base table via case_id]
        D --> D1[Missing Value Imputation & Frequency Encoding]
        D1 --> E[StandardScaler fit on Train Fold]
        E --> F[(Processed Tensors)]
    end

    subgraph Offline Tabular Autoencoder
        F --> G1[Encoder: 347D -> 64D]
        G1 --> G2[(Latent Feature Tensors Z)]
    end

    subgraph Joint Energy-Based Model
        G2 -->|Batch z_real| G[MLP: f_theta]
        G --> H[Logits: Class 0, Class 1]
        
        H --> I{Softmax}
        I --> J[Cross-Entropy Loss]
        
        H --> K{Negative LogSumExp}
        K --> L[Real Energy: E_real]
    end

    subgraph SGLD Sampler
        M[(64D Replay Buffer)] -->|95% Buffer, 5% Noise| N[z_init]
        N -->|Gradient Ascent on E| O[z_fake]
        O --> G
        O -->|Update| M
        O --> K
        K --> P[Fake Energy: E_fake]
    end

    subgraph Loss Formulation
        L --> Q((Total Loss))
        P --> Q
        J --> Q
        Q -->|Gradient Descent| G
    end

    subgraph Phase 4: Output & Monitoring
        W3 --> W4[Probabilistic Scoring]
        W4 --> W5[submission.csv Output]
        W4 --> W6[Internal Energy OOD Monitor]
    end

    subgraph Phase 3: Batched Inference Engine
        W2 --> W3[TabularJEM Forward Pass]
    end

    subgraph Phase 2: Inference-Ready Preprocessing
        W1[Raw Data Archetypes] --> W1_1[Prefix Mapping]
        W1_1 --> W1_2[Schema Alignment]
        W1_2 --> W2[Fitted Scaler & Encoder]
    end

    subgraph Phase 1: State Management
        W0[Load imputer_state.pkl] --> W1_2
    end
```

## Inference Phase Orchestration

The JEM Inference Pipeline is executed through four distinct phases:

1.  **Phase 1: State Management**: The pipeline initializes the environment by loading precomputed transformation statistics (medians, frequency maps, training schema) from the `artifact_dir`.
2.  **Phase 2: Inference-Ready Preprocessing**: The `src.data.pipeline` executes in `is_inference=True` mode, performing automatic table prefix mapping and ensuring strict schema alignment with the training set.
3.  **Phase 3: Batched Inference Engine**: To maintain a constant memory profile ($O(batch\_size)$), the inference engine uses a PyTorch `DataLoader` to stream features through the `TorchStandardScaler`, `TabularAutoencoder`, and `TabularJEM`.
4.  **Phase 4: Output & Monitoring**: The final stage aggregates probabilistic scores, generates the competition-compliant `submission.csv`, and reports diagnostic **Energy** ($E(x)$) statistics to monitor for temporal distribution shifts.
