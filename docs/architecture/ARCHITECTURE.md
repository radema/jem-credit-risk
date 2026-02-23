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

    subgraph MLOps Inference Wrapper
        W1[Raw Data] --> W2[Fitted StandardScaler]
        W2 --> W3[Pre-trained Encoder]
        W3 --> W4[Pre-trained JEM Classifier]
        W4 --> W5[Risk Probabilities & Energy Shift Monitor]
    end
```
