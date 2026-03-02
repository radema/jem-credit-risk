# Joint Energy-Based Model (JEM) for Credit Risk Stability

![Python](https://img.shields.io/badge/python-v3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-FFD43B?style=flat-square&logo=polars&logoColor=black)
![Framework](https://img.shields.io/badge/Framework-JEM-7B61FF?style=flat-square)
![SGLD](https://img.shields.io/badge/Sampling-SGLD-00F2FF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)


## Project Overview

This repository contains the implementation of a **Joint Energy-Based Model (JEM)** applied to the [Home Credit - Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability) Kaggle competition.

The project moves beyond standard discriminative classification by training a neural network that simultaneously learns the conditional probability of default $p(y|x)$ and the marginal data distribution $p(x)$. By learning the energy landscape of the feature space, the model provides a built-in mechanism to detect temporal distribution shifts (Out-Of-Distribution data) via scalar energy spikes.

### 🎯 Key Objectives

*   **Primary Goal**: Implement a JEM to classify default risk while simultaneously learning the marginal distribution of normal client profiles.
*   **Data Engineering**: Design a declarative, memory-efficient data pipeline using **Polars** (LazyFrame API) to aggregate deep relational tables (Bureau, Person, etc.) into a flat feature matrix without Out-Of-Memory (OOM) exceptions.
*   **Hierarchical Latent Space Sampling**: Pre-train an Offline Tabular Autoencoder to map the discrete and sparse tabular features into a dense, continuous latent space $\mathbb{R}^{64}$. The JEM and its SGLD sampler operate safely over this latent continuous manifold to prevent sampling collapse.
*   **MLOps & Monitoring**: Combine data scalers, the pretrained autoencoder, and the JEM classifier into a cohesive inference wrapper to avoid training-serving skew. Utilize the JEM's energy output to track distribution shifts and visualize temporal decay across the `MONTH` dimension.

### 🏗️ Architecture & Deployment

For a detailed technical breakdown of the mathematical framework, data engineering pipeline, and SGLD sampling strategy, please refer to the **[ARCHITECTURE.md](./docs/architecture/ARCHITECTURE.md)** document.

For instructions on how to package and submit this modular codebase to the Kaggle competition, see the **[KAGGLE_DEPLOYMENT.md](./docs/architecture/KAGGLE_DEPLOYMENT.md)** guide.


### 🛠️ Development Constraints

*   **Technology Stack**: Python (PyTorch for modeling, Polars for data engineering).
*   **Design Principles**: Adhere strictly to SOLID/DRY principles. Code must be highly modular and type-hinted.
*   **Environment**: Developed locally via **Antigravity**, ensuring seamless execution as a Kaggle script.


## Technology Stack

- Language: Python 3.10+
- Data Engineering: polars (LazyFrame API for out-of-core relational joins and aggregations)
- Machine Learning: torch (PyTorch for MLP architecture, custom loss functions, and Langevin Dynamics)
- Metrics: Area Under the Precision-Recall Curve (AUPRC), Gini Stability, Energy Histograms.

## Mathematical Context

This project strictly follows the JEM framework (Grathwohl et al., 2019). The architecture reuses standard classifier logits to compute both classification and generative objectives.

Model Output: A standard MLP outputs logits $f_\theta(x)[y]$.

Discriminative Phase: Supervised classification via standard Softmax cross-entropy.


$$p_\theta(y|x) = \frac{\exp(f_\theta(x)[y])}{\sum_{y'} \exp(f_\theta(x)[y'])}$$

Generative Phase: The unnormalized energy of a data point is the negative LogSumExp of its logits.


$$E_\theta(x) = -\log \sum_y \exp(f_\theta(x)[y])$$

Sampling: "Fake" data points are generated to compute Contrastive Divergence using Stochastic Gradient Langevin Dynamics (SGLD), maintaining a 95% / 5% Replay Buffer to ensure stable Markov Chain mixing.

## Expected Directory Structure

```
.
├── data/
│   ├── raw/                 # Kaggle parquet files and csv dictionaries
│   └── processed/           # Processed artifacts
│       ├── chunks/          # Chunked Parquet partitions (default training export)
│       ├── latent_chunks/   # Encoded latent .pt chunk files
│       ├── scaler.pt        # TorchStandardScaler state
│       ├── encoder.pt       # Autoencoder encoder weights
│       └── feature_cols.json
├── docs/
│   ├── architecture/        # ARCHITECTURE.md and THEORY.md
│   ├── data/                # data_dictionary.md, PIPELINE.md
│   └── planning/            # ROADMAP.md
├── src/
│   ├── data                 # Polars feature engineering, export (chunked + single-file)
│   ├── model/autoencoder    # Pre-training tabular autoencoder for latent space
│   └── model/jem            # PyTorch TabularJEM, SGLDSampler, data utils, training loops
├── tests/                   # Unit and integration tests (incl. chunked pipeline E2E)
├── pyproject.toml
└── README.md
```


## Setup & Quickstart

Install Dependencies:

```bash
uv sync
```


### Code Quality & Development

To ensure high code quality, this project uses `ruff` for linting and formatting, and `pre-commit` for automated checks.

**Install pre-commit hooks:**

```bash
uv run pre-commit install
```

**Run hooks manually on all files:**

```bash
uv run pre-commit run --all-files
```

## Data Preparation

Download the Home Credit competition data and place the parquet files in data/raw/.

## Core Modules

The codebase is split into three core modules. This separation of concerns ensures the complex data engineering does not pollute the mathematical modeling.

**src/data/** (Module): Utilizes `polars.LazyFrame` for out-of-core feature engineering. Implements a clean, chainable pipeline (`pipeline.py`) for aggregating nested historical data, executing robust missing value imputation and frequency encoding for categoricals (`imputation.py`), and exporting finalized datasets.

**src/model/autoencoder/**: Pre-trains a tabular autoencoder to map discrete variables into a dense, mathematically continuous $\mathbb{R}^{64}$ latent space required for stable Langevin sampling.

**src/model/jem/**: Contains the TabularJEM class and the SGLDSampler class (handles the MCMC sampling and Replay Buffer logic). Exports the unified `LatentJEMWrapper`.

## Development Workflow

Step 1: Execute `notebooks/JEM_Execution_Pipeline.ipynb` for the full data and training workflow, OR run the automated scripts:

Step 2: Run `src/model/autoencoder/train_autoencoder.py` to map features to the latent space mapping tensors `Z`. When chunked data is detected in `data/processed/chunks/`, this script automatically uses `streaming_fit()` and `ChunkedParquetDataset` for memory-bounded training. Otherwise, falls back to in-memory mode.

Step 3: Run `src/model/jem/train_latent.py` to initiate the dual-objective training loop on the `Z` embeddings. When `data/processed/latent_chunks/` exists, it uses `ChunkedLatentDataset` for streaming. Serializes the final LatentJEMWrapper artifact.

## Design Philosophy

- SOLID/DRY Principles: Code must be modular. Feature engineering logic must not bleed into the PyTorch modeling classes.
- No Pandas: Due to the relational depth of the dataset, pandas will cause OOM errors. All data processing must use polars.LazyFrame.
- Standardization: EBMs and SGLD are highly sensitive to the scale of the input manifold. Strict StandardScaler application (fit purely on the train set) is mandatory.

## References

- Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model. ICLR. - * Defines the joint optimization of $p(y|x)$ and $p(x)$. *
- Du, Y., & Mordatch, I. (2019). Implicit Generation and Generalization in Energy-Based Models. NeurIPS. - *Essential reading for implementing the Replay Buffer correctly.*
- Network Geometry: Spontaneous Kolmogorov-Arnold Geometry in Shallow MLPs (Freedman & Mulligan, arXiv:2509.12326). - *Understanding that your MLP's energy landscape spontaneously forms ridged, Kolmogorov-Arnold-like geometries will assist in debugging SGLD chain stagnation.*
