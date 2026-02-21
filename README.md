# Joint Energy-Based Model (JEM) for Credit Risk Stability

![Python](https://img.shields.io/badge/python-v3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Polars](https://img.shields.io/badge/Polars-FFD43B?style=flat-square&logo=polars&logoColor=black)
![Framework](https://img.shields.io/badge/Framework-JEM-7B61FF?style=flat-square)
![SGLD](https://img.shields.io/badge/Sampling-SGLD-00F2FF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)


## Project Overview

This repository contains the implementation of a Joint Energy-Based Model (JEM) applied to the Home Credit - Credit Risk Model Stability (Kaggle 2024) dataset.

The objective is to move beyond standard discriminative classification by training a neural network that simultaneously learns the conditional probability of default $p(y|x)$ and the marginal data distribution $p(x)$. By learning the energy landscape of the feature space, the model provides a built-in mechanism to detect temporal distribution shifts (Out-Of-Distribution data) via scalar energy spikes.

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
│   ├── raw/                 # Put downloaded Kaggle parquet files here
│   └── processed/           # Processed torch tensors will be saved here
├── src/
│   ├── data.py              # Polars feature engineering and temporal aggregations
│   ├── jem.py               # PyTorch TabularJEM architecture and SGLDSampler
│   └── train.py             # Training loop, loss formulation, and metric logging
├── pyproject.toml
└── README.md
```

## Setup & Quickstart

Install Dependencies:

```bash
pip install -r requirements.txt
```

(Ensure you have polars, torch, scikit-learn, numpy, and matplotlib installed).

## Data Preparation

Download the Home Credit competition data and place the parquet files in data/raw/.

## Development Workflow

Step 1: Run src/data.py to ingest the relational tables, aggregate features using Polars, and export scaled tensors.

Step 2: Run src/jem.py as a standalone script to execute unit tests verifying the SGLD backward passes.

Step 3: Run src/train.py to initiate the dual-objective training loop.

## Design Philosophy

- SOLID/DRY Principles: Code must be modular. Feature engineering logic must not bleed into the PyTorch modeling classes.
- No Pandas: Due to the relational depth of the dataset, pandas will cause OOM errors. All data processing must use polars.LazyFrame.
- Standardization: EBMs and SGLD are highly sensitive to the scale of the input manifold. Strict StandardScaler application (fit purely on the train set) is mandatory.

## References

- Grathwohl, W., et al. (2020). Your Classifier is Secretly an Energy Based Model. ICLR. - * Defines the joint optimization of $p(y|x)$ and $p(x)$. *
- Du, Y., & Mordatch, I. (2019). Implicit Generation and Generalization in Energy-Based Models. NeurIPS. - *Essential reading for implementing the Replay Buffer correctly.*
- Network Geometry: Spontaneous Kolmogorov-Arnold Geometry in Shallow MLPs (Freedman & Mulligan, arXiv:2509.12326). - *Understanding that your MLP's energy landscape spontaneously forms ridged, Kolmogorov-Arnold-like geometries will assist in debugging SGLD chain stagnation.*