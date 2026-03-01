# High-Level Implementation Strategy

This document outlines the phased, engineering-focused approach to implementing the Joint Energy-Based Model (JEM) for the Home Credit Default Risk dataset.

## Phase 1: Local Data Engineering (Antigravity)

**Goal:** Build a robust, memory-efficient pipeline capable of handling deep relational data without Out-Of-Memory (OOM) errors.

**Action 1:** Stratified Sampling. Do not start with the full 1M+ row dataset. Extract a 5% stratified sample locally to ensure rapid iteration.

**Action 2:** Declarative Aggregations. Use polars.LazyFrame to build the join logic across the base, person, and bureau tables. Focus on temporal aggregations (e.g., max, mean, var over historical applications).

**Action 3:** Missing Value Imputation & Categorical Encoding. For categorical variables, perform frequency encoding and treat "MISSING" as a valid category. For continuous variables, perform median imputation strictly alongside binary missing indicator columns. Save the aggregated, unscaled data to parquet. Standard scaling is purposely postponed (Out of Scope for this phase) to avoid data leakage before Cross Validation.

**Output:** Joined, aggregated, imputed, and unscaled datasets saved locally.

## Phase 2: Local Model Implementation (PyTorch)

**Goal:** Mathematically implement the JEM architecture and the SGLD sampler, verifying gradient flows.

**Action 1:** JEM Architecture. Build the TabularJEM MLP. Implement the forward() method for logits and the compute_energy() method using torch.logsumexp.

**Action 2:** The SGLD Sampler. Implement the Stochastic Gradient Langevin Dynamics loop. Crucially, initialize a Replay Buffer. During training, sample 95% of initial states from this buffer and 5% from uniform noise to ensure stable Markov Chain mixing.

**Action 3:** Unit Testing. Write a rigid unit test: generate a batch of random noise, pass it through the JEM, run 10 steps of SGLD, and assert that the backward pass successfully computes gradients for both the classification loss and the contrastive divergence loss without throwing NaNs.

**Output:** Tested jem.py module ready for integration.

## Phase 3: Integration & Training Loop

**Goal:** Combine the data pipeline and model into a functional dual-objective training loop.

**Action 1:** Data Preparation & Normalization. Implement the standard scaling fit strictly on the training fold (postponed from Phase 1 to prevent data leakage). Then formulate the dual objective loss: $\mathcal{L} = \mathcal{L}_{clf} + \lambda \mathcal{L}_{energy}$. Tune $\lambda$ (the generative weight) carefully so the generative task does not overpower the discriminative task.

**Action 2:** Metric Tracking. Track Area Under the Precision-Recall Curve (AUPRC) and Gini Stability for the supervised task. Simultaneously, track the mean energies of real vs. fake samples.

**Action 3:** Stability Checks. If fake energy collapses or shoots to infinity, adjust the SGLD step size or Replay Buffer sampling ratio.

## Phase 4: Kaggle Scaling & OOD Evaluation

**Goal:** Run the full pipeline on Kaggle GPUs and validate the Out-Of-Distribution (OOD) detection capabilities.

**Action 1:** Porting. Upload data.py, jem.py, and train.py to a Kaggle Notebook. Attach the full Home Credit dataset.

**Action 2:** GPU Training. Execute the training loop on the full dataset, utilizing the GPU purely for the heavy lifting of the model training (Polars will handle the CPU-bound data prep efficiently).

**Action 3:** OOD Verification. Evaluate model stability over time. Plot the average batch energy $E(x)$ over the MONTH column. If the data distribution shifts significantly in later months (a core challenge of this competition), the scalar energy should spike, successfully flagging the OOD data.
