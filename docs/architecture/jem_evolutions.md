# JEM (Joint Energy-Based Model) Evolution Proposals

## 1. Problem Statement: The Tabular Data Manifold

During the initial experimental phases with the JEM architecture on the Home Credit Default Risk dataset, a fundamental issue with **Stochastic Gradient Langevin Dynamics (SGLD)** sampling was identified.

When projecting the replay buffer state via PCA (`docs/assets/pca_replay_buffer.png`), we observed **manifold collapse and fracturing**. The SGLD sampler, meant to traverse a continuous probability distribution, occasionally generated massive outliers or settled in remote "black hole" clusters completely disconnected from the true data (`x_real`).

### The Root Cause
Standard SGLD is mathematically derived for **dense, continuous domains** (like images in the pixel space `[-1, 1]`). Tabular data (and specifically the Home Credit dataset) heavily violates this assumption because it is dominated by:
1. **Sparsity**: Many missing values imputed or mapped to specific constants.
2. **Strict Categoricals**: One-Hot Encoded vectors and Ordinals where intermediate continuous values (e.g., a one-hot column being `0.45` rather than `0` or `1`) are semantically meaningless.

When Gaussian noise is injected into these strict discrete constraints during the MCMC steps, the sampler steps "off the empirical manifold" into mathematically unbounded regions, causing energies to spiral or fracture into disconnected blobs, crippling the model's discriminative calibration.

---

## 2. Proposed Evolutions

To fully solve this issue for deep tabular EBMs, several architectural and algorithmic evolutions can be implemented beyond simple hyperparameter tuning (like lowering `sgld_steps` or increasing `reinit_freq`).

### Evolution A: Categorical MCMC (Discrete Energy-Based Models)
Instead of forcing continuous Langevin Dynamics to operate on discrete variables, we must respect the data types during the negative phase sampling.
* **Mechanism**: Separate the input features into `continuous` and `categorical` masks.
* **Continuous Features**: Continue using Langevin Dynamics (SGLD) with gradient descent.
* **Categorical Features**: Use **Gibbs Sampling** or a **Metropolis-Hastings discrete walk**. Instead of adding continuous gradients, we periodically flip categorical bits and compute if $E_{new} < E_{old}$.
* *Reference*: "Your Classifier is Secretly an Energy Based Model" (Grathwohl et al.) - extensions into discrete spaces. Also, approaches utilizing *Gumbel-Softmax* to create a differentiable relaxation of categoricals during the SGLD pass.

### Evolution B: Hierarchical Latent Space Sampling
Running SGLD in the raw input space of 300+ sparse tabular columns is unstable. We can move the SGLD process to a simpler, strictly continuous latent space.
* **Mechanism**:
  1. Pre-train a simple tabular Autoencoder (or utilize an embedding layer) mapping the 300+ mixed features to a dense, continuous latent vector $z \in \mathbb{R}^{64}$.
  2. Implement the JEM classifier *on top of* this latent space $z$.
  3. The SGLD sampler will now generate fake samples in the latent space $z_{fake}$, where the manifold is continuous, dense, and naturally bounds Gaussian noise injection.
* **Benefits**: Instantly solves the categorical constraint problem since the latent space is continuous, leading to much faster MCMC convergence and no "manifold falling".

### Evolution C: Hybrid Contrastive Divergence (CD-k) with High Re-Anchoring
A purely algorithmic mitigation that does not require architectural rewrites.
* **Mechanism**: If SGLD gets lost when taking too many steps in the tabular void, limit its freedom.
  1. Reduce `sgld_steps` dramatically (e.g., from 40 down to 5 or 10).
  2. Set `reinit_freq = 0.5` or higher in the Replay Buffer.
  3. **Data Anchoring**: Instead of seeding the replay buffer entirely from uniform noise, seed the MCMC chains by adding a small epsilon of noise *directly to the real training batch* ($x_{init} = x_{real} + \epsilon$).
* **Benefits**: This forces the fake samples to remain intimately close to the true data manifold, preventing them from wandering into extreme outlier states, at the cost of strictly exploring local modes.

---

## 3. Next Steps
For immediate stabilization while continuing Phase 1/2 of the project, **Evolution C** is the recommended fast-path. By shortening the chain lengths and re-anchoring to real data, the JEM can stabilize its classification accuracy. For a true, robust, State-of-the-Art implementation (Phase 4), **Evolution A (Categorical Masking)** or **Evolution B (Latent Space JEM)** should be prioritized.

### Evolution D: Multi-Modal / Multi-Tower Architecture
Instead of relying on Polars to arbitrarily flatten sequential data (depth=1, depth=2) through mean/max aggregations, the JEM can natively model the temporal structures by processing raw sequential data through dedicated neural towers.
* **Mechanism**:
  1. **Data Pipeline**: The DataLoader is rewritten to output structured dictionaries per batch, e.g., `{'static': tensor(batch, static_dim), 'history_d1': tensor(batch, seq_len, d1_dim)}`.
  2. **Static Tower**: Depth=0 tabular features pass through a standard MLP with Spectral Normalization, generating an embedding $H_0$.
  3. **Sequence Tower**: Depth=1 historical features are padded and passed through a powerful sequential encoder (1D-CNN, GRU, or Transformer). Output is pooled (e.g., Attention Pooling) to yield embedding $H_1$.
  4. **Fusion & JEM Head**: The representations are concatenated $[H_0, H_1]$ and passed through final Linear layers to generate the unnormalized logits used for both $P(y|x)$ (Softmax) and $E(x)$ (LogSumExp).
* **Impact on SGLD**:
  - To generate samples, the MCMC runs on the concatenated raw feature space (difficult, as you must define noise for varying sequence lengths) OR
  - **Crucially**, the SGLD runs *only on the concatenated latent embeddings* $[H_0, H_1]$, bringing us back to the benefits of Evolution B, but with vastly richer temporal representations.
* **Trade-offs**:
  - **Pros**: Completely eliminates lossy feature engineering for historical tables. The energy landscape inherently models temporal behavioral shifts (e.g., recognizing that defaults correlate with a *recent spike* in credit bureau queries, rather than just a high *average*).
  - **Cons**: Exponentially higher computational cost. RNNs/Transformers on millions of credit applications are slow. Also requires complex engineering around `collate_fn` to handle sequence padding efficiently without blowing up GPU memory.
