# JEM Tabular Evolutions & Resolutions

## 1. Context: Space Violation in Tabular SGLD
Stochastic Gradient Langevin Dynamics (SGLD) inherently demands a **smooth, continuous, and universally dense space** to operate reliably. The Home Credit standard dataset blatantly violates these laws by exhibiting:

* **Massive Sparsity**: Encoded structural voids.
* **Categorical Rigidity**: Strict variables containing semantically meaningless non-integer derivatives (e.g. state `0.54`).

**Failure State**: The SGLD traversal steps outside structural data boundaries. Resulting gradients generate uncontrolled energies, scattering metrics, and fracturing manifold topologies (observable as "black hole collapses" under PCA).

## 2. Executed & Proposed Evolutions

### 2.1 Evolution B: Latent Space Normalization (Implemented)
We resolve the categorical anomaly by remapping the tabular space onto an intrinsically smooth, deep neural volume.
* **Mechanism**: Deploy a `TabularAutoencoder` producing continuous encoding vector $z \in \mathbb{R}^{64}$.
* **Action**: SGLD limits execution purely within $Z$-space, eliminating arbitrary category boundary violations.
* **Result**: Accelerates chain convergence drastically and prevents topological collapse.

### 2.2 Evolution C: Hybrid Contrastive Seeding (Current Mitigation Backup)
A purely parameters-based heuristic preventing chain escapes.
* **Mechanism**: 
  1. Constrain SGLD chain length (`sgld_steps < 10`).
  2. Increase buffer reset probability (`reinit_freq > 0.5`).
  3. Form initialization states by attaching a minor noise epsilon strictly onto known real training points.
* **Result**: Restricts generated fakes directly against the primary manifold density.

### 2.3 Evolution A: Discrete Masking via Gibbs Sampling (Future Target)
Treat the data array selectively, acknowledging spatial variations.
* **Mechanism**:
  * Implement continuous SGLD *only* over bounded real values.
  * Utilize Metropolis-Hastings/Gibbs walks mapping valid categorical nodes, avoiding invalid continuous space injection.

### 2.4 Evolution D: Sequential Modality Engine (Future Target)
Replace generic mean/var aggregations within Polars with Neural sequence extraction.
* **Mechanism**:
  1. Forward structured depths natively into 1D-CNN or transformer blocks.
  2. Map pooled final encodings against the 0-depth MLPs.
  3. Execute SGLD locally on the unified encoded vectors.
* **Result**: Maximizes extraction of temporal patterns in credit activity at the cost of exponentially heavy architectural processing requirements.
