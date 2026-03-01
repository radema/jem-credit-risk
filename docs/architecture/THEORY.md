# The Theory of Joint Energy-Based Models (JEM)

## 1. Energy-Based Models (EBM)
EBMs define probability density via an unnormalized Boltzmann distribution constraint:
$$p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}$$

* **$E_\theta(x)$**: Maps inputs to scalar energy values (Real = Low Energy, Fake = High Energy).
* **$Z(\theta)$**: Intractable spatial integral over data domains restricting deterministic likelihood calculations.
* **Optimization**: Pursues contrastive divergence. Drops energy of observed data whilst deliberately lifting energy of generated samples.

## 2. JEM Reinterpretation
JEM re-evaluates typical discriminative categorization, extracting generative priors from existing logic. 

* **The LogSumExp Link**:
  $$E_\theta(x) = -\log \sum_y \exp(f_\theta(x)[y])$$
* **Impact**: Combines generative optimization $P(x)$ alongside standard classification target $P(y|x)$ eliminating multi-network configurations.

## 3. The Generation Engine (SGLD)
Iterative process utilized to generate fake bounds $x_{fake}$ necessary for the Contrastive Objective.

* **Stochastic Gradient Langevin Dynamics (SGLD)**: Gradient ascent process mapped against $E(x)$, moving states toward maximum density likelihoods while simultaneously injecting Gaussian turbulence to mitigate local minima capture.
* **The Replay Buffer**: Initializing continuous gradient chains from pure uniform noise takes thousands of steps. A dynamic memory buffer seeds $95\%$ of runs from previously generated manifold artifacts ensuring rapid stability and bounds tracking.

---

## 4. Operational Diagram

```mermaid
graph LR
    R[Real Data] --> F[f_theta]
    N[Fake SGLD] --> F
    
    F --> C[Logits]
    
    C --> S[Softmax]
    C --> L[-LogSumExp]
    
    S --> P[Classification Loss p y|x]
    L --> E[Energy Representation E x]
    
    E --> V[EBM Contrastive Loss]
    
    P --> T((Total Loss))
    V --> T
```