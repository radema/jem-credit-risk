---
status: approved
---
# Architecture: Early Stopping Mechanism

## Context and Purpose
Currently, models in `train_autoencoder.py` and `train_latent.py` train for a fixed number of epochs over multi-chunk datasets. Early Stopping monitors target metrics and interrupts training when a specified metric stops improving, saving computational resources and preventing overfitting on the training data.

## Missing Elements (Design Gaps Addressed)
To make early stopping robust, especially focusing on Validation Gini/Loss:
1. **Validation Split (The "What you are missing" part):** Gini optimization directly on training data guarantees overfitting in tree ensembles or neural networks over time. We need a way properly split chunks into training vs. validation or dedicate a holdout chunk for validation in JEM/Autoencoder loops to get a legitimate early-stopping boundary.
2. **Model Registry/ModelCheckpoint (Restoring Best Weights):** Early Stopping usually interrupts training $N$ epochs after the actual peak metric. If the model saves the final iteration, it'll save degraded weights compared to its peak iteration tracking the Early Stopping patience. An `EarlyStopping` object must cache the model weights in RAM (or a temporary `.pt` file) when a new best is reached and load them upon stopping.
3. **Patience & Min Delta Control:** Distinguishing variance in single epoch noise vs a real plateau.

## Proposed Components Architecture

### 1. `EarlyStopping` Callback Utility (in `src/utils` or `src/models/callbacks.py`)
A standalone class storing progress counters and holding weights.
- **Attributes:**
  - `patience` (int): Number of epochs with no improvement.
  - `min_delta` (float): Minimum change to qualify as an improvement.
  - `mode` (str): `"min"` (for loss) or `"max"` (for gini).
  - `best_score` (float), `best_weights` (OrderedDict or dict).
  - `counter` (int): Tracks current patience count.
- **Method `__call__(self, metric, model)`**: Evaluates the step. Returns `True` if early stopping triggers. Saves/Updates internal variables.
- **Method `restore_best_weights(self, model)`**: Applies the saved copy.

### 2. Validation Injection in the Engine Pipeline
**Autoencoder:**
If validation loss is desired, reserve ~10-20% chunks, or a specific dedicated validation file, to iterate during the epoch end to produce the `val_loss`.
If holding out data physically is problematic, we use the `train_loss` directly while monitoring small variations (`mode="min"`).

**JEM:**
We actively use ROC-AUC (Gini) and Binary Cross Entropy (MSE/BCE depending on loss function).
- To use `val_gini`, we compute the classifier logits on a validation chunk, evaluate standard performance (AUC * 2 - 1).
- Early stopping uses `mode="max"`, `metric=val_gini`.

### 3. Loop Replacements (in `train_autoencoder.py` and `train_latent.py`)
```python
early_stopping = EarlyStopping(patience=5, mode="min") # or max
for epoch in range(num_epochs):
    # Train pass (over chunks)
    # Validation pass (over chunks)
    metric = compute_target_metric(...)

    if early_stopping(metric, model):
        logger.info("Early stopping triggered. Restoring best weights...")
        early_stopping.restore_best_weights(model)
        break
```

## References (For Executing Agents)
To successfully implement this Bolt, the executing agent should interact with the following files:

1. **New Callback File:**
   - **`src/utils/callbacks.py`** (or create this path if missing): This should contain the generic, standalone `EarlyStopping` Python class utilizing `torch` parameters copying if necessary.

2. **Autoencoder Pipeline:**
   - **`src/model/autoencoder/train_autoencoder.py`**: Modify the epoch loop inside the `main` or training function to initialize `EarlyStopping` and evaluate the `train_loss` or `val_loss` against it at the end of each epoch iteration over the dataset chunks.

3. **JEM Pipeline:**
   - **`src/model/jem/train_latent.py`**: Modify the epoch loop (and potentially the inner train loop aggregation) to evaluate against `EarlyStopping`. This currently lacks straightforward validation logits logic, so calculating Gini (`roc_auc_score` from `sklearn.metrics`) at the end of the epoch will be required.

4. **Integration Testing:**
   - **`tests/model/test_callbacks.py`** (Create this): Write isolated test cases for the logic of `EarlyStopping` (counting, boolean trigger logic, best weight isolation).
   - **`tests/model/test_autoencoder.py`** and/or **`tests/model/test_train_latent.py`**: Assert that mocking metrics safely interrupts the training loop.

5. **Existing Documentation & Context:**
   - **`docs/architecture/ARCHITECTURE.md`**: For general system layout and model flow.
   - **`docs/pipeline/02_autoencoder_training.md`**: For understanding the un-supervised AE chunks loop.
   - **`docs/pipeline/03_jem_training.md`**: For understanding the Latent JEM classification loops and metrics.
