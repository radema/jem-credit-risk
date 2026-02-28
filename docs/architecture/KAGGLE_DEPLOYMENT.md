# Kaggle Deployment Guide

This document outlines the procedure for deploying the modular JEM Credit Risk codebase to a Kaggle Code Competition environment.

## 1. Challenge: Modular Dependency in Restricted Environments
Kaggle Code Competitions run in isolated containers without internet access. Standard local project structures (with `src/`, `models/`, and `data/` folders) must be bundled and manually attached to the Kaggle Notebook.

## 2. Step-by-Step Deployment Procedure

### Step 2.1: Prepare the Local Bundle
Bundle your Python source code and trained artifacts into a single compressed archive. Run the following command from the project root:

```bash
# Bundle source code and artifacts
zip -r kaggle_jem_bundle.zip src/ models/artifacts/
```

### Step 2.2: Upload as a Kaggle Dataset
1. Log into Kaggle and navigate to **Datasets** > **New Dataset**.
2. Upload `kaggle_jem_bundle.zip`.
3. Name the dataset (e.g., `jem-credit-risk-codebase`).
4. Set the dataset to **Private**.

### Step 2.3: Attach to Kaggle Notebook
1. Open your submission notebook in the Kaggle Editor.
2. In the right-hand sidebar, click **+ Add Data**.
3. Search for your dataset `jem-credit-risk-codebase` and add it.

### Step 2.4: Initialize the Runtime Environment
Add a initialization cell at the very top of your Kaggle Notebook to register the bundled `src` modules into the Python path.

```python
import sys
import os

# Define the mount point for your dataset
BUNDLE_PATH = "/kaggle/input/jem-credit-risk-codebase"

# Add to system path to enable modular imports
if BUNDLE_PATH not in sys.path:
    sys.path.append(BUNDLE_PATH)

# Verify the bundle contents
print("Bundled files:", os.listdir(BUNDLE_PATH))
```

### Step 2.5: Configure Path Overrides
Update your inference configuration to reference the Kaggle-specific paths for competition data and your bundled artifacts.

```python
from src.data.config import DataPipelineConfig
from src.model.jem.config import JEMConfig

# Competition Data (automatically mounted by Kaggle)
KAGGLE_DATA = "/kaggle/input/home-credit-credit-risk-model-stability"

# Your Bundled Artifacts
MY_ARTIFACTS = "/kaggle/input/jem-credit-risk-codebase/models/artifacts"

test_cfg = DataPipelineConfig(
    data_dir=KAGGLE_DATA,
    sample_ratio=1.0,
    is_inference=True,
    artifact_dir=MY_ARTIFACTS
)

jem_config = JEMConfig(
    artifact_dir=MY_ARTIFACTS,
    hidden_dims=[128, 64, 32] # Must match your training run
)
```

## 3. Deployment Checklist
- [ ] **Artifact Alignment**: Ensure `hidden_dims` in `JEMConfig` exactly matches the architecture used during training.
- [ ] **GPU/Accelerator**: If using GPU for inference, ensure the Kaggle Notebook setting is toggled to **GPU T4 x2** or **GPU P100**.
- [ ] **Pathing**: Ensure all calls to `load_inference_pipeline` or `run_pipeline` use the absolute paths defined in Step 2.5.
