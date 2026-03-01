# Kaggle Deployment Guide

## 1. Constraint Overview
Kaggle Code Competitions execute within air-gapped container networks, restricting internet retrieval. Projects utilizing modular structures (`src/`, `scripts/`) must be zipped and uploaded as external Private Datasets.

## 2. Deployment Protocol

### 2.1 Artifact Bundling
Archive the relevant Python path folders and models.
```bash
zip -r kaggle_jem_bundle.zip src/ scripts/ models/artifacts/
```

### 2.2 Dataset Provisioning
1. Navigate to Kaggle > **Datasets** > **New Dataset**.
2. Upload `kaggle_jem_bundle.zip`.
3. Set visibility to **Private**.
4. Attach the new dataset iteratively to the active Kaggle inference Notebook.

### 2.3 Environmental Bootstrapping
Inject the bundled source path into Python's `sys.path`.
```python
import sys
import os

BUNDLE_PATH = "/kaggle/input/jem-credit-risk-codebase"

if BUNDLE_PATH not in sys.path:
    sys.path.append(BUNDLE_PATH)
```

### 2.4 Configuration Alignment
Define strict override variables referencing the attached Kaggle paths before invoking the orchestration scripts.

```python
from src.data.config import DataPipelineConfig
from src.model.jem.config import JEMConfig

KAGGLE_DATA = "/kaggle/input/home-credit-credit-risk-model-stability"
MY_ARTIFACTS = "/kaggle/input/jem-credit-risk-codebase/models/artifacts"

test_cfg = DataPipelineConfig(
    data_dir=KAGGLE_DATA,
    sample_ratio=1.0,
    is_inference=True,
    artifact_dir=MY_ARTIFACTS
)

# Ensure architecture matches training exactly
jem_config = JEMConfig(
    artifact_dir=MY_ARTIFACTS,
    hidden_dims=[256, 256] 
)
```

## 3. Pre-Flight Validation Checklist
* [ ] **Compute Toggle**: Verify Notebook accelerator target (e.g., **GPU T4 x2**).
* [ ] **Hidden Dimensions**: Confirm `hidden_dims` parameter perfectly mirrors the source training configuration.
* [ ] **Absolute Paths**: Utilize explicit dataset mappings rather than relative `.cache/` or `data/` links.
