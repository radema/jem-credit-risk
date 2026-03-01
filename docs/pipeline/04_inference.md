# Inference Pipeline

## Overview
The Inference Engine (`src/model/jem/infer.py` or equivalent test scripts) completes the End-to-End processing chain. It simulates out-of-fold production calls: executing raw `depth_*` databases into static pipelines, imputing via a pre-calculated model state, encoding via the frozen Autoencoder, and predicting the risk distribution dynamically via the Joint Energy Model (JEM).

## Core Modules References
* `src/model/jem/infer.py` (or inference orchestration loop): Primary logic execution.
* `src/data/pipeline.py`: Used exclusively in `{ is_inference: True }` mode.
* `src/model/jem/data_utils.py`: Constructs prediction loaders (often single-epoch un-shuffled streams).

## Inference Flow

Inference mode fundamentally disables state-updates across all operations.

```mermaid
graph TD;
    subgraph Data Loading
        RawData["Raw Home Credit Test Shards (test_*.parquet)"]
    end

    subgraph Data Processing (Inference Mode)
        DataPipe["Data Pipeline<br>(pipeline.py)"]
        PreCalcImputer["Saved Imputer State<br>(imputer_state.pkl)"]
        TestFeatures["Unscaled Predict Features"]
    end

    subgraph Feature Transformation
        PreCalcScaler["Fitted TorchStandardScaler<br>(scaler_artifact.pkl)"]
        TestFeaturesScaled["Scaled Features"]
    end

    subgraph Autoencoder Mapping
        PreCalcEncoder["Trained Autoencoder<br>(encoder_weights.pt)"]
        TestLatentZ["Latent Vectors (z)"]
    end

    subgraph Joint Energy Model Simulation
        PreCalcJEM["Trained JEM Checkpoint<br>(jem_model.pt)"]
        ClassPred["Probability Output<br>p(y|z)"]
        EnergyPred["Out of Distribution Energy<br>E(z)"]
    end

    RawData --> DataPipe
    PreCalcImputer -->|Apply State (No Fit)| DataPipe
    DataPipe --> TestFeatures

    TestFeatures --> PreCalcScaler
    PreCalcScaler -->|Transform (No Fit)| TestFeaturesScaled

    TestFeaturesScaled --> PreCalcEncoder
    PreCalcEncoder -->|Encode Forward (Freeze Grads)| TestLatentZ

    TestLatentZ -->|Discriminative Forward| PreCalcJEM
    PreCalcJEM --> ClassPred
    TestLatentZ -->|Energy Output Margin| PreCalcJEM
    PreCalcJEM --> EnergyPred
```

## Key Constraints

* **Strict No-State Constraint:** Absolutely zero state modification during the pipeline operations. Everything runs via applied parameters (`.transform()` or `state_dict` imports).
    * `generate_stratified_sample()` is disabled. The model predicts 100% of the input IDs sequentially without row exclusion.
    * The streaming standard scaler explicitly utilizes `scaler.transform(batch)` instead of scaling aggregations.
* **Feature Synchronization:** Because categorical classes and numerical imputations depend on the `imputer_state.pkl`, the exact feature shapes mapped upstream in `data/pipeline.py` are strictly protected during test evaluations to ensure zero structural drift. If an incoming `test` record displays a completely unseen categorical text value, the Imputation engine will remap it down to `"Missing"/Mode` implicitly as it does not exist in the prior fitted registry.
