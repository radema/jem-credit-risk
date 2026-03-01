# Data Processing Pipeline

## Overview
The Data Processing Pipeline (`src/data/pipeline.py`) is responsible for taking raw multi-depth Home Credit parquet files and producing flattened, engineered, and imputed feature matrices for both training and inference. It heavily utilizes **Polars** for lazy execution and streaming capable data joins.

## Core Modules References
* `src/data/pipeline.py`: Main orchestrator `run_pipeline()`. Sets up the End-to-End lazy execution graph.
* `src/data/loader.py`: Exposes `scan_table()` which dynamically globs parquet chunks and natively normalizes schema drifts across chunks.
* `src/data/aggregators.py`: Implements logic for flattening `depth_2` (granular data) and `depth_1` datasets up to a common `depth_0` / `base` grain using group-by mechanisms.
* `src/data/imputation.py`: Handles stateful null-filling and categorical encoding.

## Pipeline Architecture

The pipeline processes massive nested datasets by defining a deep lazy-computation tree, reducing data volume iteratively via inner-joins *before* aggregations, and left-joining the finalized features to a central base.

```mermaid
graph TD;
    subgraph Storage layer
        Base[Base Targets Table]
        Depth2[Depth 2 Tables]
        Depth1[Depth 1 Tables]
        Depth0[Depth 0 Tables]
    end

    subgraph Pre-Filter Phase
        ValidCases[(Valid Case IDs<br><i>from Base</i>)]
        Base -->|Stratified Sample| ValidCases
    end

    subgraph Processing Nodes
        FilteredD2[Filtered Depth 2]
        AggD2[Aggregated Depth 2]
        FilteredD1[Filtered Depth 1]
        CombD1[Combined Depth 1+2]
        AggD1[Aggregated Depth 1]
        FilteredD0[Filtered Depth 0]
    end

    Depth2 -->|Scan| FilteredD2
    ValidCases -->|Inner Join| FilteredD2
    FilteredD2 -->|group_by max/last| AggD2

    Depth1 -->|Scan| FilteredD1
    ValidCases -->|Inner Join| FilteredD1
    AggD2 -->|Left Join| CombD1
    FilteredD1 -->|Left Join| CombD1

    CombD1 -->|group_by max/mean/last| AggD1

    Depth0 -->|Scan| FilteredD0
    ValidCases -->|Inner Join| FilteredD0

    subgraph Output Stream
        BaseFiltered[Filtered Base]
        Construct[Construct Final LF]
        Streaming[Stream Evaluate LF]
        Imputation[Handle Missing & Categoricals]
        Export[Export Unscaled Parquet & Chunks]
    end

    Base -->|Inner Join| BaseFiltered
    ValidCases --> BaseFiltered
    BaseFiltered --> Construct
    AggD1 -->|Left Join| Construct
    FilteredD0 -->|Left Join| Construct

    Construct --> Streaming
    Streaming --> Imputation
    Imputation --> Export
```

## Execution Modes
* **Training Mode:** Applies `.sample()` to generate random stratified training chunks, computes new statistics parameters for missing data/categoricals, and saves an `imputer_state.pkl`. Allows chunked `.parquet` exports.
* **Inference Mode:** Skips stratified sampling (processes 100% of prediction batch), automatically replaces `train_` file prefixes to `test_`, loads `imputer_state.pkl` statically to prevent leakage, and exports a single feature test set.

## Chunk Concatenation
When scanning `loader.py`, dataset chunks (e.g. `train_person_1_0`, `train_person_1_1`) are identified via Glob patterns. To prevent crashes due to "Null" chunks schema drift, it actively reads `schema.items()` and casts pure Nulls to competition standard types (`String` for `_D` date suffixes, `Float64` for `_A` amount features) before `pl.concat(..., how="vertical_relaxed")`.
