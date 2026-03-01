import argparse
import logging
import polars as pl
import pickle
from pathlib import Path
from typing import List

from src.data.config import DataPipelineConfig
from src.data.export import evaluate_eda_stats, export_to_parquet
from src.data.imputation import handle_missing_and_categoricals
from src.data.loader import scan_table
from src.data.sampling import generate_stratified_sample
from src.data.aggregators import aggregate_depth_1, aggregate_depth_2
from src.data.imputation import handle_missing_and_categoricals
from src.data.export import (
    evaluate_eda_stats,
    export_to_parquet,
    export_to_chunked_parquet,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    config: DataPipelineConfig,
    depth_0_tables: list[str],
    depth_1_tables: list[str],
    depth_2_tables: list[str],
):
    """
    Orchestrates the data pipeline using an end-to-end Lazy execution graph.
    """
    is_inference = config.is_inference
    mode_str = "INFERENCE" if is_inference else "TRAINING"
    logger.info(f"--- Starting Data Pipeline Run ({mode_str} Mode) ---")

    # Prefix mapping: train -> test if in inference mode
    def map_table(t: str) -> str:
        return t.replace("train_", "test_") if is_inference else t

    base_table = map_table("train_base")
    d0_tables = [map_table(t) for t in depth_0_tables]
    d1_tables = [map_table(t) for t in depth_1_tables]
    d2_tables = [map_table(t) for t in depth_2_tables]

    cache_dir = extract_relevant_parquets(config.data_dir, config.cache_dir)

    # 1. Base Loader & Sampling
    logger.info(f"Loading Base Table: {base_table}")
    base_lazy = scan_table(base_table, cache_dir)

    if is_inference:
        logger.info("Process all case_ids for inference.")
        # No sampling in inference mode, keep case_ids from base
        valid_cases_lazy = base_lazy.select("case_id")
    else:
        # Stratified Sampling (returns a LazyFrame containing valid case_ids)
        valid_cases_lazy = generate_stratified_sample(
            base_lazy, sample_ratio=config.sample_ratio
        )

    # Base dataset filtered lazily via inner join
    final_lazy = base_lazy.join(valid_cases_lazy, on="case_id", how="inner")

    # Store depth 1 aggregations dynamically
    # Group depth-2 aggregations by parent depth-1 table
    d2_aggs_by_parent = {}

    # 2. Process Depth 2
    for d2_table in d2_tables:
        logger.info(f"Processing Depth-2 Table: {d2_table}")
        lazy_d2 = scan_table(d2_table, cache_dir)
        filtered_d2 = lazy_d2.join(valid_cases_lazy, on="case_id", how="inner")

        # Aggregate depth 2 -> depth 1 grain
        agg_d2 = aggregate_depth_2(filtered_d2)

        # Mapping depth 2 to depth 1 (e.g., 'train_person_2' -> 'train_person_1')
        parent_table = d2_table.replace("_2", "_1")
        if parent_table not in d2_aggs_by_parent:
            d2_aggs_by_parent[parent_table] = []
        d2_aggs_by_parent[parent_table].append(agg_d2)

    lazy_tables_to_join = []

    # 3. Process Depth 1
    for d1_table in d1_tables:
        logger.info(f"Processing Depth-1 Table: {d1_table}")
        lazy_d1 = scan_table(d1_table, cache_dir)
        filtered_d1 = lazy_d1.join(valid_cases_lazy, on="case_id", how="inner")

        # Join corresponding depth-2 tables BEFORE depth-1 aggregation
        if d1_table in d2_aggs_by_parent:
            logger.info(f"Integrating depth-2 features into depth-1 table: {d1_table}")
            for child_agg in d2_aggs_by_parent[d1_table]:
                filtered_d1 = filtered_d1.join(
                    child_agg, on=["case_id", "num_group1"], how="left"
                )

        # Aggregate depth 1 -> depth 0 grain
        agg_d0 = aggregate_depth_1(filtered_d1)
        lazy_tables_to_join.append((d1_table, agg_d0))

    # 4. Process Depth 0
    for d0_table in d0_tables:
        logger.info(f"Processing Depth-0 Table: {d0_table}")
        lazy_d0 = scan_table(d0_table, cache_dir)
        filtered_d0 = lazy_d0.join(valid_cases_lazy, on="case_id", how="inner")
        # Depth 0 does not need aggregation, ready for join
        lazy_tables_to_join.append((d0_table, filtered_d0))

    # 5. Final Lazy Joins
    logger.info("Performing final End-to-End lazy inner joins...")
    for table_name, table_lazy in lazy_tables_to_join:
        logger.info(
            f"Appending Lazy Node: joining {table_name} to base feature matrix."
        )
        final_lazy = final_lazy.join(table_lazy, on="case_id", how="left")

    logger.info(
        "Evaluating massive Lazy Graph via Streaming Engine. This allows optimal predicate pushdown..."
    )
    final_df = final_lazy.collect(engine="streaming")

    # 6. Imputation & Categorical Encoding
    artifact_dir = Path(config.artifact_dir)
    state_path = artifact_dir / "imputer_state.pkl"

    imputer_state = None
    if is_inference:
        logger.info(f"Loading imputer state from {state_path}...")
        if state_path.exists():
            with open(state_path, "rb") as f:
                imputer_state = pickle.load(f)
        else:
            logger.error(
                f"Imputer state NOT found at {state_path}! Proceeding without state (not recommended for inference)."
            )

    logger.info("Handling missing values and frequency encoding categoricals...")
    final_df, imputer_state = handle_missing_and_categoricals(
        final_df, state=imputer_state
    )

    if not is_inference:
        # Save Imputer state only during training
        artifact_dir.mkdir(parents=True, exist_ok=True)
        with open(state_path, "wb") as f:
            pickle.dump(imputer_state, f)
        logger.info(f"Saved imputer state to {state_path}")

    # 7. Export
    prefix = "test" if is_inference else "train"
    export_path = f"data/processed/{prefix}_features_unscaled.parquet"
    logger.info(f"Execution complete. Exporting to {export_path}")

    if not is_inference:
        evaluate_eda_stats(final_df, "data/processed/feature_statistics.log")

    if config.chunked_export and not is_inference:
        chunk_dir = "data/processed/chunks"
        chunk_paths = export_to_chunked_parquet(
            final_df, chunk_dir, chunk_size=config.chunk_size, prefix=f"{prefix}_chunk"
        )
        logger.info(f"Exported {len(chunk_paths)} chunks to {chunk_dir}")
    else:
        export_to_parquet(final_df, export_path)

    logger.info(
        f"--- Data Pipeline Run Finished Successfully (Shape: {final_df.shape}) ---"
    )
    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Data Pipeline")
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=0.05,
        help="The fraction of the data to keep (e.g., 0.05 for 5%).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = DataPipelineConfig(
        data_dir="data/raw/home-credit-credit-risk-model-stability.zip",
        sample_ratio=args.sample_ratio,
        cache_dir=".cache",
    )

    # Selecting a subset of archetype tables mimicking real execution
    run_pipeline(
        config=cfg,
        depth_0_tables=["train_static_0", "train_static_cb_0"],
        depth_1_tables=["train_credit_bureau_a_1", "train_person_1"],
        depth_2_tables=["train_person_2"],
    )
