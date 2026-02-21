import logging
from typing import List
from src.data.config import DataPipelineConfig
from src.data.unpack import extract_relevant_parquets
from src.data.loader import scan_table
from src.data.sampling import generate_stratified_sample, apply_case_filter
from src.data.aggregators import aggregate_depth_1, aggregate_depth_2, join_to_base
from src.data.export import evaluate_eda_stats, export_to_parquet

logger = logging.getLogger(__name__)


def run_pipeline(
    config: DataPipelineConfig, depth_1_tables: List[str], depth_2_tables: List[str]
):
    """
    Orchestrates the data pipeline.
    """
    logger.info("--- Starting Data Pipeline Run ---")

    cache_dir = extract_relevant_parquets(config.data_dir, config.cache_dir)

    # 1. Base Loader & Sampling
    logger.info("Loading Base Table...")
    base_lazy = scan_table("train_base", cache_dir)
    base_df = base_lazy.collect()

    # Stratified Sampling
    valid_cases_df = generate_stratified_sample(
        base_df, sample_ratio=config.sample_ratio
    )

    # Setup Dictionary for Joins
    flattened_tables = {}

    # 2. Process Depth 2
    for d2_table in depth_2_tables:
        logger.info(f"Processing Depth-2 Table: {d2_table}")
        lazy_d2 = scan_table(d2_table, cache_dir)
        filtered_d2 = apply_case_filter(lazy_d2, valid_cases_df)

        # Aggregate depth 2 -> depth 1
        d1_intermediate = aggregate_depth_2(filtered_d2)

        # Aggregate depth 1 -> depth 0
        final_d0 = aggregate_depth_1(d1_intermediate)
        flattened_tables[d2_table] = final_d0

    # 3. Process Depth 1
    for d1_table in depth_1_tables:
        logger.info(f"Processing Depth-1 Table: {d1_table}")
        lazy_d1 = scan_table(d1_table, cache_dir)
        filtered_d1 = apply_case_filter(lazy_d1, valid_cases_df)

        # Aggregate depth 1 -> depth 0
        final_d0 = aggregate_depth_1(filtered_d1)
        flattened_tables[d1_table] = final_d0

    # 4. Final Join
    logger.info("Joining all aggregated tables to the base sampled dataset...")
    # Pre-filter base_lazy to valid cases before joining so output exactly matches
    stratified_base_lazy = base_lazy.join(
        valid_cases_df.lazy(), on="case_id", how="inner"
    )

    final_lazy = join_to_base(stratified_base_lazy, flattened_tables)

    logger.info("Evaluating Lazy Evaluation Graph. This might take a few moments...")
    final_df = final_lazy.collect()

    # 5. EDA & Export
    logger.info("Execution complete. Proceeding to EDA and Export.")
    evaluate_eda_stats(final_df, "data/processed/feature_statistics.log")
    export_to_parquet(final_df, "data/processed/train_features_unscaled.parquet")

    logger.info("--- Data Pipeline Run Finished Successfully ---")
    return final_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = DataPipelineConfig(
        data_dir="data/raw/home-credit-credit-risk-model-stability.zip",
        sample_ratio=0.05,
        cache_dir=".cache",
    )

    # Selecting a subset of archetype tables mimicking real execution
    run_pipeline(
        config=cfg,
        depth_1_tables=["train_credit_bureau_a_1", "train_person_1"],
        depth_2_tables=["train_person_2"],
    )
