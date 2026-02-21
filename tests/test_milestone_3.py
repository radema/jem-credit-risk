import polars as pl
from src.data.config import DataPipelineConfig
from src.data.unpack import extract_relevant_parquets
from src.data.loader import scan_table
from src.data.sampling import generate_stratified_sample, apply_case_filter
from src.data.aggregators import aggregate_depth_1, join_to_base


def test_milestone_3():
    import logging

    logging.basicConfig(level=logging.INFO)

    zip_path = "data/raw/home-credit-credit-risk-model-stability.zip"
    cache_dir = ".cache_test"

    config = DataPipelineConfig(
        data_dir=zip_path,
        sample_ratio=0.0001,
        cache_dir=cache_dir,
    )

    new_dir = extract_relevant_parquets(config.data_dir, config.cache_dir)

    base_lazy = scan_table("train_base", new_dir)
    base_df = base_lazy.collect()
    valid_cases_df = generate_stratified_sample(
        base_df, sample_ratio=config.sample_ratio
    )

    bureau_lazy = scan_table("train_credit_bureau_a_1", new_dir)
    bureau_filtered = apply_case_filter(bureau_lazy, valid_cases_df)

    # Apply depth 1 aggregation
    bureau_flattened = aggregate_depth_1(bureau_filtered)
    bureau_df = bureau_flattened.collect()

    print(f"Flattened Bureau Shape: {bureau_df.shape}")
    print(f"Sampled Base Shape: {valid_cases_df.shape}")
    assert len(bureau_df) <= len(valid_cases_df), (
        "Flattened table has more rows than base!"
    )
    print("Successfully aggregated depth-1 securely.")

    # Try join
    final_lazy = join_to_base(
        base_lazy.filter(pl.col("case_id").is_in(valid_cases_df["case_id"].implode())),
        {"bureau_a": bureau_flattened},
    )
    print(f"Final Joined Schema Length: {len(final_lazy.collect_schema())}")


if __name__ == "__main__":
    test_milestone_3()
