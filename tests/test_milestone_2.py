from src.data.config import DataPipelineConfig
from src.data.unpack import extract_relevant_parquets
from src.data.loader import scan_table
from src.data.sampling import generate_stratified_sample, apply_case_filter


def test_milestone_2():
    import logging

    logging.basicConfig(level=logging.INFO)

    zip_path = "data/raw/home-credit-credit-risk-model-stability.zip"
    cache_dir = ".cache_test"

    config = DataPipelineConfig(
        data_dir=zip_path,
        sample_ratio=0.01,  # extremely small sample for quick testing
        cache_dir=cache_dir,
    )

    new_dir = extract_relevant_parquets(config.data_dir, config.cache_dir)

    print("Loading valid cases from train_base...")
    base_lazy = scan_table("train_base", new_dir)
    base_df = base_lazy.collect()

    valid_cases_df = generate_stratified_sample(
        base_df, sample_ratio=config.sample_ratio
    )

    print("Scanning train_credit_bureau_a_1...")
    bureau_a_1_lazy = scan_table("train_credit_bureau_a_1", new_dir)

    bureau_a_1_filtered = apply_case_filter(bureau_a_1_lazy, valid_cases_df)

    print("Attempting to collect filtered table...")
    filtered_df = bureau_a_1_filtered.collect()

    print(f"Filtered Table Shape: {filtered_df.shape}")
    print("Successfully filtered and saved memory!")


if __name__ == "__main__":
    test_milestone_2()
