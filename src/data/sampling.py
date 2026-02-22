import polars as pl
import logging

logger = logging.getLogger(__name__)


def generate_stratified_sample(
    lazy_base: pl.LazyFrame,
    sample_ratio: float,
    target_col: str = "target",
    random_state: int = 42,
) -> pl.LazyFrame:
    """
    Subsamples the base dataset, stratifying by the target column.
    Adopts a lazy end-to-end evaluation paradigm by returning a LazyFrame.

    Args:
        lazy_base: The base Polars LazyFrame.
        sample_ratio: The fraction of the data to keep (e.g. 0.05 for 5%).
        target_col: Name of the target column to stratify on.
        random_state: Seed for reproducibility.

    Returns:
        A Polars LazyFrame containing ONLY the `case_id` column of the sampled valid cases.
    """
    if sample_ratio >= 1.0 or sample_ratio <= 0.0:
        logger.info(f"Sample ratio is {sample_ratio}. Returning all case_ids.")
        return lazy_base.select(["case_id"])

    # We only need case_id and target to perform stratification
    logger.info(
        "Collecting case_id and target columns to calculate stratified split..."
    )
    df_subset = lazy_base.select(["case_id", target_col]).collect()

    logger.info(
        f"Splitting case_ids by target value and sampling {sample_ratio * 100:.1f}% in each group..."
    )

    df_0 = df_subset.filter(pl.col(target_col) == 0).sample(
        fraction=sample_ratio, seed=random_state
    )
    df_1 = df_subset.filter(pl.col(target_col) == 1).sample(
        fraction=sample_ratio, seed=random_state
    )

    sampled_df = pl.concat([df_0, df_1])
    logger.info(f"Retained {len(sampled_df)} records after stratified sampling.")

    # Return strictly the case_ids as a LazyFrame for lazy inner joins downstream
    return sampled_df.select(["case_id"]).lazy()
