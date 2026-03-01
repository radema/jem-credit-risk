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


def apply_case_filter(lazy_frame: pl.LazyFrame, valid_cases_df: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    """
    Filters a LazyFrame to only include rows where the case_id is present in the valid_cases_df.

    Args:
        lazy_frame: The Polars LazyFrame to filter.
        valid_cases_df: A Polars DataFrame or LazyFrame containing valid case_ids.

    Returns:
        A filtered Polars LazyFrame.
    """
    if isinstance(valid_cases_df, pl.DataFrame):
        valid_cases_df = valid_cases_df.lazy()

    # We only need the case_id column for the join
    valid_cases_df = valid_cases_df.select(["case_id"])

    # Perform an inner join on case_id to efficiently filter the lazy_frame
    return lazy_frame.join(valid_cases_df, on="case_id", how="inner")
