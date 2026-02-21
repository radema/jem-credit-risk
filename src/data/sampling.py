import polars as pl
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def generate_stratified_sample(
    df_base: pl.DataFrame,
    sample_ratio: float,
    target_col: str = "target",
    random_state: int = 42,
) -> pl.DataFrame:
    """
    Subsamples the base dataframe in memory, stratifying by the target column.

    Args:
        df_base: The base Polars DataFrame (already collected into memory).
        sample_ratio: The fraction of the data to keep (e.g. 0.05 for 5%).
        target_col: Name of the target column to stratify on.
        random_state: Seed for reproducibility.

    Returns:
        A Polars DataFrame containing ONLY the `case_id` column of the sampled valid cases.
    """
    if sample_ratio >= 1.0 or sample_ratio <= 0.0:
        logger.info(f"Sample ratio is {sample_ratio}. Returning all case_ids.")
        return df_base.select(["case_id"])

    # We only need case_id and target to perform stratification
    df_subset = df_base.select(["case_id", target_col])

    # train_test_split natively handles numpy arrays
    X = df_subset["case_id"].to_numpy()
    y = df_subset[target_col].to_numpy()

    logger.info(
        f"Performing stratified sampling with ratio {sample_ratio} on {len(X)} records."
    )

    _, X_sample, _, _ = train_test_split(
        X,
        y,
        test_size=sample_ratio,  # We want `test_size` to act as our sampled fraction
        stratify=y,
        random_state=random_state,
    )

    logger.info(f"Retained {len(X_sample)} records after sampling.")

    # Return strictly the case_ids as a DataFrame for inner joining
    return pl.DataFrame({"case_id": X_sample})


def apply_case_filter(
    lazy_df: pl.LazyFrame, valid_cases_df: pl.DataFrame
) -> pl.LazyFrame:
    """
    Given a massive LazyFrame from a depth=1 or depth=2 table, appends an
    inner_join onto the valid_cases_df so Polars can push down the filter
    predicate during scanning.
    """
    # Convert valid_cases DataFrame to LazyFrame to perform the join lazily
    return lazy_df.join(valid_cases_df.lazy(), on="case_id", how="inner")
